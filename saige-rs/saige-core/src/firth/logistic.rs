//! Firth's penalized logistic regression.
//!
//! Modified Newton-Raphson with Firth's penalty:
//!   U_j^* = U_j + 0.5 * trace(I^{-1} * dI/d(beta_j))
//! which in practice adds h_i * (0.5 - pi_i) to the score function,
//! where h_i is the i-th diagonal of the hat matrix H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}.
//!
//! Reference: SAIGE R/Firth.R

use anyhow::Result;
use saige_linalg::decomposition::{CholeskyDecomp, QrDecomp};
use saige_linalg::dense::DenseMatrix;

/// Configuration for Firth's logistic regression.
#[derive(Debug, Clone)]
pub struct FirthConfig {
    /// Maximum Newton-Raphson iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
    /// L2 regularization parameter (optional).
    pub l2_penalty: f64,
}

impl Default for FirthConfig {
    fn default() -> Self {
        Self {
            max_iter: 25,
            tol: 1e-5,
            l2_penalty: 0.0,
        }
    }
}

/// Result of Firth's logistic regression.
#[derive(Debug, Clone)]
pub struct FirthResult {
    /// Estimated coefficients.
    pub beta: Vec<f64>,
    /// Standard errors.
    pub se: Vec<f64>,
    /// P-value for the last coefficient (the genetic variant).
    pub pvalue: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Hat matrix diagonal values.
    pub hat_diag: Vec<f64>,
}

/// Fit Firth's penalized logistic regression.
///
/// # Arguments
/// - `y`: Binary outcome (0/1), length n
/// - `x`: Design matrix (n x p), includes intercept and covariates
/// - `config`: Firth configuration
pub fn firth_logistic(y: &[f64], x: &DenseMatrix, config: &FirthConfig) -> Result<FirthResult> {
    let n = y.len();
    let p = x.ncols();
    assert_eq!(x.nrows(), n);

    // Initialize coefficients to zero
    let mut beta = vec![0.0; p];

    for iter in 0..config.max_iter {
        // Compute linear predictor and fitted values
        let eta = x.mat_vec(&beta);
        let mu: Vec<f64> = eta.iter().map(|&e| logistic(e)).collect();

        // Working weights W = mu * (1 - mu)
        let w: Vec<f64> = mu
            .iter()
            .map(|&m| {
                let v = m * (1.0 - m);
                v.max(1e-10)
            })
            .collect();

        // Compute hat matrix diagonal via QR
        // H = W^{1/2} X (X'WX)^{-1} X' W^{1/2}
        // hat_i = h_ii
        let w_sqrt: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();

        // W^{1/2} X
        let mut wx = DenseMatrix::zeros(n, p);
        for j in 0..p {
            for (i, &ws) in w_sqrt.iter().enumerate().take(n) {
                wx.set(i, j, ws * x.get(i, j));
            }
        }

        let hat_diag = match compute_hat_diagonal(&wx) {
            Ok(h) => h,
            Err(_) => vec![0.0; n],
        };

        // Firth-modified score: U_j = X' * (y - mu + h * (0.5 - mu))
        let modified_residuals: Vec<f64> = (0..n)
            .map(|i| y[i] - mu[i] + hat_diag[i] * (0.5 - mu[i]))
            .collect();

        // Score vector
        let score = x.xtwv(&vec![1.0; n], &modified_residuals);

        // Information matrix: X'WX + L2 penalty
        let mut info = x.xtwx(&w);
        if config.l2_penalty > 0.0 {
            for j in 0..p {
                info.set(j, j, info.get(j, j) + config.l2_penalty);
            }
        }

        // Newton step: delta = (X'WX)^{-1} * score
        let delta = match CholeskyDecomp::new(&info) {
            Ok(chol) => chol.solve(&score),
            Err(_) => {
                // Add regularization and retry
                for j in 0..p {
                    info.set(j, j, info.get(j, j) + 1e-6);
                }
                CholeskyDecomp::new(&info)?.solve(&score)
            }
        };

        // Update beta
        let mut max_change = 0.0_f64;
        for j in 0..p {
            beta[j] += delta[j];
            max_change = max_change.max(delta[j].abs());
        }

        if max_change < config.tol {
            let (se, pvalue) = firth_se_pvalue(x, &beta);
            return Ok(FirthResult {
                beta,
                se,
                pvalue,
                iterations: iter + 1,
                converged: true,
                hat_diag,
            });
        }
    }

    // Did not fully converge within max_iter. Still return the current
    // (bias-reduced) estimates, with SE/p-value computed at the final iterate,
    // so callers can report a finite Firth beta instead of falling back to an
    // ill-conditioned score-test beta. `converged` is false to flag this.
    let (se, pvalue) = firth_se_pvalue(x, &beta);
    Ok(FirthResult {
        beta,
        se,
        pvalue,
        iterations: config.max_iter,
        converged: false,
        hat_diag: vec![0.0; n],
    })
}

/// Compute standard errors (from the unpenalized information matrix `X'WX`)
/// and the two-sided Wald p-value for the last coefficient, evaluated at
/// `beta`. Shared by the converged and non-converged return paths.
fn firth_se_pvalue(x: &DenseMatrix, beta: &[f64]) -> (Vec<f64>, f64) {
    let p = x.ncols();
    let eta = x.mat_vec(beta);
    let mu: Vec<f64> = eta.iter().map(|&e| logistic(e)).collect();
    let w: Vec<f64> = mu.iter().map(|&m| (m * (1.0 - m)).max(1e-10)).collect();
    let info = x.xtwx(&w);

    let se: Vec<f64> = match CholeskyDecomp::new(&info) {
        Ok(chol) => {
            let inv = chol.inverse();
            (0..p).map(|j| inv.get(j, j).max(0.0).sqrt()).collect()
        }
        Err(_) => vec![f64::NAN; p],
    };

    let z = if se[p - 1].is_finite() && se[p - 1] > 1e-30 {
        beta[p - 1] / se[p - 1]
    } else {
        0.0
    };
    use statrs::distribution::{ContinuousCDF, Normal};
    let norm = Normal::new(0.0, 1.0).unwrap();
    let pvalue = 2.0 * (1.0 - norm.cdf(z.abs()));

    (se, pvalue)
}

/// Logistic function: 1 / (1 + exp(-x))
#[inline]
fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the diagonal of the hat matrix H = A * (A'A)^{-1} * A'
/// where A = W^{1/2} * X.
fn compute_hat_diagonal(wx: &DenseMatrix) -> Result<Vec<f64>> {
    let n = wx.nrows();
    let p = wx.ncols();

    // QR decomposition of W^{1/2}X
    let qr = QrDecomp::new(wx)?;

    // H = Q * Q', so h_ii = sum_j Q[i,j]^2
    let mut hat_diag = vec![0.0; n];
    for j in 0..p {
        for (i, hd) in hat_diag.iter_mut().enumerate().take(n) {
            let q_ij = qr.q.get(i, j);
            *hd += q_ij * q_ij;
        }
    }

    Ok(hat_diag)
}

/// Firth test for a genetic variant, adjusting for covariates.
///
/// Fits two models:
///   H0: y ~ covariates (without the variant)
///   H1: y ~ covariates + genotype
///
/// Returns the p-value from a likelihood ratio test or Wald test.
pub fn firth_test_variant(
    y: &[f64],
    covariates: &DenseMatrix,
    genotype: &[f64],
    config: &FirthConfig,
) -> Result<FirthResult> {
    let n = y.len();
    let p_covars = covariates.ncols();

    // Standardize the genotype column (center and scale to unit variance)
    // before fitting. A near-constant predictor (e.g. a CNV dosage where almost
    // everyone has the same copy number) makes X'WX near-singular in the
    // genotype term and destabilizes Newton-Raphson. Standardizing conditions
    // the problem; the coefficient and SE are back-transformed afterward, and
    // the Wald z (beta/se) — hence the p-value — is invariant to this rescaling.
    let g_mean: f64 = genotype.iter().take(n).sum::<f64>() / n as f64;
    let g_var: f64 = genotype
        .iter()
        .take(n)
        .map(|&gi| (gi - g_mean).powi(2))
        .sum::<f64>()
        / n as f64;
    let g_sd = g_var.sqrt();
    let standardize = g_sd > 1e-8;

    // Build full design matrix: [covariates | genotype]
    let mut x_full = DenseMatrix::zeros(n, p_covars + 1);
    for j in 0..p_covars {
        for i in 0..n {
            x_full.set(i, j, covariates.get(i, j));
        }
    }
    for (i, &gi) in genotype.iter().enumerate().take(n) {
        let val = if standardize {
            (gi - g_mean) / g_sd
        } else {
            gi
        };
        x_full.set(i, p_covars, val);
    }

    let mut result = firth_logistic(y, &x_full, config)?;

    // Back-transform the genotype coefficient and SE to the original dosage
    // scale (per-copy effect). Centering only shifts the intercept, which is
    // not reported, so no further adjustment is needed.
    if standardize {
        let last = result.beta.len() - 1;
        result.beta[last] /= g_sd;
        if result.se[last].is_finite() {
            result.se[last] /= g_sd;
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logistic_function() {
        assert!((logistic(0.0) - 0.5).abs() < 1e-10);
        assert!(logistic(10.0) > 0.999);
        assert!(logistic(-10.0) < 0.001);
    }

    #[test]
    fn test_firth_basic() {
        let n = 50;
        let y: Vec<f64> = (0..n).map(|i| if i < 25 { 1.0 } else { 0.0 }).collect();
        let mut x_data = vec![0.0; n * 2];
        for i in 0..n {
            x_data[i] = 1.0; // intercept
            x_data[n + i] = if i < 15 || (25..35).contains(&i) {
                1.0
            } else {
                0.0
            }; // genotype
        }
        let x = DenseMatrix::from_col_major(n, 2, x_data);

        let config = FirthConfig::default();
        let result = firth_logistic(&y, &x, &config).unwrap();

        assert!(result.converged, "Firth did not converge");
        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
    }

    #[test]
    fn test_firth_separation() {
        // Complete separation: all cases have genotype=1
        let n = 20;
        let y: Vec<f64> = (0..n).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();
        let mut x_data = vec![0.0; n * 2];
        for i in 0..n {
            x_data[i] = 1.0; // intercept
            x_data[n + i] = if i < 5 { 1.0 } else { 0.0 }; // genotype = case indicator
        }
        let x = DenseMatrix::from_col_major(n, 2, x_data);

        let config = FirthConfig::default();
        let result = firth_logistic(&y, &x, &config).unwrap();

        // Firth should still produce a finite estimate (unlike standard logistic)
        assert!(
            result.beta[1].is_finite(),
            "Firth beta should be finite even with separation"
        );
    }
}
