//! Average Information REML (AI-REML) for variance component estimation.
//!
//! Estimates tau = [tau_e, tau_g] by iteratively solving:
//!   tau_new = tau_old + AI^{-1} * score
//!
//! where AI is the Average Information matrix and score is the
//! REML score vector. The inner loop uses PCG to solve
//! (tau_e * W + tau_g * GRM)^{-1} * y.

use anyhow::Result;
use rayon::prelude::*;
use tracing::{debug, info, warn};

use saige_linalg::decomposition::PcgSolver;
use saige_linalg::dense::DenseMatrix;

use super::family::Family;
use super::link::TraitType;

/// Configuration for AI-REML iterations.
#[derive(Debug, Clone)]
pub struct AiRemlConfig {
    /// Maximum outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance for tau.
    pub tol: f64,
    /// PCG tolerance.
    pub pcg_tol: f64,
    /// PCG max iterations.
    pub pcg_max_iter: usize,
    /// Number of random vectors for trace estimation.
    pub n_random_vectors: u32,
    /// Whether to use sparse GRM.
    pub use_sparse_grm: bool,
    /// Seed for random number generation.
    pub seed: u64,
}

impl Default for AiRemlConfig {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 0.02,
            pcg_tol: 1e-5,
            pcg_max_iter: 500,
            n_random_vectors: 30,
            use_sparse_grm: false,
            seed: 12345,
        }
    }
}

/// Result of AI-REML estimation.
#[derive(Debug, Clone)]
pub struct AiRemlResult {
    /// Variance components [tau_e, tau_g].
    pub tau: [f64; 2],
    /// Number of iterations.
    pub iterations: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Final fitted values mu.
    pub mu: Vec<f64>,
    /// Working weights W = mu * (1-mu) for binary.
    pub working_weights: Vec<f64>,
    /// Residuals y - mu.
    pub residuals: Vec<f64>,
    /// Linear predictor eta.
    pub eta: Vec<f64>,
    /// Fixed effects coefficients alpha.
    pub alpha: Vec<f64>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
}

/// Fit the null GLMM using AI-REML.
///
/// Model: g(E[Y]) = X*alpha + b, where b ~ N(0, tau_g * GRM)
///
/// # Arguments
/// - `y`: Phenotype vector (n x 1)
/// - `x`: Design matrix (n x p), column-major
/// - `grm_vec_product`: Function computing GRM * v for a vector v
/// - `trait_type`: Binary, Quantitative, or Survival
/// - `config`: AI-REML configuration
pub fn fit_ai_reml<F>(
    y: &[f64],
    x: &DenseMatrix,
    grm_vec_product: F,
    trait_type: TraitType,
    config: &AiRemlConfig,
) -> Result<AiRemlResult>
where
    F: Fn(&[f64]) -> Vec<f64> + Send + Sync,
{
    let n = y.len();
    let p = x.ncols();
    assert_eq!(x.nrows(), n);

    let family = Family::new(trait_type);

    // For binary/survival traits, tau_e (dispersion) is fixed at 1.0.
    // Only tau_g (genetic variance) is estimated. This matches R SAIGE.
    let fix_tau_e = matches!(trait_type, TraitType::Binary | TraitType::Survival);

    // Initialize: tau_e=1 (fixed for binary), tau_g=0 (estimated from zero, matching R SAIGE)
    let mut tau = [1.0, 0.0];

    // Initial GLM fit (logistic regression without random effects)
    let mut alpha = fit_glm_irls(y, x, &family, 25)?;

    // Update eta and mu from GLM fit
    let mut eta = x.mat_vec(&alpha);
    let mut mu = family.update_mu(&eta);

    info!("Starting AI-REML with n={}, p={}", n, p);

    let pcg = PcgSolver::new(config.pcg_tol, config.pcg_max_iter);

    // Generate random vectors for trace estimation
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.seed);
    let random_vectors: Vec<Vec<f64>> = (0..config.n_random_vectors)
        .map(|_| {
            (0..n)
                .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
                .collect()
        })
        .collect();

    let mut converged = false;
    let mut n_iterations = 0;

    for iter in 0..config.max_iter {
        n_iterations = iter + 1;
        let w = family.working_weights(&mu);
        let residuals = family.residuals(y, &mu);

        // Build the Sigma operator: Sigma = tau_e * diag(1/W) + tau_g * GRM
        // In PQL, Var(Y_tilde) = tau_e * W^{-1} + tau_g * GRM
        let sigma_vec = |v: &[f64]| -> Vec<f64> {
            let grm_v = grm_vec_product(v);
            v.iter()
                .zip(w.iter())
                .zip(grm_v.iter())
                .map(|((vi, wi), gi)| tau[0] * vi / wi.max(1e-30) + tau[1] * gi)
                .collect()
        };

        // Diagonal preconditioner: diag(Sigma) ≈ tau_e/W + tau_g
        let precond = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(w.iter())
                .map(|(vi, wi)| {
                    let diag = tau[0] / wi.max(1e-30) + tau[1];
                    if diag.abs() > 1e-30 {
                        vi / diag
                    } else {
                        *vi
                    }
                })
                .collect()
        };

        // Solve for Sigma^{-1} * y_tilde using PCG
        // y_tilde = eta + W^{-1} * (y - mu)
        let y_tilde: Vec<f64> = eta
            .iter()
            .zip(w.iter())
            .zip(residuals.iter())
            .map(|((ei, wi), ri)| ei + ri / wi.max(1e-30))
            .collect();

        let pcg_result = pcg.solve(sigma_vec, precond, &y_tilde, None);
        if !pcg_result.converged {
            warn!("PCG did not converge at iteration {}", iter);
        }
        let sigma_inv_y = pcg_result.x;

        // Solve Sigma^{-1} * X columns (needed for P-projection)
        let sigma_inv_x: Vec<Vec<f64>> = (0..p)
            .map(|j| {
                let col_j = x.col(j);
                pcg.solve(&sigma_vec, &precond, &col_j, None).x
            })
            .collect();

        // X' Sigma^{-1} X
        let mut xt_sigma_inv_x = DenseMatrix::zeros(p, p);
        #[allow(clippy::needless_range_loop)]
        for j in 0..p {
            for k in j..p {
                let dot = DenseMatrix::dot(&x.col(k), &sigma_inv_x[j]);
                xt_sigma_inv_x.set(j, k, dot);
                if j != k {
                    xt_sigma_inv_x.set(k, j, dot);
                }
            }
        }

        // Solve alpha = (X'Σ⁻¹X)⁻¹ X'Σ⁻¹y
        let xt_sigma_inv_y: Vec<f64> = (0..p)
            .map(|j| DenseMatrix::dot(&x.col(j), &sigma_inv_y))
            .collect();

        let chol = match saige_linalg::decomposition::CholeskyDecomp::new(&xt_sigma_inv_x) {
            Ok(c) => c,
            Err(_) => {
                warn!("Cholesky failed at iteration {}", iter);
                break;
            }
        };
        alpha = chol.solve(&xt_sigma_inv_y);

        // Update eta and mu
        eta = x.mat_vec(&alpha);
        mu = family.update_mu(&eta);

        // P*y = Σ⁻¹y - Σ⁻¹X * (X'Σ⁻¹X)⁻¹ * X'Σ⁻¹y  =  Σ⁻¹y - Σ⁻¹X * alpha
        let mut p_y = sigma_inv_y;
        for j in 0..p {
            for i in 0..n {
                p_y[i] -= sigma_inv_x[j][i] * alpha[j];
            }
        }

        // GRM * P*y
        let grm_p_y = grm_vec_product(&p_y);

        // Score: YPAPY = (P*y)' * GRM * (P*y)
        let ypapy = DenseMatrix::dot(&p_y, &grm_p_y);

        // P-projected trace estimation: trace(P * GRM)
        // Hutchinson: E[u' * GRM * P * u] = trace(GRM * P)
        // P*u = Σ⁻¹*u - Σ⁻¹X * (X'Σ⁻¹X)⁻¹ * (Σ⁻¹X)' * u
        let trace_grm: f64 = random_vectors
            .par_iter()
            .map(|rv| {
                let sigma_inv_rv = pcg.solve(&sigma_vec, &precond, rv, None).x;

                // P*u = Σ⁻¹u - Σ⁻¹X * (X'Σ⁻¹X)⁻¹ * (Σ⁻¹X)' * u
                let sigma_ix_t_u: Vec<f64> = (0..p)
                    .map(|j| DenseMatrix::dot(&sigma_inv_x[j], rv))
                    .collect();
                let correction = chol.solve(&sigma_ix_t_u);
                let mut p_u = sigma_inv_rv;
                for j in 0..p {
                    for i in 0..n {
                        p_u[i] -= sigma_inv_x[j][i] * correction[j];
                    }
                }

                // GRM * u
                let grm_rv = grm_vec_product(rv);
                DenseMatrix::dot(&grm_rv, &p_u)
            })
            .sum::<f64>()
            / config.n_random_vectors as f64;

        let score_g = ypapy - trace_grm;

        // AI: (GRM*P*y)' * P * (GRM*P*y) — full P-projection in AI
        let sigma_inv_grm_py = pcg.solve(&sigma_vec, &precond, &grm_p_y, None).x;
        let sigma_ix_t_grm_py: Vec<f64> = (0..p)
            .map(|j| DenseMatrix::dot(&sigma_inv_x[j], &grm_p_y))
            .collect();
        let correction_grm = chol.solve(&sigma_ix_t_grm_py);
        let mut p_grm_py = sigma_inv_grm_py;
        for j in 0..p {
            for i in 0..n {
                p_grm_py[i] -= sigma_inv_x[j][i] * correction_grm[j];
            }
        }
        let ai_11 = DenseMatrix::dot(&grm_p_y, &p_grm_py);

        // For quantitative traits, also need score_e and full AI matrix
        let score_e;
        let ai_00;
        let ai_01;
        if !fix_tau_e {
            // trace(P * diag(1/W))
            let trace_w: f64 = random_vectors
                .par_iter()
                .map(|rv| {
                    let sigma_inv_rv = pcg.solve(&sigma_vec, &precond, rv, None).x;
                    let sigma_ix_t_u: Vec<f64> = (0..p)
                        .map(|j| DenseMatrix::dot(&sigma_inv_x[j], rv))
                        .collect();
                    let correction = chol.solve(&sigma_ix_t_u);
                    let mut p_u = sigma_inv_rv;
                    for j in 0..p {
                        for i in 0..n {
                            p_u[i] -= sigma_inv_x[j][i] * correction[j];
                        }
                    }
                    let inv_w_rv: Vec<f64> = rv
                        .iter()
                        .zip(w.iter())
                        .map(|(r, w)| r / w.max(1e-30))
                        .collect();
                    DenseMatrix::dot(&inv_w_rv, &p_u)
                })
                .sum::<f64>()
                / config.n_random_vectors as f64;

            let inv_w_p_y: Vec<f64> = p_y
                .iter()
                .zip(w.iter())
                .map(|(pi, wi)| pi / wi.max(1e-30))
                .collect();
            let ypa0py = DenseMatrix::dot(&p_y, &inv_w_p_y);
            score_e = ypa0py - trace_w;

            // AI entries for diag(1/W) component
            let sigma_inv_inv_w_py = pcg.solve(&sigma_vec, &precond, &inv_w_p_y, None).x;
            let sigma_ix_t_inv_w_py: Vec<f64> = (0..p)
                .map(|j| DenseMatrix::dot(&sigma_inv_x[j], &inv_w_p_y))
                .collect();
            let correction_inv_w = chol.solve(&sigma_ix_t_inv_w_py);
            let mut p_inv_w_py = sigma_inv_inv_w_py;
            for j in 0..p {
                for i in 0..n {
                    p_inv_w_py[i] -= sigma_inv_x[j][i] * correction_inv_w[j];
                }
            }
            ai_00 = DenseMatrix::dot(&inv_w_p_y, &p_inv_w_py);
            ai_01 = DenseMatrix::dot(&inv_w_p_y, &p_grm_py);
        } else {
            score_e = 0.0;
            ai_00 = 0.0;
            ai_01 = 0.0;
        }

        // R SAIGE convention: score and AI both omit the 0.5 REML factor,
        // so delta = score / AI = (YPAPY - Trace) / ((APy)'P(APy)) is correct.
        let score = [score_e, score_g];

        // Update tau: tau_new = tau_old + AI^{-1} * score
        let (tau_new, max_change) = if fix_tau_e {
            // Binary/survival: tau_e fixed at 1.0, scalar update for tau_g only
            if ai_11.abs() < 1e-30 {
                warn!("AI[1,1] is near-zero at iteration {}", iter);
                break;
            }
            let delta_1 = score[1] / ai_11;
            // Step halving: if tau_g would go negative, halve step until non-negative
            let mut step = 1.0;
            let mut tau_g_new = tau[1] + step * delta_1;
            while tau_g_new < 0.0 && step > 1e-10 {
                step *= 0.5;
                tau_g_new = tau[1] + step * delta_1;
            }
            tau_g_new = tau_g_new.max(1e-10);
            let change_1 =
                (tau_g_new - tau[1]).abs() / (tau_g_new.abs() + tau[1].abs() + config.tol);
            ([1.0, tau_g_new], change_1)
        } else {
            // Quantitative: full 2x2 AI update for both components
            let ai_det = ai_00 * ai_11 - ai_01 * ai_01;
            if ai_det.abs() < 1e-30 {
                warn!("AI matrix is singular at iteration {}", iter);
                break;
            }
            let delta_0 = (ai_11 * score[0] - ai_01 * score[1]) / ai_det;
            let delta_1 = (-ai_01 * score[0] + ai_00 * score[1]) / ai_det;
            // Step halving: if either tau would go negative, halve step
            let mut step = 1.0;
            let mut t0 = tau[0] + step * delta_0;
            let mut t1 = tau[1] + step * delta_1;
            while (t0 < 0.0 || t1 < 0.0) && step > 1e-10 {
                step *= 0.5;
                t0 = tau[0] + step * delta_0;
                t1 = tau[1] + step * delta_1;
            }
            let t = [t0.max(1e-10), t1.max(1e-10)];
            let change_0 = (t[0] - tau[0]).abs() / (t[0].abs() + tau[0].abs() + config.tol);
            let change_1 = (t[1] - tau[1]).abs() / (t[1].abs() + tau[1].abs() + config.tol);
            (t, change_0.max(change_1))
        };

        debug!(
            "AI-REML iter {}: tau=[{:.6}, {:.6}], change={:.2e}{}",
            iter,
            tau_new[0],
            tau_new[1],
            max_change,
            if fix_tau_e { " (tau_e fixed)" } else { "" }
        );

        tau = tau_new;

        if max_change < config.tol {
            info!("AI-REML converged after {} iterations", iter + 1);
            converged = true;
            break;
        }
    }

    if !converged {
        warn!(
            "AI-REML did not converge after {} iterations",
            config.max_iter
        );
    }

    let w = family.working_weights(&mu);
    let residuals = family.residuals(y, &mu);

    // Compute observation-level log-likelihood
    let log_likelihood = match trait_type {
        TraitType::Binary | TraitType::Survival => {
            // Bernoulli log-likelihood: sum(y*log(mu) + (1-y)*log(1-mu))
            y.iter()
                .zip(mu.iter())
                .map(|(&yi, &mi)| {
                    let mi = mi.clamp(1e-15, 1.0 - 1e-15);
                    yi * mi.ln() + (1.0 - yi) * (1.0 - mi).ln()
                })
                .sum::<f64>()
        }
        TraitType::Quantitative => {
            // Gaussian log-likelihood: -n/2*log(2*pi*tau_e) - sum((y-mu)^2)/(2*tau_e)
            let n_f = n as f64;
            let ss: f64 = residuals.iter().map(|r| r * r).sum();
            -0.5 * n_f * (2.0 * std::f64::consts::PI * tau[0]).ln() - 0.5 * ss / tau[0]
        }
    };

    Ok(AiRemlResult {
        tau,
        iterations: n_iterations,
        converged,
        mu,
        working_weights: w,
        residuals,
        eta,
        alpha,
        log_likelihood,
    })
}

/// Fit a GLM using IRLS (Iteratively Reweighted Least Squares).
/// Used for initialization before AI-REML.
fn fit_glm_irls(y: &[f64], x: &DenseMatrix, family: &Family, max_iter: usize) -> Result<Vec<f64>> {
    let _n = y.len();
    let p = x.ncols();

    let mut mu = family.initialize_mu(y);
    let mut eta = family.link(&mu);
    let mut alpha = vec![0.0; p];

    for _iter in 0..max_iter {
        let w = family.working_weights(&mu);
        let z: Vec<f64> = eta
            .iter()
            .zip(y.iter())
            .zip(mu.iter())
            .zip(w.iter())
            .map(|(((ei, yi), mi), wi)| ei + (yi - mi) / wi.max(1e-30))
            .collect();

        // Weighted least squares: (X'WX)^{-1} X'Wz
        let xtwx = x.xtwx(&w);
        let xtwz = x.xtwv(&w, &z);

        let alpha_new = match saige_linalg::decomposition::CholeskyDecomp::new(&xtwx) {
            Ok(chol) => chol.solve(&xtwz),
            Err(_) => {
                // Fallback: add small diagonal
                let mut xtwx_reg = xtwx.clone();
                for i in 0..p {
                    xtwx_reg.set(i, i, xtwx_reg.get(i, i) + 1e-6);
                }
                saige_linalg::decomposition::CholeskyDecomp::new(&xtwx_reg)?.solve(&xtwz)
            }
        };

        eta = x.mat_vec(&alpha_new);
        mu = family.update_mu(&eta);

        let change: f64 = alpha_new
            .iter()
            .zip(alpha.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f64>();

        alpha = alpha_new;

        if change < 1e-8 {
            break;
        }
    }

    Ok(alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm_irls_quantitative() {
        // Simple linear regression: y = 1 + 2*x + noise
        let n = 100;
        let mut y = Vec::with_capacity(n);
        let mut x_data = vec![0.0; n * 2]; // intercept + one covariate

        for i in 0..n {
            let xi = i as f64 / n as f64;
            x_data[i] = 1.0; // intercept
            x_data[n + i] = xi; // covariate
            y.push(1.0 + 2.0 * xi);
        }

        let x = DenseMatrix::from_col_major(n, 2, x_data);
        let family = Family::new(TraitType::Quantitative);
        let alpha = fit_glm_irls(&y, &x, &family, 25).unwrap();

        assert!((alpha[0] - 1.0).abs() < 0.1, "intercept: {}", alpha[0]);
        assert!((alpha[1] - 2.0).abs() < 0.1, "slope: {}", alpha[1]);
    }

    #[test]
    fn test_glm_irls_binary() {
        // Logistic regression: ~50% prevalence with mild covariate effect
        let n = 200;
        let mut y = Vec::with_capacity(n);
        let mut x_data = vec![0.0; n * 2];

        use rand::Rng;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);

        for i in 0..n {
            let xi = (i as f64 / n as f64) * 2.0 - 1.0; // range [-1, 1]
            x_data[i] = 1.0;
            x_data[n + i] = xi;
            // P(Y=1) = logistic(0.5 * xi) - mild effect, no separation
            let prob = 1.0 / (1.0 + (-0.5 * xi).exp());
            y.push(if rng.gen::<f64>() < prob { 1.0 } else { 0.0 });
        }

        let x = DenseMatrix::from_col_major(n, 2, x_data);
        let family = Family::new(TraitType::Binary);
        let alpha = fit_glm_irls(&y, &x, &family, 50).unwrap();

        // Coefficients should be finite and reasonable
        assert!(alpha[0].is_finite(), "intercept: {}", alpha[0]);
        assert!(alpha[1].is_finite(), "slope: {}", alpha[1]);
        assert!(alpha[0].abs() < 3.0, "intercept too large: {}", alpha[0]);
        assert!(alpha[1].abs() < 5.0, "slope too large: {}", alpha[1]);
    }

    #[test]
    fn test_ai_reml_quantitative_small() {
        // Simple quantitative model with identity GRM
        let n = 20;
        let mut y = Vec::with_capacity(n);
        let mut x_data = vec![0.0; n]; // intercept only

        for (i, xi) in x_data.iter_mut().enumerate().take(n) {
            *xi = 1.0;
            y.push(1.0 + (i as f64 * 0.1));
        }

        let x = DenseMatrix::from_col_major(n, 1, x_data);
        let grm_vec = |v: &[f64]| -> Vec<f64> { v.to_vec() }; // identity GRM

        let config = AiRemlConfig {
            max_iter: 10,
            tol: 1e-4,
            pcg_tol: 1e-4,
            pcg_max_iter: 100,
            n_random_vectors: 5,
            use_sparse_grm: false,
            seed: 42,
        };

        let result = fit_ai_reml(&y, &x, grm_vec, TraitType::Quantitative, &config).unwrap();

        // tau should be positive
        assert!(result.tau[0] > 0.0, "tau_e={}", result.tau[0]);
        assert!(result.tau[1] >= 0.0, "tau_g={}", result.tau[1]);

        // iterations should be > 0 and <= max_iter
        assert!(result.iterations > 0, "iterations={}", result.iterations);
        assert!(
            result.iterations <= config.max_iter,
            "iterations={}",
            result.iterations
        );

        // log-likelihood should be finite and negative (Gaussian)
        assert!(
            result.log_likelihood.is_finite(),
            "loglik={}",
            result.log_likelihood
        );
        assert!(
            result.log_likelihood < 0.0,
            "loglik should be negative: {}",
            result.log_likelihood
        );
    }
}
