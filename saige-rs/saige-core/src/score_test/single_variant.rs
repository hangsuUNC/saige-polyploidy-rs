//! Single-variant score test.
//!
//! Computes the score test statistic for a single genetic variant:
//!   g_tilde = g - X * (X'VX)^{-1} * X'V * g
//!   S = g_tilde' * residuals / tau_e
//!   var = g_tilde' * diag(mu2 * tau_e) * g_tilde, adjusted by variance ratio
//!   T = S^2 / var ~ chi-sq(1)
//!
//! Reference: SAIGE src/SAIGE_test.cpp

use anyhow::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use saige_linalg::dense::DenseMatrix;

use crate::glmm::link::TraitType;
use crate::glmm::variance_ratio::select_variance_ratio;
use crate::spa::binary::spa_binary;
use crate::spa::fast::spa_binary_fast;

/// Result of a single-variant score test.
#[derive(Debug, Clone)]
pub struct SingleVariantResult {
    /// Marker ID.
    pub marker_id: String,
    /// Chromosome.
    pub chrom: String,
    /// Position.
    pub pos: u64,
    /// Reference allele (Allele1 in SAIGE output).
    pub ref_allele: String,
    /// Alternative allele (Allele2 in SAIGE output).
    pub alt_allele: String,
    /// Allele count of alt allele.
    pub ac: f64,
    /// Allele frequency of alt allele.
    pub af: f64,
    /// Minor allele count.
    pub mac: f64,
    /// Imputation info score (1.0 for hard-called genotypes).
    pub info: f64,
    /// Number of valid samples.
    pub n: usize,
    /// Score statistic.
    pub score: f64,
    /// Variance of score statistic (varT).
    pub var_t: f64,
    /// Variance adjusted by VR (varTstar).
    pub var_t_star: f64,
    /// Chi-squared test statistic.
    pub tstat: f64,
    /// P-value (from chi-squared or SPA).
    pub pvalue: f64,
    /// P-value without SPA adjustment.
    pub pvalue_noadj: f64,
    /// Whether SPA was applied.
    pub is_spa: bool,
    /// Whether SPA converged.
    pub spa_converged: bool,
    /// Beta (effect size estimate).
    pub beta: f64,
    /// Standard error of beta.
    pub se_beta: f64,
    /// Variance ratio used.
    pub variance_ratio: f64,
    /// AF in cases (binary only).
    pub af_cases: f64,
    /// AF in controls (binary only).
    pub af_controls: f64,
    /// N cases (binary only).
    pub n_cases: usize,
    /// N controls (binary only).
    pub n_controls: usize,
}

/// Engine for running score tests. Initialized from a null model.
pub struct ScoreTestEngine {
    /// Trait type.
    pub trait_type: TraitType,
    /// Fitted values mu from null model.
    pub mu: Vec<f64>,
    /// mu * (1 - mu) for binary, 1 for quantitative.
    pub mu2: Vec<f64>,
    /// Residuals (y - mu).
    pub residuals: Vec<f64>,
    /// Variance component tau_e.
    pub tau_e: f64,
    /// Variance component tau_g.
    pub tau_g: f64,
    /// Precomputed (X'VX)^{-1} * X'V of shape (p x n).
    pub xvx_inv_xv: DenseMatrix,
    /// Design matrix X (n x p).
    pub x: DenseMatrix,
    /// Overall variance ratio.
    pub variance_ratio: f64,
    /// Categorical variance ratios.
    pub categorical_vr: Vec<(f64, f64)>,
    /// Whether to apply SPA.
    pub use_spa: bool,
    /// Whether to use fast SPA.
    pub use_fast_spa: bool,
    /// SPA tolerance.
    pub spa_tol: f64,
    /// SPA p-value threshold (only apply SPA if p < this).
    pub spa_pval_cutoff: f64,
    /// Original y values (for computing case/control AF).
    pub y: Option<Vec<f64>>,
}

impl ScoreTestEngine {
    /// Run the score test for a single marker.
    pub fn test_marker(
        &self,
        g: &[f64],
        marker_id: &str,
        chrom: &str,
        pos: u64,
        ref_allele: &str,
        alt_allele: &str,
    ) -> Result<SingleVariantResult> {
        let n = self.mu.len();
        assert_eq!(g.len(), n);

        // Compute allele frequency and count.
        // The copy-number maximum ("ploidy") is inferred per marker as
        // max(2.0, max non-missing dosage) so AF stays in [0, 1] and MAC stays
        // non-negative for CNV / polyploid dosages (> 2). For ordinary diploid
        // markers ploidy == 2.0, recovering the original 2N behavior.
        let mut sum = 0.0;
        let mut n_valid = 0;
        let mut ploidy = 2.0_f64;
        for &gi in g {
            if !gi.is_nan() {
                sum += gi;
                n_valid += 1;
                if gi > ploidy {
                    ploidy = gi;
                }
            }
        }
        let total = ploidy * n_valid as f64;
        let af = if n_valid > 0 { sum / total } else { 0.0 };
        let ac = sum;
        let mac = if n_valid > 0 { ac.min(total - ac) } else { 0.0 };

        // Compute g_tilde = g - X * (X'VX)^{-1} * X'V * g
        let xvx_inv_xv_g = self.xvx_inv_xv.mat_vec(g);
        let x_proj = self.x.mat_vec(&xvx_inv_xv_g);
        let g_tilde: Vec<f64> = g
            .iter()
            .zip(x_proj.iter())
            .map(|(gi, xi)| gi - xi)
            .collect();

        // Score statistic: S = g_tilde' * residuals / tau_e
        // Matches R SAIGE: S = dot(t_gtilde, m_res) / m_tauvec[0]
        let score_raw: f64 = g_tilde
            .iter()
            .zip(self.residuals.iter())
            .map(|(gi, ri)| gi * ri)
            .sum();
        let score = score_raw / self.tau_e;

        // Select variance ratio
        let vr = select_variance_ratio(mac, self.variance_ratio, &self.categorical_vr);

        // Variance: var_t = g_tilde' * diag(mu2 * tau_e) * g_tilde
        // Matches R SAIGE: t_P2Vec = t_gtilde % m_mu2 * m_tauvec[0]; var2 = dot(t_P2Vec, t_gtilde)
        let var_t: f64 = g_tilde
            .iter()
            .zip(self.mu2.iter())
            .map(|(gi, mi)| gi * gi * mi)
            .sum::<f64>()
            * self.tau_e;

        // Adjusted variance: var_t_star = var_t * vr
        // Matches R SAIGE: var1 = var2 * m_varRatioVal
        let var_t_star = var_t * vr;

        // Test statistic: T = S^2 / var_t_star
        let tstat = if var_t_star > 1e-30 {
            score * score / var_t_star
        } else {
            0.0
        };

        // P-value from chi-squared(1)
        let chi2 = ChiSquared::new(1.0).unwrap();
        let pvalue_noadj = 1.0 - chi2.cdf(tstat);

        // Apply SPA for binary traits if needed
        let (pvalue, is_spa, spa_converged) = if self.use_spa
            && self.trait_type == TraitType::Binary
            && pvalue_noadj < self.spa_pval_cutoff
        {
            // SPA q-value: q = S / sqrt(vr) + m1
            // Matches R SAIGE: q = t_Tstat / sqrt(t_var1/t_var2) + m1
            let m1: f64 = self
                .mu
                .iter()
                .zip(g_tilde.iter())
                .map(|(mi, gi)| mi * gi)
                .sum();
            let q = score / vr.sqrt() + m1;
            // Mirror q around m1 for two-sided test
            let qinv = 2.0 * m1 - q;

            let spa_result = if self.use_fast_spa {
                spa_binary_fast(&self.mu, &g_tilde, q, qinv, pvalue_noadj, self.spa_tol)
            } else {
                spa_binary(&self.mu, &g_tilde, q, qinv, pvalue_noadj, self.spa_tol)
            };
            (spa_result.pvalue, spa_result.is_spa, spa_result.converged)
        } else {
            (pvalue_noadj, false, true)
        };

        // Beta = S / var1, SE = |Beta| / sqrt(|stat|) = 1/sqrt(var1)
        // Matches R SAIGE: t_Beta = S/var1; t_seBeta = fabs(t_Beta) / sqrt(fabs(stat))
        let beta = if var_t_star > 1e-30 {
            score / var_t_star
        } else {
            0.0
        };
        let se_beta = if var_t_star > 1e-30 {
            1.0 / var_t_star.sqrt()
        } else {
            f64::INFINITY
        };

        // Case/control AF (binary traits only)
        let (af_cases, af_controls, n_cases, n_controls) = if let Some(ref y) = self.y {
            compute_case_control_af(g, y, ploidy)
        } else {
            (f64::NAN, f64::NAN, 0, 0)
        };

        Ok(SingleVariantResult {
            marker_id: marker_id.to_string(),
            chrom: chrom.to_string(),
            pos,
            ref_allele: ref_allele.to_string(),
            alt_allele: alt_allele.to_string(),
            ac,
            af,
            mac,
            info: 1.0, // hard-called genotypes
            n: n_valid,
            score,
            var_t,
            var_t_star,
            tstat,
            pvalue,
            pvalue_noadj,
            is_spa,
            spa_converged,
            beta,
            se_beta,
            variance_ratio: vr,
            af_cases,
            af_controls,
            n_cases,
            n_controls,
        })
    }
}

/// Compute allele frequency separately for cases and controls.
fn compute_case_control_af(g: &[f64], y: &[f64], ploidy: f64) -> (f64, f64, usize, usize) {
    let mut sum_cases = 0.0;
    let mut n_cases = 0usize;
    let mut sum_controls = 0.0;
    let mut n_controls = 0usize;

    for (gi, yi) in g.iter().zip(y.iter()) {
        if gi.is_nan() {
            continue;
        }
        if *yi > 0.5 {
            sum_cases += gi;
            n_cases += 1;
        } else {
            sum_controls += gi;
            n_controls += 1;
        }
    }

    // Normalize by the per-marker copy-number max (2.0 for diploid) so
    // case/control AF are comparable to the overall AF and stay in [0, 1].
    let af_cases = if n_cases > 0 {
        sum_cases / (ploidy * n_cases as f64)
    } else {
        f64::NAN
    };
    let af_controls = if n_controls > 0 {
        sum_controls / (ploidy * n_controls as f64)
    } else {
        f64::NAN
    };

    (af_cases, af_controls, n_cases, n_controls)
}

/// Write association results header to match SAIGE's format.
///
/// SAIGE uses space-separated output with specific column names.
pub fn write_results_header(writer: &mut impl std::io::Write) -> Result<()> {
    writeln!(
        writer,
        "CHR POS SNPID Allele1 Allele2 AC_Allele2 AF_Allele2 imputationInfo N BETA SE Tstat p.value p.value.NA Is.SPA.converge varT varTstar AF.Cases AF.Controls N.Cases N.Controls"
    )?;
    Ok(())
}

/// Write a single result line in SAIGE's format.
pub fn write_result_line(
    writer: &mut impl std::io::Write,
    result: &SingleVariantResult,
) -> Result<()> {
    writeln!(
        writer,
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        result.chrom,
        result.pos,
        result.marker_id,
        result.ref_allele,
        result.alt_allele,
        result.ac,
        result.af,
        result.info,
        result.n,
        result.beta,
        result.se_beta,
        result.tstat,
        result.pvalue,
        result.pvalue_noadj,
        if result.spa_converged { 1 } else { 0 },
        result.var_t,
        result.var_t_star,
        if result.af_cases.is_nan() {
            "NA".to_string()
        } else {
            format!("{}", result.af_cases)
        },
        if result.af_controls.is_nan() {
            "NA".to_string()
        } else {
            format!("{}", result.af_controls)
        },
        result.n_cases,
        result.n_controls,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_test_basic() {
        let n = 10;
        let mu = vec![0.5; n];
        let mu2 = vec![0.25; n]; // mu * (1-mu)
        let residuals = vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let x = DenseMatrix::from_col_major(n, 1, vec![1.0; n]); // intercept only
        let xvx_inv_xv = DenseMatrix::from_col_major(1, n, vec![1.0 / n as f64; n]);

        let engine = ScoreTestEngine {
            trait_type: TraitType::Binary,
            mu: mu.clone(),
            mu2,
            residuals,
            tau_e: 1.0,
            tau_g: 0.1,
            xvx_inv_xv,
            x,
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            use_spa: false,
            use_fast_spa: false,
            spa_tol: 1e-6,
            spa_pval_cutoff: 0.05,
            y: None,
        };

        let g = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0];
        let result = engine.test_marker(&g, "rs1", "1", 100, "A", "T").unwrap();

        assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0);
        assert!(result.af >= 0.0 && result.af <= 1.0);
        assert!(result.var_t > 0.0);
        assert!(result.var_t_star > 0.0);
    }

    #[test]
    fn test_cnv_dosage_valid_af_and_finite_beta() {
        // CNV dosages in [0, 8] (most samples at 2). The old 2N denominator
        // produced AF > 1 and negative MAC; the inferred ploidy must fix both,
        // and the reported beta must stay finite.
        let n = 10;
        let mu = vec![0.5; n];
        let mu2 = vec![0.25; n];
        let residuals = vec![0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5, -0.5];
        let x = DenseMatrix::from_col_major(n, 1, vec![1.0; n]);
        let xvx_inv_xv = DenseMatrix::from_col_major(1, n, vec![1.0 / n as f64; n]);

        let engine = ScoreTestEngine {
            trait_type: TraitType::Binary,
            mu: mu.clone(),
            mu2,
            residuals,
            tau_e: 1.0,
            tau_g: 0.1,
            xvx_inv_xv,
            x,
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            use_spa: false,
            use_fast_spa: false,
            spa_tol: 1e-6,
            spa_pval_cutoff: 0.05,
            y: None,
        };

        let g = vec![2.0, 2.0, 4.0, 2.0, 8.0, 2.0, 2.0, 6.0, 2.0, 2.0];
        let result = engine
            .test_marker(&g, "cnv1", "chrM", 100, "A", "G")
            .unwrap();

        assert!((0.0..=1.0).contains(&result.af), "af={}", result.af);
        assert!(result.mac >= 0.0, "mac={}", result.mac);
        assert!(result.beta.is_finite(), "beta={}", result.beta);
    }

    #[test]
    fn test_case_control_af() {
        let g = vec![0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 0.0];
        let y = vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let (af_cases, af_controls, n_cases, n_controls) = compute_case_control_af(&g, &y, 2.0);

        assert_eq!(n_cases, 5);
        assert_eq!(n_controls, 5);
        // Cases: (0+1+2+0+1) / (2*5) = 4/10 = 0.4
        assert!((af_cases - 0.4).abs() < 1e-10, "af_cases={}", af_cases);
        // Controls: (0+2+1+0+0) / (2*5) = 3/10 = 0.3
        assert!(
            (af_controls - 0.3).abs() < 1e-10,
            "af_controls={}",
            af_controls
        );
    }
}
