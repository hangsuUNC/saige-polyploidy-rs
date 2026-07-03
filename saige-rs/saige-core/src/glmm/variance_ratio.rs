//! Variance ratio estimation for calibrating score test statistics.
//!
//! The variance ratio adjusts the score test variance to account for
//! the approximation made by using the working model instead of the
//! full mixed model. It is estimated by comparing the exact variance
//! (from the mixed model) with the approximate variance (from the
//! working model) across a set of randomly selected markers.
//!
//! Algorithm (from SAIGE R/SAIGE_fitGLMM_fast.R):
//!   For each marker G0:
//!     1. Flip alleles if AF > 0.5: with P = max(2, max dosage) (copy-number
//!        max, = 2 for diploid), if sum(G0)/(P*N) > 0.5, G0 = P - G0
//!     2. Compute AC = sum(G0)
//!     3. X-adjust: G = G0 - X*(X'VX)^{-1}*(X'V)*G0
//!     4. Normalize: g = G / sqrt(AC)
//!     5. var1 = (G' * Sigma^{-1} * G - G' * Sigma^{-1} * X * (X'Sigma^{-1}X)^{-1} * X' * Sigma^{-1} * G) / AC
//!     6. var2 = inner_product(mu*(1-mu), g*g)  for binary
//!            = inner_product(g, g)              for quantitative
//!     7. VR_marker = var1 / var2
//!   Final VR = mean(VR_marker)

use anyhow::Result;
use tracing::{debug, info};

use saige_linalg::dense::DenseMatrix;

use super::link::TraitType;

/// Configuration for variance ratio estimation.
#[derive(Debug, Clone)]
pub struct VarianceRatioConfig {
    /// Number of markers to use for estimation.
    pub n_markers: usize,
    /// Minimum MAC for markers to be included.
    pub min_mac: f64,
    /// Whether to use categorical variance ratios.
    pub use_categorical: bool,
    /// MAC category boundaries for exclusion (lower bounds).
    pub cate_min_mac_exclude: Vec<f64>,
    /// MAC category boundaries for inclusion (upper bounds).
    pub cate_max_mac_include: Vec<f64>,
    /// Coefficient of variation cutoff for convergence.
    pub ratio_cv_cutoff: f64,
    /// Random seed.
    pub seed: u64,
    /// Ploidy mode for the AF computation and minor-allele flip. Defaults to
    /// diploid; the GRM / variance-ratio markers are usually diploid SNPs.
    pub ploidy: saige_geno::traits::PloidyMode,
}

impl Default for VarianceRatioConfig {
    fn default() -> Self {
        Self {
            n_markers: 30,
            min_mac: 20.0,
            use_categorical: false,
            cate_min_mac_exclude: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 10.5, 20.5],
            cate_max_mac_include: vec![1.5, 2.5, 3.5, 4.5, 5.5, 10.5, 20.5, f64::INFINITY],
            ratio_cv_cutoff: 0.001,
            seed: 12345,
            ploidy: saige_geno::traits::PloidyMode::Diploid,
        }
    }
}

/// Result of variance ratio estimation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VarianceRatioResult {
    /// Overall variance ratio.
    pub variance_ratio: f64,
    /// Categorical variance ratios (if applicable).
    /// Each entry: (MAC upper bound, variance ratio).
    pub categorical_vr: Vec<(f64, f64)>,
    /// Number of markers used.
    pub n_markers_used: usize,
    /// Per-marker variance ratios (for diagnostics).
    pub per_marker_vr: Vec<f64>,
}

/// Estimate the variance ratio following SAIGE's algorithm.
///
/// For each selected marker g:
///   1. Flip to minor allele, compute AC
///   2. X-adjust: g_tilde = g - X * (X'VX)^{-1} * X'V * g
///   3. var1 = g_tilde' * Sigma^{-1} * g_tilde / AC
///   4. var2 = sum(mu*(1-mu) * (g_tilde/sqrt(AC))^2) for binary,
///      or sum((g_tilde/sqrt(AC))^2) for quantitative
///   5. VR = var1 / var2
///
/// Final VR = mean(VR_marker), with adaptive convergence on CV.
#[allow(clippy::too_many_arguments)]
pub fn estimate_variance_ratio<F>(
    dosage_vectors: &[Vec<f64>],
    mac_values: &[f64],
    mu: &[f64],
    _tau: [f64; 2],
    trait_type: TraitType,
    x: &DenseMatrix,
    xvx_inv_xv: &DenseMatrix,
    sigma_inv_vec: F,
    config: &VarianceRatioConfig,
) -> Result<VarianceRatioResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = mu.len();
    let mut per_marker_vr = Vec::new();
    let mut mac_for_vr = Vec::new();

    info!(
        "Estimating variance ratio using up to {} candidate markers",
        dosage_vectors.len()
    );

    // Shuffle marker indices for random selection
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(config.seed);
    let mut indices: Vec<usize> = (0..dosage_vectors.len()).collect();
    indices.shuffle(&mut rng);

    let mut n_tested = 0;
    let mut target_markers = config.n_markers;

    for &idx in &indices {
        let g_raw = &dosage_vectors[idx];
        let mac = mac_values[idx];

        if mac < config.min_mac {
            continue;
        }
        if g_raw.len() != n {
            continue;
        }

        // Flip to minor allele if needed. The copy-number maximum ("ploidy")
        // comes from the configured PloidyMode so the AF and the flip
        // `ploidy - g` generalize beyond the diploid `2 - g` (diploid = 2,
        // haploid = 1, auto = per-marker max). Diploid recovers prior behavior.
        let ploidy = config.ploidy.resolve(g_raw);
        let af = g_raw.iter().filter(|d| !d.is_nan()).sum::<f64>() / (ploidy * n as f64);
        let g0: Vec<f64> = if af > 0.5 {
            g_raw.iter().map(|&gi| ploidy - gi).collect()
        } else {
            g_raw.clone()
        };

        let ac = g0.iter().sum::<f64>();
        if ac < 1.0 {
            continue;
        }

        // X-adjust: G = G0 - X * (X'VX)^{-1} * (X'V) * G0
        let xvx_inv_xv_g = xvx_inv_xv.mat_vec(&g0);
        let x_xvx_inv_xv_g = x.mat_vec(&xvx_inv_xv_g);
        let g_tilde: Vec<f64> = g0
            .iter()
            .zip(x_xvx_inv_xv_g.iter())
            .map(|(gi, xi)| gi - xi)
            .collect();

        // Exact variance: g_tilde' * Sigma^{-1} * g_tilde / AC
        let sigma_inv_g = sigma_inv_vec(&g_tilde);
        let var1: f64 = g_tilde
            .iter()
            .zip(sigma_inv_g.iter())
            .map(|(gi, si)| gi * si)
            .sum::<f64>()
            / ac;

        // Approximate variance: depends on trait type
        // g = g_tilde / sqrt(AC)
        let var2 = match trait_type {
            TraitType::Binary => {
                // var2 = sum(mu*(1-mu) * g^2) where g = g_tilde/sqrt(AC)
                g_tilde
                    .iter()
                    .zip(mu.iter())
                    .map(|(gi, &mi)| mi * (1.0 - mi) * gi * gi / ac)
                    .sum::<f64>()
            }
            TraitType::Quantitative => {
                // var2 = sum(g^2) = sum(g_tilde^2 / AC)
                g_tilde.iter().map(|gi| gi * gi / ac).sum::<f64>()
            }
            TraitType::Survival => {
                // var2 = sum(mu * g^2) where g = g_tilde/sqrt(AC)
                g_tilde
                    .iter()
                    .zip(mu.iter())
                    .map(|(gi, &mi)| mi * gi * gi / ac)
                    .sum::<f64>()
            }
        };

        if var2 < 1e-30 {
            continue;
        }

        let vr = var1 / var2;

        debug!("Marker {}: AC={:.0}, VR={:.6}", idx, ac, vr);

        per_marker_vr.push(vr);
        mac_for_vr.push(mac);
        n_tested += 1;

        // Check convergence via coefficient of variation
        if n_tested >= target_markers {
            let cv = cal_cv(&per_marker_vr);
            if cv <= config.ratio_cv_cutoff {
                break;
            }
            // Need more markers
            target_markers += 10;
            debug!(
                "VR CV={:.6} > cutoff, increasing to {} markers",
                cv, target_markers
            );
        }
    }

    if per_marker_vr.is_empty() {
        anyhow::bail!("No markers available for variance ratio estimation");
    }

    // SAIGE uses mean (not median) for the final VR
    let variance_ratio = mean(&per_marker_vr);

    // Compute categorical VR if requested
    let categorical_vr = if config.use_categorical {
        compute_categorical_vr(
            &per_marker_vr,
            &mac_for_vr,
            &config.cate_min_mac_exclude,
            &config.cate_max_mac_include,
        )
    } else {
        Vec::new()
    };

    info!(
        "Variance ratio: {:.6} (from {} markers, CV={:.6})",
        variance_ratio,
        per_marker_vr.len(),
        cal_cv(&per_marker_vr),
    );

    Ok(VarianceRatioResult {
        variance_ratio,
        categorical_vr,
        n_markers_used: per_marker_vr.len(),
        per_marker_vr,
    })
}

/// Coefficient of variation as used by SAIGE: (SD / mean) / N.
fn cal_cv(values: &[f64]) -> f64 {
    let n = values.len();
    if n <= 1 {
        return f64::INFINITY;
    }
    let m = mean(values);
    if m.abs() < 1e-30 {
        return f64::INFINITY;
    }
    let variance = values.iter().map(|&x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0);
    let sd = variance.sqrt();
    (sd / m) / n as f64
}

/// Compute categorical variance ratios by MAC bin.
fn compute_categorical_vr(
    vr_values: &[f64],
    mac_values: &[f64],
    min_exclude: &[f64],
    max_include: &[f64],
) -> Vec<(f64, f64)> {
    let mut result = Vec::new();

    for (min_ex, max_in) in min_exclude.iter().zip(max_include.iter()) {
        let vrs_in_bin: Vec<f64> = vr_values
            .iter()
            .zip(mac_values.iter())
            .filter(|(_, &mac)| mac > *min_ex && mac <= *max_in)
            .map(|(&vr, _)| vr)
            .collect();

        if !vrs_in_bin.is_empty() {
            result.push((*max_in, mean(&vrs_in_bin)));
        }
    }

    result
}

/// Select the appropriate variance ratio for a given MAC value.
pub fn select_variance_ratio(mac: f64, overall_vr: f64, categorical_vr: &[(f64, f64)]) -> f64 {
    if categorical_vr.is_empty() {
        return overall_vr;
    }

    for &(upper, vr) in categorical_vr {
        if mac <= upper {
            return vr;
        }
    }

    overall_vr
}

/// Compute the mean of a slice.
fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Write variance ratio results to a file in SAIGE's format.
///
/// SAIGE format: space-separated, no header
/// ```text
/// 0.94022084164312 null 1
/// ```
/// For categorical:
/// ```text
/// 0.8 null 1
/// 0.9 null 2
/// ```
pub fn write_variance_ratio_file(
    result: &VarianceRatioResult,
    path: &std::path::Path,
) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;

    if result.categorical_vr.is_empty() {
        writeln!(f, "{} null 1", result.variance_ratio)?;
    } else {
        for (k, (_, vr)) in result.categorical_vr.iter().enumerate() {
            writeln!(f, "{} null {}", vr, k + 1)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0]) - 2.0).abs() < 1e-10);
        assert!((mean(&[1.0, 3.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cal_cv() {
        // All same values -> CV = 0
        let cv = cal_cv(&[1.0, 1.0, 1.0, 1.0]);
        assert!(cv < 1e-10, "cv={}", cv);

        // Some variation
        let cv2 = cal_cv(&[0.9, 1.0, 1.1]);
        assert!(cv2 > 0.0);
        assert!(cv2 < 1.0); // Should be small for tight distribution
    }

    #[test]
    fn test_select_variance_ratio() {
        let categorical = vec![(10.0, 0.8), (20.0, 0.9), (f64::INFINITY, 1.0)];
        assert_eq!(select_variance_ratio(5.0, 0.95, &categorical), 0.8);
        assert_eq!(select_variance_ratio(15.0, 0.95, &categorical), 0.9);
        assert_eq!(select_variance_ratio(100.0, 0.95, &categorical), 1.0);
        // Empty categorical -> use overall
        assert_eq!(select_variance_ratio(5.0, 0.95, &[]), 0.95);
    }
}
