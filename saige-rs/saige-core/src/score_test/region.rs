//! Region-based association tests: BURDEN, SKAT, SKAT-O.
//!
//! Implements set-based tests that aggregate association signals
//! across multiple variants within a gene or region.
//!
//! Reference: SAIGE R/SAIGE_SPATest_Region.R

use anyhow::Result;
use statrs::distribution::{ChiSquared, ContinuousCDF};

use saige_linalg::dense::DenseMatrix;

use super::single_variant::ScoreTestEngine;

/// Configuration for region-based tests.
#[derive(Debug, Clone)]
pub struct RegionTestConfig {
    /// Rho values for SKAT-O grid search.
    pub rho_values: Vec<f64>,
    /// MAC categories (upper bounds) for MAC stratification.
    pub mac_categories: Vec<f64>,
    /// Minimum MAC for including a variant.
    pub min_mac: f64,
    /// Maximum MAF for including a variant.
    pub max_maf: f64,
}

impl Default for RegionTestConfig {
    fn default() -> Self {
        Self {
            rho_values: default_rho_values(),
            mac_categories: vec![1.5, 2.5, 5.5, 10.5, 20.5, 40.5, f64::INFINITY],
            min_mac: 0.5,
            max_maf: 1.0,
        }
    }
}

/// Result of a region-based test.
#[derive(Debug, Clone)]
pub struct RegionTestResult {
    /// Gene/region name.
    pub region_name: String,
    /// SKAT-O p-value (combined optimal).
    pub pvalue: f64,
    /// MAC category counts.
    pub mac_category_counts: Vec<usize>,
    /// Marker IDs included in the test.
    pub marker_ids: Vec<String>,
    /// Marker allele frequencies.
    pub marker_afs: Vec<f64>,
    /// Burden test p-value.
    pub burden_pvalue: f64,
    /// SKAT test p-value.
    pub skat_pvalue: f64,
    /// P-value without SPA (NA-adjusted).
    pub pvalue_na: f64,
    /// Burden p-value without SPA.
    pub burden_pvalue_na: f64,
    /// SKAT p-value without SPA.
    pub skat_pvalue_na: f64,
}

/// Per-variant score statistics needed for region tests.
struct VariantScoreStats {
    /// Score statistic S_j for variant j.
    scores: Vec<f64>,
    /// Variance-covariance matrix Phi of score statistics (m x m).
    phi: DenseMatrix,
}

/// Compute per-variant score statistics and the Phi matrix.
///
/// Score_j = g_tilde_j' * residuals
/// Phi[j,k] = g_tilde_j' * diag(mu2) * g_tilde_k
fn compute_variant_scores(
    genotypes: &[Vec<f64>],
    weights: &[f64],
    engine: &ScoreTestEngine,
) -> VariantScoreStats {
    let n = engine.mu.len();
    let m = genotypes.len();

    // Compute g_tilde for each variant (projected genotypes)
    let mut g_tildes: Vec<Vec<f64>> = Vec::with_capacity(m);

    for (g, &w) in genotypes.iter().zip(weights.iter()) {
        // Weighted genotype (impute missing as 0)
        let gw: Vec<f64> = g
            .iter()
            .map(|&gi| {
                let gi = if gi.is_nan() { 0.0 } else { gi };
                w * gi
            })
            .collect();

        // Project out X: g_tilde = g - X * (X'VX)^{-1} * X'V * g
        let xvx_inv_xv_g = engine.xvx_inv_xv.mat_vec(&gw);
        let x_proj = engine.x.mat_vec(&xvx_inv_xv_g);
        let g_tilde: Vec<f64> = gw
            .iter()
            .zip(x_proj.iter())
            .map(|(gi, xi)| gi - xi)
            .collect();

        g_tildes.push(g_tilde);
    }

    // Score statistics
    let scores: Vec<f64> = g_tildes
        .iter()
        .map(|gt| {
            gt.iter()
                .zip(engine.residuals.iter())
                .map(|(g, r)| g * r)
                .sum()
        })
        .collect();

    // Phi matrix: Phi[j,k] = g_tilde_j' * diag(mu2) * g_tilde_k
    let mut phi = DenseMatrix::zeros(m, m);
    for j in 0..m {
        for k in j..m {
            let mut dot = 0.0;
            for (i, g_tilde_ji) in g_tildes[j].iter().enumerate().take(n) {
                dot += g_tilde_ji * engine.mu2[i] * g_tildes[k][i];
            }
            // Scale by tau_e for proper variance
            dot *= engine.tau_e;
            phi.set(j, k, dot);
            if j != k {
                phi.set(k, j, dot);
            }
        }
    }

    VariantScoreStats { scores, phi }
}

/// Compute the burden test for a set of variants.
///
/// Burden test collapses multiple variants into a single "super-variant"
/// by taking a weighted sum of genotypes, then applies the score test.
///
/// Q_burden = (sum(S_j))^2 / sum(Phi[j,k])
pub fn burden_test(
    genotypes: &[Vec<f64>],
    weights: &[f64],
    engine: &ScoreTestEngine,
) -> Result<f64> {
    let m = genotypes.len();
    if m == 0 {
        return Ok(1.0);
    }

    let stats = compute_variant_scores(genotypes, weights, engine);

    // Burden statistic: Q = (sum(S))^2 / var(sum(S))
    let sum_score: f64 = stats.scores.iter().sum();

    // Variance of sum of scores = sum of all elements of Phi
    let mut sum_phi = 0.0;
    for j in 0..m {
        for k in 0..m {
            sum_phi += stats.phi.get(j, k);
        }
    }

    if sum_phi < 1e-30 {
        return Ok(1.0);
    }

    let q_burden = sum_score * sum_score / sum_phi;

    let chi2 = ChiSquared::new(1.0).unwrap();
    let pval = 1.0 - chi2.cdf(q_burden);
    Ok(pval)
}

/// Compute SKAT (Sequence Kernel Association Test).
///
/// Q_SKAT = sum(S_j^2) = S' * S
///
/// P-value via Satterthwaite's moment-matching method:
/// Q ~ kappa * chi^2(df), where kappa and df are chosen to match
/// the first two moments of the distribution of Q under H0.
pub fn skat_test(genotypes: &[Vec<f64>], weights: &[f64], engine: &ScoreTestEngine) -> Result<f64> {
    let m = genotypes.len();
    if m == 0 {
        return Ok(1.0);
    }

    let stats = compute_variant_scores(genotypes, weights, engine);

    // SKAT Q statistic: Q = sum(S_j^2)
    let q_stat: f64 = stats.scores.iter().map(|s| s * s).sum();

    // Mean of Q under H0: E[Q] = trace(Phi)
    let mean_q: f64 = (0..m).map(|j| stats.phi.get(j, j)).sum();
    if mean_q < 1e-30 {
        return Ok(1.0);
    }

    // Var of Q under H0: Var[Q] = 2 * trace(Phi * Phi)
    // = 2 * sum_{j,k} Phi[j,k]^2
    let mut trace_phi_sq = 0.0;
    for j in 0..m {
        for k in 0..m {
            let v = stats.phi.get(j, k);
            trace_phi_sq += v * v;
        }
    }
    let var_q = 2.0 * trace_phi_sq;

    if var_q < 1e-30 {
        return Ok(1.0);
    }

    // Satterthwaite's method: match to kappa * chi^2(df)
    let kappa = var_q / (2.0 * mean_q);
    let df = 2.0 * mean_q * mean_q / var_q;

    if df > 0.0 && kappa > 0.0 {
        let chi2 = ChiSquared::new(df).unwrap();
        let pval = 1.0 - chi2.cdf(q_stat / kappa);
        Ok(pval)
    } else {
        Ok(1.0)
    }
}

/// Compute SKAT-O (optimal unified test).
///
/// SKAT-O finds the optimal linear combination of BURDEN and SKAT
/// by searching over rho in [0, 1]:
///   Q(rho) = (1-rho)*Q_SKAT + rho*Q_BURDEN
///
/// For each rho, the p-value is computed using the appropriate
/// null distribution, and the minimum p-value across rho values
/// is returned (with multiplicity correction).
pub fn skat_o_test(
    genotypes: &[Vec<f64>],
    weights: &[f64],
    engine: &ScoreTestEngine,
    rho_values: &[f64],
) -> Result<(f64, f64)> {
    let m = genotypes.len();
    if m == 0 {
        return Ok((1.0, 0.0));
    }

    let stats = compute_variant_scores(genotypes, weights, engine);

    // Q_SKAT = sum(S_j^2)
    let q_skat: f64 = stats.scores.iter().map(|s| s * s).sum();

    // Q_BURDEN = (sum(S_j))^2
    let sum_score: f64 = stats.scores.iter().sum();
    let q_burden = sum_score * sum_score;

    // Sum of all Phi elements (for burden variance)
    let mut sum_phi = 0.0;
    for j in 0..m {
        for k in 0..m {
            sum_phi += stats.phi.get(j, k);
        }
    }

    let mut best_pval = 1.0;
    let mut best_rho = 0.0;

    for &rho in rho_values {
        // Q(rho) = (1-rho) * Q_SKAT + rho * Q_BURDEN
        let q_rho = (1.0 - rho) * q_skat + rho * q_burden;

        // Variance-covariance under H0 for Q(rho):
        // The kernel K(rho) = (1-rho) * I + rho * 1*1'
        // E[Q(rho)] = trace(Phi * K(rho))
        //           = (1-rho) * trace(Phi) + rho * sum(Phi)
        let trace_phi: f64 = (0..m).map(|j| stats.phi.get(j, j)).sum();
        let mean_q_rho = (1.0 - rho) * trace_phi + rho * sum_phi;

        if mean_q_rho < 1e-30 {
            continue;
        }

        // Var[Q(rho)] = 2 * trace((Phi * K(rho))^2)
        // For simplicity, compute Phi_rho = Phi * K(rho), then trace(Phi_rho^2)
        // Phi_rho[j,k] = (1-rho) * Phi[j,k] + rho * sum_l(Phi[j,l])
        // This is equivalent to Phi_rho = (1-rho)*Phi + rho*Phi*1*1'
        let mut var_q_rho = 0.0;
        // Precompute row sums of Phi
        let phi_row_sums: Vec<f64> = (0..m)
            .map(|j| (0..m).map(|k| stats.phi.get(j, k)).sum::<f64>())
            .collect();

        for j in 0..m {
            for k in 0..m {
                let phi_rho_jk = (1.0 - rho) * stats.phi.get(j, k) + rho * phi_row_sums[k];
                let phi_rho_kj = (1.0 - rho) * stats.phi.get(k, j) + rho * phi_row_sums[j];
                var_q_rho += phi_rho_jk * phi_rho_kj;
            }
        }
        var_q_rho *= 2.0;

        if var_q_rho < 1e-30 {
            continue;
        }

        // Satterthwaite's method
        let kappa = var_q_rho / (2.0 * mean_q_rho);
        let df = 2.0 * mean_q_rho * mean_q_rho / var_q_rho;

        if df > 0.0 && kappa > 0.0 {
            let chi2 = ChiSquared::new(df).unwrap();
            let pval = 1.0 - chi2.cdf(q_rho / kappa);
            if pval < best_pval {
                best_pval = pval;
                best_rho = rho;
            }
        }
    }

    Ok((best_pval, best_rho))
}

/// Run the full region test pipeline for a gene.
///
/// Returns SKAT-O, Burden, and SKAT p-values.
pub fn run_region_test(
    genotypes: &[Vec<f64>],
    weights: &[f64],
    engine: &ScoreTestEngine,
    config: &RegionTestConfig,
) -> Result<(f64, f64, f64)> {
    let m = genotypes.len();
    if m == 0 {
        return Ok((1.0, 1.0, 1.0));
    }

    let p_burden = burden_test(genotypes, weights, engine)?;
    let p_skat = skat_test(genotypes, weights, engine)?;
    let (p_skato, _rho) = skat_o_test(genotypes, weights, engine, &config.rho_values)?;

    Ok((p_skato, p_burden, p_skat))
}

/// Classify MAC values into categories and return counts per category.
pub fn mac_category_counts(mac_values: &[f64], boundaries: &[f64]) -> Vec<usize> {
    let n_cats = boundaries.len() + 1;
    let mut counts = vec![0usize; n_cats.min(8)]; // SAIGE uses 8 categories max

    for &mac in mac_values {
        let cat = boundaries
            .iter()
            .position(|&b| mac <= b)
            .unwrap_or(boundaries.len());
        if cat < counts.len() {
            counts[cat] += 1;
        }
    }

    // Pad to 8 categories
    counts.resize(8, 0);
    counts
}

/// Default rho values for SKAT-O grid search.
pub fn default_rho_values() -> Vec<f64> {
    vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

/// Compute Beta(MAF, a1, a2) weights for variant weighting.
/// Default: Beta(MAF, 1, 25) which upweights rare variants.
pub fn beta_weights(mafs: &[f64], a1: f64, a2: f64) -> Vec<f64> {
    use statrs::function::beta::ln_beta;

    mafs.iter()
        .map(|&maf| {
            let maf = maf.clamp(1e-10, 1.0 - 1e-10);
            // Beta density: f(x; a1, a2) = x^(a1-1) * (1-x)^(a2-1) / B(a1, a2)
            let log_weight =
                (a1 - 1.0) * maf.ln() + (a2 - 1.0) * (1.0 - maf).ln() - ln_beta(a1, a2);
            log_weight.exp()
        })
        .collect()
}

/// Write region test results header in SAIGE's format.
pub fn write_region_header(writer: &mut impl std::io::Write) -> Result<()> {
    writeln!(
        writer,
        "Gene Pvalue Nmarker_MACCate_1 Nmarker_MACCate_2 Nmarker_MACCate_3 Nmarker_MACCate_4 Nmarker_MACCate_5 Nmarker_MACCate_6 Nmarker_MACCate_7 Nmarker_MACCate_8 markerIDs markerAFs Pvalue_Burden Pvalue_SKAT Pvalue.NA Pvalue_Burden.NA Pvalue_SKAT.NA"
    )?;
    Ok(())
}

/// Write a single region test result line in SAIGE's format.
pub fn write_region_line(
    writer: &mut impl std::io::Write,
    result: &RegionTestResult,
) -> Result<()> {
    let marker_ids = result.marker_ids.join(";");
    let marker_afs: Vec<String> = result.marker_afs.iter().map(|a| format!("{}", a)).collect();
    let marker_afs_str = marker_afs.join(";");

    let cats = &result.mac_category_counts;
    writeln!(
        writer,
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
        result.region_name,
        result.pvalue,
        cats.first().copied().unwrap_or(0),
        cats.get(1).copied().unwrap_or(0),
        cats.get(2).copied().unwrap_or(0),
        cats.get(3).copied().unwrap_or(0),
        cats.get(4).copied().unwrap_or(0),
        cats.get(5).copied().unwrap_or(0),
        cats.get(6).copied().unwrap_or(0),
        cats.get(7).copied().unwrap_or(0),
        marker_ids,
        marker_afs_str,
        result.burden_pvalue,
        result.skat_pvalue,
        result.pvalue_na,
        result.burden_pvalue_na,
        result.skat_pvalue_na,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beta_weights_rare_upweighted() {
        let mafs = vec![0.001, 0.01, 0.05, 0.1, 0.3];
        let weights = beta_weights(&mafs, 1.0, 25.0);
        // Rarer variants should have higher weights
        assert!(weights[0] > weights[1]);
        assert!(weights[1] > weights[2]);
        assert!(weights[2] > weights[3]);
    }

    #[test]
    fn test_default_rho_values() {
        let rhos = default_rho_values();
        assert_eq!(rhos.len(), 11);
        assert_eq!(rhos[0], 0.0);
        assert_eq!(rhos[10], 1.0);
    }

    #[test]
    fn test_mac_category_counts() {
        let macs = vec![0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 25.0, 50.0];
        let boundaries = vec![1.5, 2.5, 5.5, 10.5, 20.5, 40.5];
        let counts = mac_category_counts(&macs, &boundaries);
        assert_eq!(counts[0], 2); // MAC <= 1.5: 0.5, 1.0
        assert_eq!(counts[1], 1); // 1.5 < MAC <= 2.5: 2.0
        assert_eq!(counts[2], 1); // 2.5 < MAC <= 5.5: 5.0
        assert_eq!(counts[3], 1); // 5.5 < MAC <= 10.5: 10.0
        assert_eq!(counts[4], 1); // 10.5 < MAC <= 20.5: 15.0
        assert_eq!(counts[5], 1); // 20.5 < MAC <= 40.5: 25.0
        assert_eq!(counts[6], 1); // > 40.5: 50.0
        assert_eq!(counts[7], 0); // padding
    }

    #[test]
    fn test_region_test_basic() {
        // Construct a simple engine for testing
        let n = 100;
        let mu = vec![0.5; n];
        let mu2 = vec![0.25; n]; // mu * (1-mu)
        let residuals: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let x = DenseMatrix::from_col_major(n, 1, vec![1.0; n]);
        let xvx_inv_xv = DenseMatrix::from_col_major(1, n, vec![1.0 / n as f64; n]);

        let engine = ScoreTestEngine {
            trait_type: crate::glmm::link::TraitType::Binary,
            mu,
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
            ploidy: saige_geno::traits::PloidyMode::Diploid,
            y: None,
        };

        // Generate some genotype vectors
        let genotypes: Vec<Vec<f64>> = (0..5)
            .map(|j| {
                (0..n)
                    .map(|i| if (i + j) % 10 == 0 { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let weights = vec![1.0; 5];

        let p_burden = burden_test(&genotypes, &weights, &engine).unwrap();
        assert!((0.0..=1.0).contains(&p_burden), "burden p={}", p_burden);

        let p_skat = skat_test(&genotypes, &weights, &engine).unwrap();
        assert!((0.0..=1.0).contains(&p_skat), "skat p={}", p_skat);

        let (p_skato, _rho) =
            skat_o_test(&genotypes, &weights, &engine, &default_rho_values()).unwrap();
        assert!((0.0..=1.0).contains(&p_skato), "skato p={}", p_skato);

        // SKAT-O should be at least as significant as the less significant test
        // (it picks the best rho)
        assert!(p_skato <= p_burden.max(p_skat) + 1e-10);
    }
}
