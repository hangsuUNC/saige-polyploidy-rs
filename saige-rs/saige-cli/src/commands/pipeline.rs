//! Shared Step 1 / Step 2 pipeline helpers.
//!
//! Used by `fit-null`, `test`, and `phewas` so the null-model fit and the
//! score-test engine construction live in exactly one place.

use anyhow::Result;
use tracing::info;

use saige_core::glmm::ai_reml::{fit_ai_reml, AiRemlConfig};
use saige_core::glmm::link::TraitType;
use saige_core::glmm::pcg::OnTheFlyGrm;
use saige_core::glmm::variance_ratio::{
    estimate_variance_ratio, VarianceRatioConfig, VarianceRatioResult,
};
use saige_core::model::null_model::NullModel;
use saige_core::score_test::single_variant::ScoreTestEngine;
use saige_geno::plink::PlinkReader;
use saige_geno::traits::{GenotypeReader, PloidyMode};
use saige_linalg::decomposition::PcgSolver;
use saige_linalg::dense::DenseMatrix;

/// Configuration for fitting a null GLMM (shared by `fit-null` and `phewas`).
#[derive(Debug, Clone)]
pub struct NullFitConfig {
    pub trait_type: TraitType,
    pub ploidy: PloidyMode,
    pub min_maf: f64,
    pub max_missing_rate: f64,
    pub n_random_vectors: u32,
    pub n_markers_vr: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub seed: u64,
    pub use_sparse_grm: bool,
    pub use_categorical_vr: bool,
    /// Skip variance-ratio estimation (use VR = 1.0). Useful for PheWAS with
    /// few test variants (per Wei's suggestion) and avoids VR miscalibration.
    pub skip_vr: bool,
}

/// Fit a null GLMM for a single phenotype.
///
/// `plink` must already have its ploidy mode and sample subset set to
/// `valid_ids`. `y` is the phenotype vector (length n) and `x_data` the
/// col-major design matrix (n x p, intercept + covariates) for those samples.
/// Reads GRM/VR markers, fits AI-REML, estimates the variance ratio (unless
/// `skip_vr`), and returns the in-memory [`NullModel`].
pub fn fit_null_model(
    plink: &mut PlinkReader,
    valid_ids: Vec<String>,
    y: Vec<f64>,
    x_data: Vec<f64>,
    p: usize,
    cfg: &NullFitConfig,
) -> Result<NullModel> {
    let n = y.len();
    let x = DenseMatrix::from_col_major(n, p, x_data.clone());
    let n_samples_valid = valid_ids.len();

    // Read genotype dosages. R SAIGE reserves a random subset of markers for VR
    // estimation and excludes them from GRM construction; we match that.
    let n_markers = plink.n_markers();

    struct MarkerRow {
        dosages: Vec<f64>,
        af: f64,
        mac: f64,
    }

    let mut all_passing: Vec<MarkerRow> = Vec::new();
    for m in 0..n_markers {
        let data = plink.read_marker(m as u64)?;
        let missing_rate = 1.0 - (data.n_valid as f64 / n_samples_valid as f64);
        if data.af >= cfg.min_maf
            && data.af <= 1.0 - cfg.min_maf
            && missing_rate <= cfg.max_missing_rate
        {
            all_passing.push(MarkerRow {
                dosages: data.dosages,
                af: data.af,
                mac: data.mac,
            });
        }
    }
    info!(
        "{} markers pass QC (MAF >= {}, missing <= {})",
        all_passing.len(),
        cfg.min_maf,
        cfg.max_missing_rate,
    );

    // Select VR candidate indices: draw 1000 random indices (matching R SAIGE)
    // from markers with MAC >= 20, then deduplicate.
    use rand::Rng;
    use rand::SeedableRng;
    let mut vr_rng = rand_chacha::ChaCha8Rng::seed_from_u64(cfg.seed);
    let n_passing = all_passing.len().max(1);
    let mut vr_candidate_indices: Vec<usize> =
        (0..1000).map(|_| vr_rng.gen_range(0..n_passing)).collect();
    vr_candidate_indices.sort_unstable();
    vr_candidate_indices.dedup();
    vr_candidate_indices.retain(|&i| i < all_passing.len() && all_passing[i].mac >= 20.0);
    let vr_set: std::collections::HashSet<usize> = vr_candidate_indices.iter().copied().collect();

    // Split into GRM and VR pools (mutually exclusive, matching R SAIGE).
    let mut grm_dosages = Vec::new();
    let mut grm_afs = Vec::new();
    let mut vr_dosages = Vec::new();
    let mut vr_macs = Vec::new();

    for (i, marker) in all_passing.into_iter().enumerate() {
        if vr_set.contains(&i) {
            vr_dosages.push(marker.dosages);
            vr_macs.push(marker.mac);
        } else {
            grm_dosages.push(marker.dosages);
            grm_afs.push(marker.af);
        }
    }
    info!(
        "Using {} markers for GRM, {} reserved for VR estimation",
        grm_dosages.len(),
        vr_dosages.len(),
    );

    // Build on-the-fly GRM (takes reference, copies internally).
    let grm = OnTheFlyGrm::new(&grm_dosages, &grm_afs);
    let grm_vec = move |v: &[f64]| -> Vec<f64> { grm.mat_vec(v) };

    // Fit null model using AI-REML.
    let reml_config = AiRemlConfig {
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        pcg_tol: 1e-5,
        pcg_max_iter: 500,
        n_random_vectors: cfg.n_random_vectors,
        use_sparse_grm: cfg.use_sparse_grm,
        seed: cfg.seed,
    };

    info!("Fitting null model with AI-REML...");
    let reml_result = fit_ai_reml(&y, &x, grm_vec, cfg.trait_type, &reml_config)?;
    info!(
        "AI-REML result: tau=[{:.6}, {:.6}], converged={}",
        reml_result.tau[0], reml_result.tau[1], reml_result.converged
    );

    // Compute XVX_inv_XV.
    let w = reml_result.working_weights.clone();
    let xvx = x.xtwx(&w);
    let xvx_inv = saige_linalg::decomposition::inverse_spd(&xvx)?;
    let xvx_inv_xv_data: Vec<f64> = {
        let mut data = vec![0.0; p * n];
        for j in 0..p {
            for i in 0..n {
                let mut val = 0.0;
                for k in 0..p {
                    val += xvx_inv.get(j, k) * x.get(i, k) * w[i];
                }
                data[j * n + i] = val;
            }
        }
        data
    };
    let xvx_inv_xv = DenseMatrix::from_col_major(p, n, xvx_inv_xv_data.clone());

    // Estimate variance ratios (unless skipped).
    let tau = reml_result.tau;
    let vr_result = if cfg.skip_vr {
        info!("Skipping variance ratio estimation (--skip-vr); using VR = 1.0");
        VarianceRatioResult {
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            n_markers_used: 0,
            per_marker_vr: Vec::new(),
        }
    } else if vr_dosages.is_empty() {
        info!("No markers available for VR estimation, using default VR = 1.0");
        VarianceRatioResult {
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            n_markers_used: 0,
            per_marker_vr: Vec::new(),
        }
    } else {
        info!("Estimating variance ratios...");
        let mu_for_vr = reml_result.mu.clone();
        let w_for_vr = reml_result.working_weights.clone();
        let grm_for_vr = OnTheFlyGrm::new(&grm_dosages, &grm_afs);
        let pcg = PcgSolver::new(1e-5, 500);

        let sigma_inv = |v: &[f64]| -> Vec<f64> {
            let sigma_op = |sv: &[f64]| -> Vec<f64> {
                let grm_sv = grm_for_vr.mat_vec(sv);
                sv.iter()
                    .zip(w_for_vr.iter())
                    .zip(grm_sv.iter())
                    .map(|((svi, wi), gi)| tau[0] * svi / wi.max(1e-30) + tau[1] * gi)
                    .collect()
            };
            let precond = |pv: &[f64]| -> Vec<f64> {
                pv.iter()
                    .zip(w_for_vr.iter())
                    .map(|(pvi, wi)| {
                        let diag = tau[0] / wi.max(1e-30) + tau[1];
                        if diag.abs() > 1e-30 {
                            pvi / diag
                        } else {
                            *pvi
                        }
                    })
                    .collect()
            };
            pcg.solve(sigma_op, precond, v, None).x
        };

        let vr_config = VarianceRatioConfig {
            n_markers: cfg.n_markers_vr,
            min_mac: 20.0,
            use_categorical: cfg.use_categorical_vr,
            seed: cfg.seed,
            ploidy: cfg.ploidy,
            ..Default::default()
        };

        estimate_variance_ratio(
            &vr_dosages,
            &vr_macs,
            &mu_for_vr,
            tau,
            cfg.trait_type,
            &x,
            &xvx_inv_xv,
            sigma_inv,
            &vr_config,
        )?
    };
    info!("Variance ratio: {:.6}", vr_result.variance_ratio);

    let model = NullModel::new(
        cfg.trait_type,
        valid_ids,
        reml_result.tau,
        reml_result.alpha,
        reml_result.mu,
        y,
        x_data,
        p,
        xvx_inv_xv_data,
        vr_result,
    );

    Ok(model)
}

/// Build a score-test engine from a fitted null model.
pub fn build_engine(
    model: &NullModel,
    use_spa: bool,
    use_fast_spa: bool,
    spa_pval_cutoff: f64,
    ploidy: PloidyMode,
) -> ScoreTestEngine {
    let n = model.n_samples;
    let p = model.n_covariates;
    let x = DenseMatrix::from_col_major(n, p, model.x_flat.clone());
    let xvx_inv_xv = DenseMatrix::from_col_major(p, n, model.xvx_inv_xv_flat.clone());

    ScoreTestEngine {
        trait_type: model.trait_type,
        mu: model.mu.clone(),
        mu2: model.mu2.clone(),
        residuals: model.residuals.clone(),
        tau_e: model.tau[0],
        tau_g: model.tau[1],
        xvx_inv_xv,
        x,
        variance_ratio: model.variance_ratio.variance_ratio,
        categorical_vr: model.variance_ratio.categorical_vr.clone(),
        use_spa: use_spa && model.trait_type == TraitType::Binary,
        use_fast_spa,
        spa_tol: 1e-6,
        spa_pval_cutoff,
        ploidy,
        y: if model.trait_type == TraitType::Binary {
            Some(model.y.clone())
        } else {
            None
        },
    }
}
