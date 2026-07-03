//! Step 1: Fit the null GLMM.
//!
//! saige fit-null --plink-file ... --pheno-file ... --pheno-col ... --trait-type binary --output-prefix ...

use anyhow::Result;
use clap::Args;
use tracing::info;

use saige_core::glmm::ai_reml::{fit_ai_reml, AiRemlConfig};
use saige_core::glmm::link::TraitType;
use saige_core::glmm::pcg::OnTheFlyGrm;
use saige_core::glmm::variance_ratio::{
    estimate_variance_ratio, write_variance_ratio_file, VarianceRatioConfig, VarianceRatioResult,
};
use saige_core::model::null_model::NullModel;
use saige_core::model::serialization;
use saige_geno::phenotype;
use saige_geno::plink::PlinkReader;
use saige_geno::sample;
use saige_geno::traits::{GenotypeReader, PloidyMode};
use saige_linalg::decomposition::PcgSolver;
use saige_linalg::dense::DenseMatrix;

#[derive(Args)]
pub struct FitNullArgs {
    /// PLINK file prefix (bed/bim/fam)
    #[arg(long)]
    plink_file: String,

    /// Phenotype file path
    #[arg(long)]
    pheno_file: String,

    /// Phenotype column name
    #[arg(long)]
    pheno_col: String,

    /// Covariate column names (comma-separated)
    #[arg(long, default_value = "")]
    covar_cols: String,

    /// Sample ID column name
    #[arg(long, default_value = "IID")]
    sample_id_col: String,

    /// Trait type: binary, quantitative, or survival
    #[arg(long, default_value = "binary")]
    trait_type: String,

    /// Output file prefix
    #[arg(long)]
    output_prefix: String,

    /// Use sparse GRM
    #[arg(long, default_value = "false")]
    use_sparse_grm: bool,

    /// Sparse GRM file path
    #[arg(long)]
    sparse_grm_file: Option<String>,

    /// Use LOCO (Leave-One-Chromosome-Out)
    #[arg(long, default_value = "true")]
    loco: bool,

    /// Number of random vectors for trace estimation
    #[arg(long, default_value = "30")]
    n_random_vectors: u32,

    /// Number of markers for variance ratio estimation
    #[arg(long, default_value = "30")]
    n_markers_vr: usize,

    /// Maximum AI-REML iterations
    #[arg(long, default_value = "30")]
    max_iter: usize,

    /// Convergence tolerance
    #[arg(long, default_value = "1e-5")]
    tol: f64,

    /// Random seed
    #[arg(long, default_value = "12345")]
    seed: u64,

    /// Minimum MAF for GRM markers
    #[arg(long, default_value = "0.01")]
    min_maf: f64,

    /// Maximum missing rate for GRM markers
    #[arg(long, default_value = "0.15")]
    max_missing_rate: f64,

    /// Whether to use categorical variance ratios
    #[arg(long, default_value = "false")]
    use_categorical_vr: bool,

    /// Ploidy for the GRM / variance-ratio markers: "diploid" (2, default),
    /// "haploid" (1), "auto" (per-marker max), or a positive number. Usually
    /// left at diploid since the GRM is built from nuclear SNPs.
    #[arg(long, default_value = "diploid")]
    ploidy: String,

    /// Also save JSON sidecar for debugging
    #[arg(long, default_value = "false")]
    save_json: bool,
}

pub fn run(args: FitNullArgs) -> Result<()> {
    let trait_type = match args.trait_type.to_lowercase().as_str() {
        "binary" => TraitType::Binary,
        "quantitative" | "quant" => TraitType::Quantitative,
        "survival" => TraitType::Survival,
        _ => anyhow::bail!("Unknown trait type: {}", args.trait_type),
    };

    info!("=== SAIGE Step 1: Fit Null Model ===");
    info!("Trait type: {:?}", trait_type);
    info!("PLINK file: {}", args.plink_file);
    info!("Phenotype file: {}", args.pheno_file);
    info!("Phenotype column: {}", args.pheno_col);

    // Load genotype data
    let mut plink = PlinkReader::new(&args.plink_file)?;
    info!(
        "Loaded {} markers x {} samples from PLINK files",
        plink.n_markers(),
        plink.n_samples()
    );

    // Load phenotype data
    let covar_cols: Vec<String> = if args.covar_cols.is_empty() {
        Vec::new()
    } else {
        args.covar_cols
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    let pheno_data = phenotype::parse_phenotype_file(
        std::path::Path::new(&args.pheno_file),
        &args.pheno_col,
        &covar_cols,
        &args.sample_id_col,
    )?;
    info!(
        "Loaded phenotypes for {} samples",
        pheno_data.sample_ids.len()
    );

    // Intersect samples
    let geno_ids = plink.sample_ids().to_vec();
    let intersection = sample::intersect_samples(&[&pheno_data.sample_ids, &geno_ids]);
    info!("Sample intersection: {} samples", intersection.ids.len());

    if intersection.ids.is_empty() {
        anyhow::bail!("No overlapping samples between phenotype and genotype files");
    }

    // Filter to valid samples (non-missing phenotype and covariates)
    let valid_indices = phenotype::valid_sample_indices(&pheno_data);
    let valid_ids: Vec<String> = valid_indices
        .iter()
        .filter(|&&i| intersection.ids.contains(&pheno_data.sample_ids[i]))
        .map(|&i| pheno_data.sample_ids[i].clone())
        .collect();

    info!("Valid samples after filtering: {}", valid_ids.len());

    // Resolve ploidy for GRM / VR markers and apply to the reader.
    let ploidy = PloidyMode::parse(&args.ploidy)?;
    info!("Ploidy mode (GRM/VR markers): {:?}", ploidy);
    plink.set_ploidy(ploidy);

    // Set sample subset in genotype reader
    plink.set_sample_subset(&valid_ids)?;

    // Build phenotype and design matrix for valid samples
    let valid_pheno_indices: Vec<usize> = valid_ids
        .iter()
        .map(|id| pheno_data.sample_ids.iter().position(|s| s == id).unwrap())
        .collect();

    let y: Vec<f64> = valid_pheno_indices
        .iter()
        .map(|&i| pheno_data.phenotype[i])
        .collect();

    let n = y.len();
    let p = covar_cols.len() + 1; // +1 for intercept
    let mut x_data = vec![0.0; n * p];
    for xi in x_data.iter_mut().take(n) {
        *xi = 1.0; // intercept
    }
    for (j, _) in covar_cols.iter().enumerate() {
        for (idx, &pheno_idx) in valid_pheno_indices.iter().enumerate() {
            x_data[(j + 1) * n + idx] = pheno_data.covariates[pheno_idx][j];
        }
    }
    let x = DenseMatrix::from_col_major(n, p, x_data.clone());

    // Read genotype dosages.
    // R SAIGE reserves a random subset of markers for VR estimation and excludes
    // them from GRM construction to ensure independence. We match that behavior:
    // 1. First pass: read all markers passing QC
    // 2. Randomly select VR candidate indices (1000 draws with dedup, ~900 unique)
    // 3. Second pass: split into GRM-only and VR-only pools
    info!("Reading genotypes...");
    let n_markers = plink.n_markers();

    struct MarkerData {
        dosages: Vec<f64>,
        af: f64,
        mac: f64,
    }

    let mut all_passing: Vec<MarkerData> = Vec::new();
    let n_samples_valid = valid_ids.len();
    for m in 0..n_markers {
        let data = plink.read_marker(m as u64)?;
        let missing_rate = 1.0 - (data.n_valid as f64 / n_samples_valid as f64);
        if data.af >= args.min_maf
            && data.af <= 1.0 - args.min_maf
            && missing_rate <= args.max_missing_rate
        {
            all_passing.push(MarkerData {
                dosages: data.dosages,
                af: data.af,
                mac: data.mac,
            });
        }
    }
    info!(
        "{} markers pass QC (MAF >= {}, missing <= {})",
        all_passing.len(),
        args.min_maf,
        args.max_missing_rate,
    );

    // Select VR candidate indices: draw 1000 random indices (matching R SAIGE)
    // from markers with MAC >= 20, then deduplicate.
    use rand::Rng;
    use rand::SeedableRng;
    let mut vr_rng = rand_chacha::ChaCha8Rng::seed_from_u64(args.seed);
    let n_passing = all_passing.len();
    let mut vr_candidate_indices: Vec<usize> =
        (0..1000).map(|_| vr_rng.gen_range(0..n_passing)).collect();
    vr_candidate_indices.sort_unstable();
    vr_candidate_indices.dedup();
    // Only keep candidates with MAC >= 20
    vr_candidate_indices.retain(|&i| all_passing[i].mac >= 20.0);
    let vr_set: std::collections::HashSet<usize> = vr_candidate_indices.iter().copied().collect();

    // Split into GRM and VR pools (mutually exclusive, matching R SAIGE)
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

    // Build on-the-fly GRM (takes reference, copies internally)
    let grm = OnTheFlyGrm::new(&grm_dosages, &grm_afs);
    let grm_vec = move |v: &[f64]| -> Vec<f64> { grm.mat_vec(v) };

    // Fit null model using AI-REML
    let config = AiRemlConfig {
        max_iter: args.max_iter,
        tol: args.tol,
        pcg_tol: 1e-5,
        pcg_max_iter: 500,
        n_random_vectors: args.n_random_vectors,
        use_sparse_grm: args.use_sparse_grm,
        seed: args.seed,
    };

    info!("Fitting null model with AI-REML...");
    let reml_result = fit_ai_reml(&y, &x, grm_vec, trait_type, &config)?;

    info!(
        "AI-REML result: tau=[{:.6}, {:.6}], converged={}",
        reml_result.tau[0], reml_result.tau[1], reml_result.converged
    );

    // Compute XVX_inv_XV
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

    // Estimate variance ratios using PCG solver
    info!("Estimating variance ratios...");
    let tau = reml_result.tau;
    let mu_for_vr = reml_result.mu.clone();
    let w_for_vr = reml_result.working_weights.clone();

    // Reconstruct GRM for VR estimation (the first one was moved into grm_vec closure)
    let grm_for_vr = OnTheFlyGrm::new(&grm_dosages, &grm_afs);
    let pcg = PcgSolver::new(1e-5, 500);

    // Sigma^{-1} operator: solves Sigma * x = v where Sigma = tau_e * diag(1/W) + tau_g * GRM
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
        n_markers: args.n_markers_vr,
        min_mac: 20.0,
        use_categorical: args.use_categorical_vr,
        seed: args.seed,
        ploidy,
        ..Default::default()
    };

    let vr_result = if vr_dosages.is_empty() {
        info!("No markers available for VR estimation, using default VR=1.0");
        VarianceRatioResult {
            variance_ratio: 1.0,
            categorical_vr: Vec::new(),
            n_markers_used: 0,
            per_marker_vr: Vec::new(),
        }
    } else {
        estimate_variance_ratio(
            &vr_dosages,
            &vr_macs,
            &mu_for_vr,
            tau,
            trait_type,
            &x,
            &xvx_inv_xv,
            sigma_inv,
            &vr_config,
        )?
    };

    info!("Variance ratio: {:.6}", vr_result.variance_ratio);

    // Write variance ratio file
    let vr_path = std::path::Path::new(&args.output_prefix).with_extension("varianceRatio.txt");
    write_variance_ratio_file(&vr_result, &vr_path)?;
    info!("Variance ratio written to {}", vr_path.display());

    // Build null model
    let model = NullModel::new(
        trait_type,
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

    // Save model
    let model_path = std::path::Path::new(&args.output_prefix).with_extension("saige.model");
    serialization::save_model(&model, &model_path)?;
    info!("Model saved to {}", model_path.display());

    if args.save_json {
        let json_path =
            std::path::Path::new(&args.output_prefix).with_extension("saige.model.json");
        serialization::save_model_json(&model, &json_path)?;
        info!("JSON sidecar saved to {}", json_path.display());
    }

    // Print summary
    println!("{}", serialization::model_summary(&model));

    Ok(())
}
