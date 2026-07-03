//! Step 1: Fit the null GLMM.
//!
//! saige fit-null --plink-file ... --pheno-file ... --pheno-col ... --trait-type binary --output-prefix ...

use anyhow::Result;
use clap::Args;
use tracing::info;

use saige_core::glmm::link::TraitType;
use saige_core::glmm::variance_ratio::write_variance_ratio_file;
use saige_core::model::serialization;
use saige_geno::phenotype;
use saige_geno::plink::PlinkReader;
use saige_geno::sample;
use saige_geno::traits::{GenotypeReader, PloidyMode};

use super::pipeline::{fit_null_model, NullFitConfig};

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

    /// Skip variance-ratio estimation (use VR = 1.0). Useful when few variants
    /// will be tested (e.g. PheWAS) and avoids VR miscalibration.
    #[arg(long, default_value = "false")]
    skip_vr: bool,

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

    // Fit the null model (shared core).
    let cfg = NullFitConfig {
        trait_type,
        ploidy,
        min_maf: args.min_maf,
        max_missing_rate: args.max_missing_rate,
        n_random_vectors: args.n_random_vectors,
        n_markers_vr: args.n_markers_vr,
        max_iter: args.max_iter,
        tol: args.tol,
        seed: args.seed,
        use_sparse_grm: args.use_sparse_grm,
        use_categorical_vr: args.use_categorical_vr,
        skip_vr: args.skip_vr,
    };
    let model = fit_null_model(&mut plink, valid_ids, y, x_data, p, &cfg)?;

    // Write variance ratio file
    let vr_path = std::path::Path::new(&args.output_prefix).with_extension("varianceRatio.txt");
    write_variance_ratio_file(&model.variance_ratio, &vr_path)?;
    info!("Variance ratio written to {}", vr_path.display());

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
