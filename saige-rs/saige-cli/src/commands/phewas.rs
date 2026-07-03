//! PheWAS: test a small set of variants against many phenotypes.
//!
//! For each phenotype (phecode) this runs Step 1 (fit null GLMM) and Step 2
//! (single-variant association) in one process, parallelized across phenotypes.
//! The expensive per-phenotype work is the null fit; running them in one binary
//! with a shared parse and a rayon fan-out — plus optional `--skip-vr` — is the
//! optimization over invoking `fit-null` + `test` once per phecode.
//!
//! saige phewas --plink-file grm --pheno-file pheno.tsv \
//!   --phecodes P1,P2,... --covar-cols age,sex,PC1,PC2 \
//!   --test-vcf variants.vcf.gz --test-ploidy haploid --output-file out.tsv

use std::io::{BufWriter, Write};

use anyhow::{Context, Result};
use clap::Args;
use rayon::prelude::*;
use tracing::{info, warn};

use saige_core::glmm::link::TraitType;
use saige_core::score_test::single_variant::SingleVariantResult;
use saige_geno::traits::{GenotypeReader, PloidyMode};
use saige_geno::{phenotype, sample};
use saige_linalg::dense::DenseMatrix;

use super::pipeline::{build_engine, fit_null_model, NullFitConfig};

#[derive(Args)]
pub struct PheWasArgs {
    /// PLINK prefix used to build the GRM / fit Step 1 (nuclear, usually diploid)
    #[arg(long)]
    plink_file: String,

    /// Phenotype file path (one column per phecode)
    #[arg(long)]
    pheno_file: String,

    /// Phenotype (phecode) column names, comma-separated
    #[arg(long, default_value = "")]
    phecodes: String,

    /// File with one phecode column name per line (alternative to --phecodes)
    #[arg(long)]
    phecode_file: Option<String>,

    /// Covariate column names (comma-separated)
    #[arg(long, default_value = "")]
    covar_cols: String,

    /// Sample ID column name
    #[arg(long, default_value = "IID")]
    sample_id_col: String,

    /// Trait type: binary, quantitative, or survival
    #[arg(long, default_value = "binary")]
    trait_type: String,

    /// Combined output table path
    #[arg(long)]
    output_file: String,

    // ---- Step 2 test variants (choose one) ----
    /// VCF/BCF file of test variants
    #[arg(long)]
    test_vcf: Option<String>,

    /// PLINK prefix of test variants
    #[arg(long)]
    test_plink: Option<String>,

    /// BGEN file of test variants
    #[arg(long)]
    test_bgen: Option<String>,

    /// Ploidy for the TEST variants: "diploid", "haploid" (e.g. mtDNA), "auto"
    /// (CNV), or a positive number.
    #[arg(long, default_value = "diploid")]
    test_ploidy: String,

    // ---- Step 1 / GRM options ----
    /// Ploidy for the GRM / VR markers (usually diploid nuclear SNPs)
    #[arg(long, default_value = "diploid")]
    ploidy: String,

    /// Use sparse GRM
    #[arg(long, default_value = "false")]
    use_sparse_grm: bool,

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
    grm_min_maf: f64,

    /// Maximum missing rate for GRM markers
    #[arg(long, default_value = "0.15")]
    max_missing_rate: f64,

    /// Whether to use categorical variance ratios
    #[arg(long, default_value = "false")]
    use_categorical_vr: bool,

    /// Skip variance-ratio estimation (VR = 1.0) — recommended for PheWAS
    #[arg(long, default_value = "true")]
    skip_vr: bool,

    // ---- Step 2 test/filter options ----
    /// Apply SPA (binary traits)
    #[arg(long, default_value = "true")]
    is_spa: bool,

    /// Use the fast SPA approximation
    #[arg(long, default_value = "true")]
    is_fast_spa: bool,

    /// SPA p-value cutoff (only apply SPA if p < cutoff)
    #[arg(long, default_value = "0.05")]
    spa_pval_cutoff: f64,

    /// Apply Firth correction for the effect size (binary)
    #[arg(long, default_value = "false")]
    is_firth: bool,

    /// Firth p-value cutoff
    #[arg(long, default_value = "0.01")]
    firth_cutoff: f64,

    /// Minimum MAC filter for test variants
    #[arg(long, default_value = "0.5")]
    min_mac: f64,

    /// Minimum MAF filter for test variants
    #[arg(long, default_value = "0.0")]
    min_maf: f64,

    /// Restrict test variants to this chromosome (empty = all)
    #[arg(long, default_value = "")]
    chrom: String,
}

/// Open the test-variant reader according to which `--test-*` flag was given.
fn open_test_reader(args: &PheWasArgs) -> Result<Box<dyn GenotypeReader>> {
    if let Some(ref p) = args.test_vcf {
        Ok(Box::new(saige_geno::vcf::VcfReader::new(p)?))
    } else if let Some(ref p) = args.test_plink {
        Ok(Box::new(saige_geno::plink::PlinkReader::new(p)?))
    } else if let Some(ref p) = args.test_bgen {
        Ok(Box::new(saige_geno::bgen::BgenReader::new(p)?))
    } else {
        anyhow::bail!("Must specify one of --test-vcf, --test-plink, or --test-bgen")
    }
}

/// Resolve the phecode list from --phecodes or --phecode-file.
fn resolve_phecodes(args: &PheWasArgs) -> Result<Vec<String>> {
    if let Some(ref path) = args.phecode_file {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read phecode file: {path}"))?;
        Ok(contents
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect())
    } else {
        Ok(args
            .phecodes
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect())
    }
}

pub fn run(args: PheWasArgs) -> Result<()> {
    let trait_type = match args.trait_type.to_lowercase().as_str() {
        "binary" => TraitType::Binary,
        "quantitative" | "quant" => TraitType::Quantitative,
        "survival" => TraitType::Survival,
        _ => anyhow::bail!("Unknown trait type: {}", args.trait_type),
    };

    let phecodes = resolve_phecodes(&args)?;
    if phecodes.is_empty() {
        anyhow::bail!("No phecodes given; use --phecodes or --phecode-file");
    }
    let covar_cols: Vec<String> = if args.covar_cols.is_empty() {
        Vec::new()
    } else {
        args.covar_cols
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };

    let grm_ploidy = PloidyMode::parse(&args.ploidy)?;
    let test_ploidy = PloidyMode::parse(&args.test_ploidy)?;

    info!("=== SAIGE PheWAS ===");
    info!("{} phecodes, trait={:?}", phecodes.len(), trait_type);
    info!(
        "GRM ploidy: {:?}, test ploidy: {:?}",
        grm_ploidy, test_ploidy
    );
    if args.skip_vr {
        info!("Variance-ratio estimation is OFF (--skip-vr)");
    }

    // Fan out across phenotypes. Each task opens its own readers (mmap-backed,
    // so the OS page cache is shared), keeping the work Send-safe.
    let results: Vec<(String, Vec<SingleVariantResult>)> = phecodes
        .par_iter()
        .map(|phecode| {
            match run_one_phecode(
                &args,
                phecode,
                trait_type,
                &covar_cols,
                grm_ploidy,
                test_ploidy,
            ) {
                Ok(rows) => (phecode.clone(), rows),
                Err(e) => {
                    warn!("Phecode {phecode} failed: {e:#}");
                    (phecode.clone(), Vec::new())
                }
            }
        })
        .collect();

    // Write the combined table.
    let out = std::fs::File::create(&args.output_file)
        .with_context(|| format!("Failed to create output file: {}", args.output_file))?;
    let mut w = BufWriter::new(out);
    writeln!(
        w,
        "phenotype\tCHR\tPOS\tSNPID\tAllele1\tAllele2\tAC_Allele2\tAF_Allele2\tN\tBETA\tSE\tTstat\tp.value\tp.value.NA\tIs.SPA\tvarT\tvarTstar\tAF.Cases\tAF.Controls\tN.Cases\tN.Controls"
    )?;

    let mut n_rows = 0usize;
    for (phecode, rows) in &results {
        for r in rows {
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                phecode,
                r.chrom,
                r.pos,
                r.marker_id,
                r.ref_allele,
                r.alt_allele,
                r.ac,
                r.af,
                r.n,
                r.beta,
                r.se_beta,
                r.tstat,
                r.pvalue,
                r.pvalue_noadj,
                if r.is_spa { 1 } else { 0 },
                r.var_t,
                r.var_t_star,
                fmt_opt(r.af_cases),
                fmt_opt(r.af_controls),
                r.n_cases,
                r.n_controls,
            )?;
            n_rows += 1;
        }
    }
    w.flush()?;

    info!(
        "PheWAS complete: {} phecodes, {} result rows -> {}",
        phecodes.len(),
        n_rows,
        args.output_file
    );
    Ok(())
}

fn fmt_opt(v: f64) -> String {
    if v.is_nan() {
        "NA".to_string()
    } else {
        format!("{v}")
    }
}

/// Run Step 1 + Step 2 for a single phecode. Returns one row per tested variant.
fn run_one_phecode(
    args: &PheWasArgs,
    phecode: &str,
    trait_type: TraitType,
    covar_cols: &[String],
    grm_ploidy: PloidyMode,
    test_ploidy: PloidyMode,
) -> Result<Vec<SingleVariantResult>> {
    // Parse phenotype + covariates for this phecode.
    let pheno_data = phenotype::parse_phenotype_file(
        std::path::Path::new(&args.pheno_file),
        phecode,
        covar_cols,
        &args.sample_id_col,
    )?;

    // Open the GRM reader and intersect samples.
    let mut plink = saige_geno::plink::PlinkReader::new(&args.plink_file)?;
    let geno_ids = plink.sample_ids().to_vec();
    let intersection = sample::intersect_samples(&[&pheno_data.sample_ids, &geno_ids]);
    if intersection.ids.is_empty() {
        anyhow::bail!("no overlapping samples");
    }

    let valid_indices = phenotype::valid_sample_indices(&pheno_data);
    let valid_ids: Vec<String> = valid_indices
        .iter()
        .filter(|&&i| intersection.ids.contains(&pheno_data.sample_ids[i]))
        .map(|&i| pheno_data.sample_ids[i].clone())
        .collect();
    if valid_ids.is_empty() {
        anyhow::bail!("no valid samples after filtering");
    }

    plink.set_ploidy(grm_ploidy);
    plink.set_sample_subset(&valid_ids)?;

    // Build y and design matrix for valid samples.
    let valid_pheno_indices: Vec<usize> = valid_ids
        .iter()
        .map(|id| pheno_data.sample_ids.iter().position(|s| s == id).unwrap())
        .collect();
    let y: Vec<f64> = valid_pheno_indices
        .iter()
        .map(|&i| pheno_data.phenotype[i])
        .collect();
    let n = y.len();
    let p = covar_cols.len() + 1;
    let mut x_data = vec![0.0; n * p];
    for xi in x_data.iter_mut().take(n) {
        *xi = 1.0;
    }
    for (j, _) in covar_cols.iter().enumerate() {
        for (idx, &pheno_idx) in valid_pheno_indices.iter().enumerate() {
            x_data[(j + 1) * n + idx] = pheno_data.covariates[pheno_idx][j];
        }
    }

    // Step 1: fit the null model.
    let cfg = NullFitConfig {
        trait_type,
        ploidy: grm_ploidy,
        min_maf: args.grm_min_maf,
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

    // Step 2: test the variant set against this null model.
    let engine = build_engine(
        &model,
        args.is_spa,
        args.is_fast_spa,
        args.spa_pval_cutoff,
        test_ploidy,
    );

    let mut test_reader = open_test_reader(args)?;
    test_reader.set_ploidy(test_ploidy);
    test_reader.set_sample_subset(&model.sample_ids)?;

    let n_markers = test_reader.n_markers();
    let mut rows = Vec::new();
    for i in 0..n_markers {
        let marker = test_reader.read_marker(i as u64)?;
        if marker.mac < args.min_mac {
            continue;
        }
        if marker.af < args.min_maf || marker.af > 1.0 - args.min_maf {
            continue;
        }
        if !args.chrom.is_empty() && marker.info.chrom != args.chrom {
            continue;
        }

        let mut result = engine.test_marker(
            &marker.dosages,
            &marker.info.id,
            &marker.info.chrom,
            marker.info.pos,
            &marker.info.ref_allele,
            &marker.info.alt_allele,
        )?;

        // Firth effect-size (keeps the score/SPA p-value).
        if args.is_firth && trait_type == TraitType::Binary && result.pvalue < args.firth_cutoff {
            let firth_config = saige_core::firth::logistic::FirthConfig::default();
            let x = DenseMatrix::from_col_major(
                model.n_samples,
                model.n_covariates,
                model.x_flat.clone(),
            );
            if let Ok(fr) = saige_core::firth::logistic::firth_test_variant(
                &model.y,
                &x,
                &marker.dosages,
                &firth_config,
            ) {
                let fb = fr.beta[fr.beta.len() - 1];
                if fb.is_finite() {
                    result.beta = fb;
                    result.se_beta = fr.se[fr.se.len() - 1];
                }
            }
        }

        rows.push(result);
    }

    Ok(rows)
}
