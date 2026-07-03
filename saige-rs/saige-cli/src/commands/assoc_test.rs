//! Step 2: Association testing.
//!
//! saige test --bgen-file ... --model-file ... --output-file ... [--group-file ...]

use std::io::BufWriter;
use std::path::Path;

use anyhow::Result;
use clap::Args;
use tracing::info;

use saige_core::glmm::link::TraitType;
use saige_core::model::serialization::load_model;
use saige_core::score_test::region::{
    self, write_region_header, write_region_line, RegionTestConfig, RegionTestResult,
};
use saige_core::score_test::single_variant::{
    write_result_line, write_results_header, ScoreTestEngine,
};
use saige_geno::group_file::GroupFile;
use saige_geno::traits::GenotypeReader;
use saige_linalg::dense::DenseMatrix;

#[derive(Args)]
pub struct AssocTestArgs {
    /// Model file from Step 1 (.saige.model)
    #[arg(long)]
    model_file: String,

    /// Output file path
    #[arg(long)]
    output_file: String,

    /// BGEN file path
    #[arg(long)]
    bgen_file: Option<String>,

    /// PLINK file prefix
    #[arg(long)]
    plink_file: Option<String>,

    /// VCF file path
    #[arg(long)]
    vcf_file: Option<String>,

    /// Group file for region-based tests
    #[arg(long)]
    group_file: Option<String>,

    /// Whether to apply SPA
    #[arg(long, default_value = "true")]
    is_spa: bool,

    /// Whether to use fast SPA
    #[arg(long, default_value = "true")]
    is_fast_spa: bool,

    /// SPA p-value cutoff (only apply SPA if p < cutoff)
    #[arg(long, default_value = "0.05")]
    spa_pval_cutoff: f64,

    /// Use Firth correction
    #[arg(long, default_value = "false")]
    is_firth: bool,

    /// Firth p-value cutoff
    #[arg(long, default_value = "0.01")]
    firth_cutoff: f64,

    /// Minimum MAC filter
    #[arg(long, default_value = "0.5")]
    min_mac: f64,

    /// Minimum MAF filter
    #[arg(long, default_value = "0.0")]
    min_maf: f64,

    /// Minimum info score filter
    #[arg(long, default_value = "0.0")]
    min_info: f64,

    /// Chromosome to test (empty = all)
    #[arg(long, default_value = "")]
    chrom: String,

    /// LOCO chromosome (use LOCO results for this chromosome)
    #[arg(long)]
    loco_chrom: Option<String>,
}

pub fn run(args: AssocTestArgs) -> Result<()> {
    info!("=== SAIGE Step 2: Association Testing ===");

    // Load model
    let model = load_model(Path::new(&args.model_file))?;
    info!(
        "Loaded model: {} samples, trait={:?}, VR={:.4}",
        model.n_samples, model.trait_type, model.variance_ratio.variance_ratio
    );

    // Open genotype file
    let mut reader: Box<dyn GenotypeReader> = if let Some(ref bgen_path) = args.bgen_file {
        Box::new(saige_geno::bgen::BgenReader::new(bgen_path)?)
    } else if let Some(ref plink_path) = args.plink_file {
        Box::new(saige_geno::plink::PlinkReader::new(plink_path)?)
    } else if let Some(ref vcf_path) = args.vcf_file {
        Box::new(saige_geno::vcf::VcfReader::new(vcf_path)?)
    } else {
        anyhow::bail!("Must specify --bgen-file, --plink-file, or --vcf-file");
    };

    info!(
        "Genotype file: {} markers x {} samples",
        reader.n_markers(),
        reader.n_samples()
    );

    // Set sample subset to match model
    reader.set_sample_subset(&model.sample_ids)?;

    // Build score test engine
    let n = model.n_samples;
    let p = model.n_covariates;
    let x = DenseMatrix::from_col_major(n, p, model.x_flat.clone());
    let xvx_inv_xv = DenseMatrix::from_col_major(p, n, model.xvx_inv_xv_flat.clone());

    let engine = ScoreTestEngine {
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
        use_spa: args.is_spa && model.trait_type == TraitType::Binary,
        use_fast_spa: args.is_fast_spa,
        spa_tol: 1e-6,
        spa_pval_cutoff: args.spa_pval_cutoff,
        y: if model.trait_type == TraitType::Binary {
            Some(model.y.clone())
        } else {
            None
        },
    };

    // Check if group file is provided for region-based tests
    if let Some(ref group_path) = args.group_file {
        return run_region_tests(&args, &engine, &mut reader, group_path, &model);
    }

    // Single-variant testing
    let output_file = std::fs::File::create(&args.output_file)?;
    let mut writer = BufWriter::new(output_file);
    write_results_header(&mut writer)?;

    let n_markers = reader.n_markers();
    info!("Testing {} markers...", n_markers);

    let mut n_tested = 0;
    let mut n_skipped = 0;

    for i in 0..n_markers {
        let marker_data = reader.read_marker(i as u64)?;

        // Apply filters
        if marker_data.mac < args.min_mac {
            n_skipped += 1;
            continue;
        }
        if marker_data.af < args.min_maf || marker_data.af > 1.0 - args.min_maf {
            n_skipped += 1;
            continue;
        }
        if !args.chrom.is_empty() && marker_data.info.chrom != args.chrom {
            n_skipped += 1;
            continue;
        }

        // Run score test
        let mut result = engine.test_marker(
            &marker_data.dosages,
            &marker_data.info.id,
            &marker_data.info.chrom,
            marker_data.info.pos,
            &marker_data.info.ref_allele,
            &marker_data.info.alt_allele,
        )?;

        // Apply Firth correction for binary traits when p < cutoff
        if args.is_firth
            && model.trait_type == TraitType::Binary
            && result.pvalue < args.firth_cutoff
        {
            let firth_config = saige_core::firth::logistic::FirthConfig::default();
            let firth_result = saige_core::firth::logistic::firth_test_variant(
                &model.y,
                &DenseMatrix::from_col_major(n, p, model.x_flat.clone()),
                &marker_data.dosages,
                &firth_config,
            );
            // Firth is used for effect-size (BETA/SE) estimation; the
            // association p-value from the score test / SPA is retained
            // (matching R SAIGE's `is_Firth_beta` semantics). Adopt the Firth
            // beta whenever it is finite — including from a non-converged fit —
            // so we report a bias-reduced effect size instead of the
            // ill-conditioned score-test beta (S/var), which explodes when the
            // dosage has tiny dispersion.
            if let Ok(fr) = firth_result {
                let fb = fr.beta[fr.beta.len() - 1];
                let fse = fr.se[fr.se.len() - 1];
                if fb.is_finite() {
                    result.beta = fb;
                    result.se_beta = fse;
                    if !fr.converged {
                        tracing::warn!(
                            "Firth did not fully converge for marker {}; reporting last-iterate beta={:.6}",
                            result.marker_id,
                            fb
                        );
                    }
                }
            }
        }

        write_result_line(&mut writer, &result)?;
        n_tested += 1;
    }

    info!(
        "Testing complete: {} markers tested, {} skipped",
        n_tested, n_skipped
    );
    info!("Results written to {}", args.output_file);

    Ok(())
}

/// Run region-based (gene-based) tests using BURDEN, SKAT, and SKAT-O.
fn run_region_tests(
    args: &AssocTestArgs,
    engine: &ScoreTestEngine,
    reader: &mut Box<dyn GenotypeReader>,
    group_path: &str,
    model: &saige_core::model::null_model::NullModel,
) -> Result<()> {
    info!("Running region-based tests with group file: {}", group_path);

    let group_file = GroupFile::parse(group_path)?;
    let gene_names = group_file.gene_names();
    info!("{} genes to test", gene_names.len());

    // Build variant ID -> marker index map
    let mut variant_id_map = std::collections::HashMap::new();
    for i in 0..reader.n_markers() {
        let marker_info = reader.marker_info(i as u64)?;
        variant_id_map.insert(marker_info.id.clone(), i);
    }

    let config = RegionTestConfig::default();

    // Open output file
    let output_file = std::fs::File::create(&args.output_file)?;
    let mut writer = BufWriter::new(output_file);
    write_region_header(&mut writer)?;

    let mut n_tested = 0;
    let mut n_skipped = 0;

    for gene in &gene_names {
        let group = match group_file.group_for_gene(gene) {
            Some(g) => g,
            None => {
                n_skipped += 1;
                continue;
            }
        };

        // Read genotype dosages for each variant in this gene
        let mut genotypes: Vec<Vec<f64>> = Vec::new();
        let mut marker_ids: Vec<String> = Vec::new();
        let mut marker_afs: Vec<f64> = Vec::new();
        let mut mac_values: Vec<f64> = Vec::new();

        for vid in &group.variant_ids {
            if let Some(&idx) = variant_id_map.get(vid) {
                let data = reader.read_marker(idx as u64)?;

                // Apply MAC filter
                if data.mac < args.min_mac {
                    continue;
                }
                // Apply MAF filter
                if data.af < args.min_maf || data.af > 1.0 - args.min_maf {
                    continue;
                }

                genotypes.push(data.dosages);
                marker_ids.push(vid.clone());
                marker_afs.push(data.af);
                mac_values.push(data.mac);
            }
        }

        if genotypes.is_empty() {
            info!("Gene {}: no variants passed filters, skipping", gene);
            n_skipped += 1;
            continue;
        }

        // Compute Beta(MAF, 1, 25) weights
        let weights = region::beta_weights(&marker_afs, 1.0, 25.0);

        // MAC category counts
        let mac_cats = region::mac_category_counts(&mac_values, &config.mac_categories);

        // Run BURDEN, SKAT, SKAT-O tests
        let (p_skato, p_burden, p_skat) =
            region::run_region_test(&genotypes, &weights, engine, &config)?;

        let result = RegionTestResult {
            region_name: gene.clone(),
            pvalue: p_skato,
            mac_category_counts: mac_cats,
            marker_ids,
            marker_afs,
            burden_pvalue: p_burden,
            skat_pvalue: p_skat,
            // NA-adjusted p-values: for now, same as unadjusted
            // (SPA adjustment for region tests would need more work)
            pvalue_na: p_skato,
            burden_pvalue_na: p_burden,
            skat_pvalue_na: p_skat,
        };

        write_region_line(&mut writer, &result)?;
        n_tested += 1;

        if n_tested % 100 == 0 {
            info!("Tested {} genes...", n_tested);
        }
    }

    info!(
        "Region testing complete: {} genes tested, {} skipped",
        n_tested, n_skipped
    );
    info!("Results written to {}", args.output_file);

    // Also write per-marker (single-variant) results if model is available
    let _ = model; // suppress unused warning

    Ok(())
}
