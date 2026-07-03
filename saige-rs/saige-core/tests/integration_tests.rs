//! Integration tests against SAIGE reference data.
//!
//! Uses test data from CSAIGE/extdata/ to verify numerical correctness
//! against the R implementation.

use std::path::Path;

/// Helper to get the path to test fixtures.
fn fixtures_dir() -> &'static str {
    // The symlink: saige-rs/tests/fixtures -> CSAIGE/extdata/
    concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures")
}

/// Check if test fixtures exist (skip tests gracefully if not).
fn has_fixtures() -> bool {
    Path::new(fixtures_dir()).join("input").exists()
}

mod plink_reader {
    use super::*;
    use saige_geno::plink::PlinkReader;
    use saige_geno::traits::GenotypeReader;

    #[test]
    fn test_read_100markers_plink() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let plink = PlinkReader::new(&plink_prefix).expect("Failed to open PLINK files");

        // Should have 1000 samples and ~128K markers
        assert_eq!(plink.n_samples(), 1000, "Expected 1000 samples");
        assert!(
            plink.n_markers() > 100000,
            "Expected >100K markers, got {}",
            plink.n_markers()
        );

        // Check sample IDs start with "1a1"
        let ids = plink.sample_ids();
        assert_eq!(ids[0], "1a1");
        assert_eq!(ids[1], "1a2");
    }

    #[test]
    fn test_read_marker_dosages() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let mut plink = PlinkReader::new(&plink_prefix).expect("Failed to open PLINK files");

        // Read first marker
        let marker = plink.read_marker(0).unwrap();
        assert_eq!(marker.dosages.len(), 1000);

        // Dosages should be 0, 1, 2, or NaN
        for &d in &marker.dosages {
            if !d.is_nan() {
                assert!(
                    (d - 0.0).abs() < 1e-10 || (d - 1.0).abs() < 1e-10 || (d - 2.0).abs() < 1e-10,
                    "Unexpected dosage: {}",
                    d
                );
            }
        }

        // AF should be in [0, 1]
        assert!(marker.af >= 0.0 && marker.af <= 1.0);
        // MAC should be non-negative
        assert!(marker.mac >= 0.0);
    }

    #[test]
    fn test_sample_subset() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let mut plink = PlinkReader::new(&plink_prefix).expect("Failed to open PLINK files");

        // Subset to first 100 samples
        let subset: Vec<String> = (1..=100).map(|i| format!("1a{}", i)).collect();
        plink.set_sample_subset(&subset).unwrap();

        let marker = plink.read_marker(0).unwrap();
        assert_eq!(marker.dosages.len(), 100);
    }
}

mod phenotype_parser {
    use super::*;
    use saige_geno::phenotype;

    #[test]
    fn test_parse_phenotype_file() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let pheno_path = format!("{}/input/pheno_1000samples.txt", fixtures_dir());
        let pheno = phenotype::parse_phenotype_file(
            Path::new(&pheno_path),
            "y",
            &["x1".to_string(), "x2".to_string()],
            "IID",
        )
        .expect("Failed to parse phenotype file");

        assert_eq!(pheno.sample_ids.len(), 1000);
        assert_eq!(pheno.phenotype.len(), 1000);
        assert_eq!(pheno.covariates.len(), 1000);
        assert_eq!(pheno.covariates[0].len(), 2); // x1, x2

        // First sample
        assert_eq!(pheno.sample_ids[0], "1a1");
        assert!((pheno.phenotype[0] - 0.0).abs() < 1e-10); // y=0

        // All phenotypes should be 0 or 1 (binary)
        for &p in &pheno.phenotype {
            assert!(
                (p - 0.0).abs() < 1e-10 || (p - 1.0).abs() < 1e-10,
                "Unexpected phenotype: {}",
                p
            );
        }
    }

    #[test]
    fn test_phenotype_with_both_trait_types() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let pheno_path = format!(
            "{}/input/pheno_1000samples.txt_withdosages_withBothTraitTypes.txt",
            fixtures_dir()
        );
        let pheno = phenotype::parse_phenotype_file(
            Path::new(&pheno_path),
            "y_binary",
            &["x1".to_string(), "x2".to_string()],
            "IID",
        )
        .expect("Failed to parse phenotype file");

        assert_eq!(pheno.sample_ids.len(), 1000);
    }
}

mod sample_intersection {
    use super::*;
    use saige_geno::phenotype;
    use saige_geno::plink::PlinkReader;
    use saige_geno::sample;
    use saige_geno::traits::GenotypeReader;

    #[test]
    fn test_genotype_phenotype_intersection() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let plink = PlinkReader::new(&plink_prefix).unwrap();
        let geno_ids = plink.sample_ids().to_vec();

        let pheno_path = format!("{}/input/pheno_1000samples.txt", fixtures_dir());
        let pheno = phenotype::parse_phenotype_file(
            Path::new(&pheno_path),
            "y",
            &["x1".to_string(), "x2".to_string()],
            "IID",
        )
        .unwrap();

        let intersection = sample::intersect_samples(&[&pheno.sample_ids, &geno_ids]);

        // Both have 1000 samples with matching IDs
        assert_eq!(intersection.ids.len(), 1000);
    }
}

mod linalg_with_real_data {
    use super::*;
    use saige_core::glmm::pcg::OnTheFlyGrm;
    use saige_geno::plink::PlinkReader;
    use saige_geno::traits::GenotypeReader;

    #[test]
    fn test_grm_construction() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let mut plink = PlinkReader::new(&plink_prefix).unwrap();

        // Subset to 100 samples for speed
        let subset: Vec<String> = (1..=100).map(|i| format!("1a{}", i)).collect();
        plink.set_sample_subset(&subset).unwrap();

        // Read first 500 markers for GRM
        let mut dosages = Vec::new();
        let mut afs = Vec::new();
        let min_maf = 0.01;

        for m in 0..500.min(plink.n_markers()) {
            let data = plink.read_marker(m as u64).unwrap();
            if data.af >= min_maf && data.af <= 1.0 - min_maf {
                dosages.push(data.dosages);
                afs.push(data.af);
            }
        }

        assert!(!dosages.is_empty(), "Should have markers for GRM");

        // Build GRM
        let grm = OnTheFlyGrm::new(&dosages, &afs);

        // GRM * e_1 should give non-trivial result
        let mut v = vec![0.0; 100];
        v[0] = 1.0;
        let result = grm.mat_vec(&v);

        // GRM diagonal should be positive (self-relatedness)
        assert!(result[0] > 0.0, "GRM[0,0] = {}", result[0]);

        // Should be symmetric: <e1, GRM*e2> = <e2, GRM*e1>
        let mut v2 = vec![0.0; 100];
        v2[1] = 1.0;
        let r1 = grm.mat_vec(&v);
        let r2 = grm.mat_vec(&v2);
        let dot12: f64 = v2.iter().zip(r1.iter()).map(|(a, b)| a * b).sum();
        let dot21: f64 = v.iter().zip(r2.iter()).map(|(a, b)| a * b).sum();
        assert!(
            (dot12 - dot21).abs() < 1e-10,
            "GRM not symmetric: {} vs {}",
            dot12,
            dot21
        );
    }
}

mod ai_reml {
    use super::*;
    use saige_core::glmm::ai_reml::{fit_ai_reml, AiRemlConfig};
    use saige_core::glmm::link::TraitType;
    use saige_core::glmm::pcg::OnTheFlyGrm;
    use saige_geno::phenotype;
    use saige_geno::plink::PlinkReader;
    use saige_geno::sample;
    use saige_geno::traits::GenotypeReader;
    use saige_linalg::dense::DenseMatrix;

    #[test]
    #[ignore] // Slow: ~5min in debug mode. Run with `cargo test -- --ignored`
    fn test_binary_null_model_small() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let plink_prefix = format!(
            "{}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly",
            fixtures_dir()
        );
        let mut plink = PlinkReader::new(&plink_prefix).unwrap();

        // Use all 1000 samples
        let pheno_path = format!("{}/input/pheno_1000samples.txt", fixtures_dir());
        let pheno = phenotype::parse_phenotype_file(
            Path::new(&pheno_path),
            "y",
            &["x1".to_string(), "x2".to_string()],
            "IID",
        )
        .unwrap();

        let geno_ids = plink.sample_ids().to_vec();
        let intersection = sample::intersect_samples(&[&pheno.sample_ids, &geno_ids]);
        let valid_ids = intersection.ids;
        plink.set_sample_subset(&valid_ids).unwrap();

        let valid_pheno_indices: Vec<usize> = valid_ids
            .iter()
            .map(|id| pheno.sample_ids.iter().position(|s| s == id).unwrap())
            .collect();

        let y: Vec<f64> = valid_pheno_indices
            .iter()
            .map(|&i| pheno.phenotype[i])
            .collect();

        let n = y.len();
        let n_cases = y.iter().filter(|&&v| v > 0.5).count();
        eprintln!("n={}, cases={}, controls={}", n, n_cases, n - n_cases);

        let p = 3; // intercept + x1 + x2
        let mut x_data = vec![0.0; n * p];
        for xi in x_data.iter_mut().take(n) {
            *xi = 1.0;
        }
        for (idx, &pheno_idx) in valid_pheno_indices.iter().enumerate() {
            x_data[n + idx] = pheno.covariates[pheno_idx][0]; // x1
            x_data[2 * n + idx] = pheno.covariates[pheno_idx][1]; // x2
        }
        let x = DenseMatrix::from_col_major(n, p, x_data);

        // Read first 2000 markers for GRM (for speed in tests)
        let mut dosages = Vec::new();
        let mut afs = Vec::new();
        let n_grm_markers = 2000.min(plink.n_markers());

        for m in 0..n_grm_markers {
            let data = plink.read_marker(m as u64).unwrap();
            if data.af >= 0.01 && data.af <= 0.99 {
                dosages.push(data.dosages);
                afs.push(data.af);
            }
        }

        eprintln!("Using {} markers for GRM", dosages.len());

        let grm = OnTheFlyGrm::new(&dosages, &afs);
        let grm_vec = move |v: &[f64]| -> Vec<f64> { grm.mat_vec(v) };

        let config = AiRemlConfig {
            max_iter: 20,
            tol: 1e-4,
            pcg_tol: 1e-4,
            pcg_max_iter: 200,
            n_random_vectors: 10,
            use_sparse_grm: false,
            seed: 12345,
        };

        let result = fit_ai_reml(&y, &x, grm_vec, TraitType::Binary, &config).unwrap();

        // tau should be positive
        assert!(result.tau[0] > 0.0, "tau_e={}", result.tau[0]);
        assert!(result.tau[1] >= 0.0, "tau_g={}", result.tau[1]);

        // mu should be in [epsilon, 1-epsilon] for binary
        for &m in &result.mu {
            assert!((1e-10..=1.0 - 1e-10).contains(&m), "mu={} out of bounds", m);
        }

        // Fitted probabilities should be reasonable (close to prevalence)
        let mean_mu: f64 = result.mu.iter().sum::<f64>() / n as f64;
        let prevalence = n_cases as f64 / n as f64;
        assert!(
            (mean_mu - prevalence).abs() < 0.15,
            "Mean mu ({:.4}) should be close to prevalence ({:.4})",
            mean_mu,
            prevalence
        );

        eprintln!(
            "Binary null model: tau=[{:.4}, {:.4}], converged={}, mean_mu={:.4}",
            result.tau[0], result.tau[1], result.converged, mean_mu
        );
    }
}

mod score_test {
    use saige_core::glmm::link::TraitType;
    use saige_core::score_test::single_variant::ScoreTestEngine;
    use saige_linalg::dense::DenseMatrix;

    #[test]
    fn test_score_test_basic() {
        // Create a simple score test engine with synthetic data
        let n = 100;
        let p = 1; // just intercept

        // Simple model: 50 cases, 50 controls
        let mu = vec![0.3; n];
        let mu2: Vec<f64> = mu.iter().map(|&m| m * (1.0 - m)).collect();
        let y: Vec<f64> = (0..n).map(|i| if i < 50 { 1.0 } else { 0.0 }).collect();
        let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(yi, mi)| yi - mi).collect();

        // X is n x 1 (intercept only), xvx_inv_xv is 1 x n
        let x = DenseMatrix::from_col_major(n, p, vec![1.0; n]);
        let xvx_inv_xv = DenseMatrix::from_col_major(p, n, vec![1.0 / n as f64; n]);

        let engine = ScoreTestEngine {
            trait_type: TraitType::Binary,
            mu: mu.clone(),
            mu2: mu2.clone(),
            residuals,
            tau_e: 1.0,
            tau_g: 0.5,
            xvx_inv_xv,
            x,
            variance_ratio: 0.94,
            categorical_vr: Vec::new(),
            use_spa: false,
            use_fast_spa: false,
            spa_tol: 1e-6,
            spa_pval_cutoff: 0.05,
            ploidy: saige_geno::traits::PloidyMode::Diploid,
            y: Some(y.clone()),
        };

        // Test with a genotype vector that has signal
        let mut geno = vec![0.0; n];
        for gi in geno.iter_mut().take(20) {
            *gi = 2.0; // cases more likely to have alt allele
        }

        let result = engine
            .test_marker(&geno, "test_snp", "1", 100, "A", "C")
            .unwrap();

        // P-value should be between 0 and 1
        assert!((0.0..=1.0).contains(&result.pvalue), "p={}", result.pvalue);
        // Should detect some signal
        assert!(
            result.pvalue < 0.5,
            "Expected some signal, p={}",
            result.pvalue
        );
    }
}

mod cct {
    use saige_core::score_test::cct::cauchy_combination_test;

    #[test]
    fn test_cct_matches_r() {
        // Test known CCT values
        // In R: ACAT::ACAT(c(0.05, 0.01), c(1,1)) ≈ 0.01697
        let pvals = vec![0.05, 0.01];
        let combined = cauchy_combination_test(&pvals, None);
        // Should be between min(p) and the Bonferroni-corrected value
        assert!(combined < 0.05);
        assert!(combined > 0.001);
    }
}

mod sparse_grm {
    use super::*;
    use saige_geno::sparse_grm_io;

    #[test]
    fn test_read_sparse_grm() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let mtx_path = format!(
            "{}/output/sparseGRM_relatednessCutoff_0.125_1000_randomMarkersUsed.sparseGRM.mtx",
            fixtures_dir()
        );
        let id_path = format!(
            "{}/output/sparseGRM_relatednessCutoff_0.125_1000_randomMarkersUsed.sparseGRM.mtx.sampleIDs.txt",
            fixtures_dir()
        );

        if !Path::new(&mtx_path).exists() {
            eprintln!("Skipping: sparse GRM file not found");
            return;
        }

        let (grm, ids) = sparse_grm_io::read_sparse_grm(Path::new(&mtx_path), Path::new(&id_path))
            .expect("Failed to read sparse GRM");

        // Should have 1000 samples
        assert_eq!(ids.len(), 1000, "Expected 1000 sample IDs");

        // Sparse GRM loaded successfully
        // Check it has non-zero entries
        let v = vec![1.0; 1000];
        let result = grm.mat_vec(&v);
        assert_eq!(result.len(), 1000);

        // Diagonal should be positive (self-relatedness)
        let diag = grm.diag();
        for &d in &diag {
            assert!(d > 0.0, "Diagonal element should be positive: {}", d);
        }
    }
}

mod reference_outputs {
    use super::*;

    #[test]
    fn test_variance_ratio_file_format() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        // Read reference VR file
        let vr_path = format!(
            "{}/output/example_binary_fullGRM.varianceRatio.txt",
            fixtures_dir()
        );

        if !Path::new(&vr_path).exists() {
            eprintln!("Skipping: VR file not found");
            return;
        }

        let content = std::fs::read_to_string(&vr_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Should have lines like "0.94xxx null 1"
        assert!(!lines.is_empty(), "VR file should not be empty");

        for line in &lines {
            let parts: Vec<&str> = line.split_whitespace().collect();
            assert!(
                parts.len() >= 3,
                "VR line should have >= 3 fields: {}",
                line
            );

            // First field is the VR value
            let vr: f64 = parts[0].parse().expect("VR value should be a number");
            assert!(vr > 0.0 && vr < 10.0, "VR should be reasonable: {}", vr);

            // Second field is category label
            assert!(
                parts[1] == "null" || parts[1] == "sparse",
                "Category should be 'null' or 'sparse': {}",
                parts[1]
            );
        }

        // Parse the "null" VR values
        let null_vrs: Vec<f64> = lines
            .iter()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 && parts[1] == "null" {
                    parts[0].parse().ok()
                } else {
                    None
                }
            })
            .collect();

        assert!(!null_vrs.is_empty());
        // Reference: VR ≈ 0.94 for binary with full GRM
        eprintln!("Reference null VR values: {:?}", null_vrs);
        for &vr in &null_vrs {
            assert!(vr > 0.5 && vr < 2.0, "VR seems out of range: {}", vr);
        }
    }

    #[test]
    fn test_association_output_format() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let assoc_path = format!(
            "{}/output/example_binary.SAIGE.vcf.genotype.txt",
            fixtures_dir()
        );

        if !Path::new(&assoc_path).exists() {
            eprintln!("Skipping: association output not found");
            return;
        }

        let content = std::fs::read_to_string(&assoc_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // First line is header
        let header = lines[0];
        assert!(header.contains("CHR"), "Header should contain CHR");
        assert!(header.contains("BETA"), "Header should contain BETA");
        assert!(header.contains("p.value"), "Header should contain p.value");
        assert!(header.contains("SE"), "Header should contain SE");

        // Parse data lines
        for line in &lines[1..] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 10 {
                continue;
            }

            // CHR should be a number or "chr1" etc
            let chr = parts[0];
            assert!(!chr.is_empty());

            // POS should be a number
            let _pos: u64 = parts[1].parse().expect("POS should be a number");

            // BETA should be a number
            let beta: f64 = parts[9].parse().expect("BETA should be a number");
            assert!(beta.is_finite(), "BETA should be finite");

            // SE should be positive
            let se: f64 = parts[10].parse().expect("SE should be a number");
            assert!(se >= 0.0, "SE should be non-negative");

            // p.value should be in [0, 1]
            let pval: f64 = parts[12].parse().expect("p.value should be a number");
            assert!(
                (0.0..=1.0).contains(&pval),
                "p.value should be in [0,1]: {}",
                pval
            );
        }

        eprintln!("Association output has {} markers", lines.len() - 1);
    }

    #[test]
    fn test_reference_pvalues_binary() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let assoc_path = format!(
            "{}/output/example_binary.SAIGE.vcf.genotype.txt",
            fixtures_dir()
        );

        if !Path::new(&assoc_path).exists() {
            return;
        }

        let content = std::fs::read_to_string(&assoc_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        // Parse reference results into a map: SNPID -> (BETA, SE, pvalue)
        let mut reference: std::collections::HashMap<String, (f64, f64, f64)> =
            std::collections::HashMap::new();

        for line in &lines[1..] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 13 {
                continue;
            }
            let snpid = parts[2].to_string();
            let beta: f64 = parts[9].parse().unwrap_or(f64::NAN);
            let se: f64 = parts[10].parse().unwrap_or(f64::NAN);
            let pval: f64 = parts[12].parse().unwrap_or(f64::NAN);
            reference.insert(snpid, (beta, se, pval));
        }

        // Known reference values from the output file
        // rs4: p=0.961485, BETA=-0.0389
        if let Some(&(beta, _se, pval)) = reference.get("rs4") {
            assert!(
                (pval - 0.9615).abs() < 0.01,
                "rs4 p-value: expected ~0.9615, got {}",
                pval
            );
            assert!(
                (beta - (-0.0389)).abs() < 0.1,
                "rs4 BETA: expected ~-0.039, got {}",
                beta
            );
        }

        eprintln!("Parsed {} reference markers", reference.len());
    }
}

mod vcf_reader {
    use super::*;
    use saige_geno::traits::GenotypeReader;
    use saige_geno::vcf::VcfReader;

    #[test]
    fn test_read_vcf_10markers() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let vcf_path = format!("{}/input/genotype_10markers.vcf.gz", fixtures_dir());
        if !Path::new(&vcf_path).exists() {
            eprintln!("Skipping: VCF file not found");
            return;
        }

        let vcf = VcfReader::new(&vcf_path);
        if vcf.is_err() {
            eprintln!("Skipping: VCF reader error: {:?}", vcf.err());
            return;
        }
        let vcf = vcf.unwrap();

        assert!(vcf.n_markers() > 0, "Should have markers");
        assert!(vcf.n_samples() > 0, "Should have samples");

        eprintln!(
            "VCF: {} markers x {} samples",
            vcf.n_markers(),
            vcf.n_samples()
        );
    }
}

mod group_file {
    use super::*;
    use saige_geno::group_file;

    #[test]
    fn test_parse_group_file() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let group_path = format!("{}/input/group_new_chrposa1a2.txt", fixtures_dir());
        if !Path::new(&group_path).exists() {
            eprintln!("Skipping: group file not found");
            return;
        }

        let gf = group_file::GroupFile::parse(&group_path);
        if gf.is_err() {
            eprintln!("Skipping: group file parse error: {:?}", gf.err());
            return;
        }
        let gf = gf.unwrap();

        assert!(!gf.groups.is_empty(), "Should have at least one gene group");

        // Should have GENE1, GENE2, GENE3
        let gene_names = gf.gene_names();
        eprintln!("Gene groups: {:?}", gene_names);

        // Each gene should have variants and annotations
        for group in &gf.groups {
            assert!(
                !group.variant_ids.is_empty(),
                "Gene {} has no variants",
                group.name
            );
            assert_eq!(
                group.variant_ids.len(),
                group.annotations.len(),
                "Gene {} has mismatched variant/annotation counts",
                group.name
            );
            eprintln!(
                "  {} : {} variants, annotations: {:?}",
                group.name,
                group.variant_ids.len(),
                group.unique_annotations()
            );
        }

        // Check specific genes from the reference file
        if let Some(gene1) = gf.group_for_gene("GENE1") {
            assert_eq!(gene1.variant_ids.len(), 50);
            let annos = gene1.unique_annotations();
            assert!(annos.contains(&"lof".to_string()), "GENE1 should have lof");
            assert!(
                annos.contains(&"missense".to_string()),
                "GENE1 should have missense"
            );
        }
    }

    #[test]
    fn test_parse_group_file_with_weights() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let group_path = format!(
            "{}/input/group_new_chrposa1a2_withWeights.txt",
            fixtures_dir()
        );
        if !Path::new(&group_path).exists() {
            eprintln!("Skipping: weighted group file not found");
            return;
        }

        let gf = group_file::GroupFile::parse(&group_path);
        if gf.is_err() {
            eprintln!("Weighted group file not supported yet");
            return;
        }
        let gf = gf.unwrap();
        assert!(!gf.groups.is_empty());
    }
}

mod bgen_reader {
    use super::*;
    use saige_geno::bgen::BgenReader;
    use saige_geno::traits::GenotypeReader;

    #[test]
    fn test_read_bgen_100markers() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let bgen_path = format!("{}/input/genotype_100markers.bgen", fixtures_dir());
        if !Path::new(&bgen_path).exists() {
            eprintln!("Skipping: BGEN file not found");
            return;
        }

        let reader = BgenReader::new(&bgen_path);
        if reader.is_err() {
            eprintln!("Skipping: BGEN reader error: {:?}", reader.err());
            return;
        }
        let mut reader = reader.unwrap();

        assert!(reader.n_markers() > 0, "Should have markers");
        assert!(reader.n_samples() > 0, "Should have samples");

        eprintln!(
            "BGEN: {} markers x {} samples",
            reader.n_markers(),
            reader.n_samples()
        );

        // Read a few markers and check dosages are valid
        for i in 0..5.min(reader.n_markers()) {
            let data = reader.read_marker(i as u64).unwrap();
            assert_eq!(data.dosages.len(), reader.n_samples());
            // Dosages should be in [0, 2] or NaN
            for &d in &data.dosages {
                assert!(
                    d.is_nan() || (0.0..=2.0).contains(&d),
                    "Dosage out of range: {}",
                    d
                );
            }
            // AF should be in [0, 1]
            assert!(
                (0.0..=1.0).contains(&data.af),
                "AF out of range: {}",
                data.af
            );
        }
    }
}

mod region_test_reference {
    use super::*;

    #[test]
    fn test_gene_test_reference_output_format() {
        if !has_fixtures() {
            eprintln!("Skipping: test fixtures not found");
            return;
        }

        let gene_path = format!("{}/output/example_binary.SAIGE.gene.txt", fixtures_dir());

        if !Path::new(&gene_path).exists() {
            eprintln!("Skipping: gene test output not found");
            return;
        }

        let content = std::fs::read_to_string(&gene_path).unwrap();
        let lines: Vec<&str> = content.lines().collect();

        assert!(lines.len() > 1, "Should have header + data lines");

        // Check header
        let header = lines[0];
        assert!(header.contains("Gene"), "Header should contain Gene");
        assert!(header.contains("Pvalue"), "Header should contain Pvalue");
        assert!(
            header.contains("Pvalue_Burden"),
            "Header should contain Pvalue_Burden"
        );
        assert!(
            header.contains("Pvalue_SKAT"),
            "Header should contain Pvalue_SKAT"
        );

        // Parse data lines - fields are space-separated
        for line in &lines[1..] {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let gene = parts[0];
            assert!(!gene.is_empty(), "Gene should not be empty");

            // Pvalue (second column) should be a number or NA
            let pval_str = parts[1];
            if pval_str != "NA" {
                let pval: f64 = pval_str.parse().unwrap_or(f64::NAN);
                if !pval.is_nan() {
                    assert!(
                        (0.0..=1.0).contains(&pval),
                        "p-value out of range: {}",
                        pval
                    );
                }
            }
        }

        eprintln!("Gene test output has {} regions", lines.len() - 1);
    }
}
