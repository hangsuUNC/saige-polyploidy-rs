//! Property-based tests using proptest.
//!
//! These tests verify invariants that must hold for all valid inputs,
//! rather than checking specific numerical values. They complement
//! the unit tests and integration tests by exploring the input space
//! more broadly, catching edge cases in:
//!   - p-value bounds
//!   - Matrix symmetry and positive semi-definiteness
//!   - Numerical stability with degenerate inputs
//!   - Statistical invariants (allele order, etc.)

use proptest::prelude::*;

use saige_core::firth::logistic::{firth_logistic, FirthConfig};
use saige_core::glmm::link::TraitType;
use saige_core::glmm::pcg::OnTheFlyGrm;
use saige_core::grm::dense::compute_grm_from_dosages;
use saige_core::score_test::cct::cauchy_combination_test;
use saige_core::score_test::exact::exact_test_hypergeometric;
use saige_core::score_test::single_variant::ScoreTestEngine;
use saige_core::spa::binary::{k0_binom, k1_adj_binom, k2_binom, spa_binary};
use saige_linalg::dense::DenseMatrix;

// ---------------------------------------------------------------------------
// Strategy helpers (used implicitly by proptest macros below)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 1. Score test p-values must be in [0, 1]
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_score_test_pvalue_in_unit_interval(
        n in 10usize..30,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mu: Vec<f64> = (0..n).map(|_| 0.1 + rng.gen::<f64>() * 0.8).collect();
        let mu2: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
        let y: Vec<f64> = mu.iter().map(|&m| if rng.gen::<f64>() < m { 1.0 } else { 0.0 }).collect();
        let residuals: Vec<f64> = y.iter().zip(mu.iter()).map(|(yi, mi)| yi - mi).collect();

        // Intercept-only design matrix
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
            ploidy: saige_geno::traits::PloidyMode::Diploid,
            y: Some(y),
        };

        // Generate a polymorphic genotype vector
        let g: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let result = engine.test_marker(&g, "rs_test", "1", 100, "A", "T").unwrap();

        prop_assert!(result.pvalue >= 0.0, "p-value < 0: {}", result.pvalue);
        prop_assert!(result.pvalue <= 1.0, "p-value > 1: {}", result.pvalue);
        prop_assert!(result.pvalue_noadj >= 0.0);
        prop_assert!(result.pvalue_noadj <= 1.0);
        prop_assert!(result.var_t >= 0.0, "var_t < 0: {}", result.var_t);
        prop_assert!(result.var_t_star >= 0.0, "var_t_star < 0: {}", result.var_t_star);
        prop_assert!(result.af >= 0.0 && result.af <= 1.0);
    }
}

// ---------------------------------------------------------------------------
// 2. SPA p-values must be in [0, 1]
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_spa_pvalue_in_unit_interval(
        n in 10usize..50,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mu: Vec<f64> = (0..n).map(|_| 0.05 + rng.gen::<f64>() * 0.9).collect();
        let g: Vec<f64> = (0..n).map(|_| {
            let r: f64 = rng.gen();
            if r < 0.5 { 0.0 } else if r < 0.8 { 1.0 } else { 2.0 }
        }).collect();

        // Score: sum(g_i * (y_i - mu_i)) with y as Bernoulli(mu)
        let q: f64 = g.iter().zip(mu.iter())
            .map(|(&gi, &mi)| gi * (if rng.gen::<f64>() < mi { 1.0 } else { 0.0 } - mi))
            .sum();

        let m1: f64 = mu.iter().zip(g.iter()).map(|(m, gi)| m * gi).sum();
        let q_spa = q + m1;
        let qinv = 2.0 * m1 - q_spa;
        let result = spa_binary(&mu, &g, q_spa, qinv, 0.5, 1e-5);

        prop_assert!(result.pvalue >= 0.0, "SPA p-value < 0: {}", result.pvalue);
        prop_assert!(result.pvalue <= 1.0, "SPA p-value > 1: {}", result.pvalue);
    }
}

// ---------------------------------------------------------------------------
// 3. SPA CGF properties: K(0) = 0, K''(t) >= 0
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn prop_spa_cgf_properties(
        n in 5usize..20,
        seed in 0u64..500,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mu: Vec<f64> = (0..n).map(|_| 0.05 + rng.gen::<f64>() * 0.9).collect();
        let g: Vec<f64> = (0..n).map(|_| (rng.gen::<u32>() % 3) as f64).collect();

        // K(0) must be 0
        let k0 = k0_binom(0.0, &mu, &g);
        prop_assert!((k0).abs() < 1e-10, "K(0) should be 0, got {}", k0);

        // K'(0) = sum(mu_i * g_i) (verify via k1_adj with q=0)
        let k1_at_0 = k1_adj_binom(0.0, &mu, &g, 0.0);
        let expected_k1: f64 = mu.iter().zip(g.iter()).map(|(m, gi)| m * gi).sum();
        prop_assert!((k1_at_0 - expected_k1).abs() < 1e-10,
            "K'(0) mismatch: {} vs {}", k1_at_0, expected_k1);

        // K''(t) must be non-negative for all t
        for &t in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let k2 = k2_binom(t, &mu, &g);
            prop_assert!(k2 >= -1e-15, "K''({}) should be >= 0, got {}", t, k2);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. CCT p-values in [0, 1] and monotonicity
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_cct_pvalue_in_unit_interval(
        pvals in prop::collection::vec(0.001f64..0.999, 1..10),
    ) {
        let combined = cauchy_combination_test(&pvals, None);
        prop_assert!(combined >= 0.0, "CCT p < 0: {}", combined);
        prop_assert!(combined <= 1.0, "CCT p > 1: {}", combined);
    }

    #[test]
    fn prop_cct_monotonicity(
        base_pvals in prop::collection::vec(0.01f64..0.99, 2..6),
    ) {
        // Adding a very significant p-value should make the combined p smaller
        let p_combined_before = cauchy_combination_test(&base_pvals, None);

        let mut with_significant = base_pvals.clone();
        with_significant.push(0.0001);
        let p_combined_after = cauchy_combination_test(&with_significant, None);

        prop_assert!(p_combined_after <= p_combined_before + 1e-10,
            "Adding significant p-value should not increase combined: {} -> {}",
            p_combined_before, p_combined_after);
    }
}

// ---------------------------------------------------------------------------
// 5. Exact test p-values in [0, 1]
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_exact_test_pvalue_in_unit_interval(
        n_total in 10usize..50,
        case_frac in 2usize..8,  // fraction of cases: case_frac/10
        alt_frac in 1usize..5,   // fraction of ALT: alt_frac/10
    ) {
        let n_case = (n_total * case_frac / 10).max(1).min(n_total - 1);
        let n_alt = (n_total * alt_frac / 10).max(1).min(n_total - 1);

        // n_case_alt must be in valid range
        let n_control = n_total - n_case;
        let min_k = n_alt.saturating_sub(n_control);
        let max_k = n_case.min(n_alt);

        if min_k <= max_k {
            let n_case_alt = (min_k + max_k) / 2;
            let result = exact_test_hypergeometric(n_case_alt, n_case, n_alt, n_total);

            prop_assert!(result.pvalue >= 0.0, "exact p < 0: {}", result.pvalue);
            prop_assert!(result.pvalue <= 1.0 + 1e-10, "exact p > 1: {}", result.pvalue);
        }
    }
}

// ---------------------------------------------------------------------------
// 6. GRM symmetry
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_grm_symmetric(
        seed in 0u64..500,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let n = 8;
        let m = 20;
        let mut dosages = Vec::new();
        let mut afs = Vec::new();

        for _ in 0..m {
            let af = 0.1 + rng.gen::<f64>() * 0.4;
            let g: Vec<f64> = (0..n).map(|_| {
                let r: f64 = rng.gen();
                if r < (1.0 - af).powi(2) { 0.0 }
                else if r < (1.0 - af).powi(2) + 2.0 * af * (1.0 - af) { 1.0 }
                else { 2.0 }
            }).collect();
            dosages.push(g);
            afs.push(af);
        }

        let grm = compute_grm_from_dosages(&dosages, &afs);

        for i in 0..n {
            for j in 0..n {
                let diff = (grm.get(i, j) - grm.get(j, i)).abs();
                prop_assert!(diff < 1e-12,
                    "GRM not symmetric at ({},{}): {} vs {}", i, j, grm.get(i, j), grm.get(j, i));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. GRM positive semi-definiteness
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_grm_positive_semidefinite(
        seed in 0u64..300,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let n = 6;
        let m = 30;
        let mut dosages = Vec::new();
        let mut afs = Vec::new();

        for _ in 0..m {
            let af = 0.1 + rng.gen::<f64>() * 0.4;
            let g: Vec<f64> = (0..n).map(|_| {
                let r: f64 = rng.gen();
                if r < (1.0 - af).powi(2) { 0.0 }
                else if r < (1.0 - af).powi(2) + 2.0 * af * (1.0 - af) { 1.0 }
                else { 2.0 }
            }).collect();
            dosages.push(g);
            afs.push(af);
        }

        let grm = compute_grm_from_dosages(&dosages, &afs);

        // Check PSD by verifying v' * GRM * v >= 0 for random vectors
        for _ in 0..10 {
            let v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
            let grm_v = grm.mat_vec(&v);
            let vgv: f64 = v.iter().zip(grm_v.iter()).map(|(vi, gi)| vi * gi).sum();
            prop_assert!(vgv >= -1e-10,
                "GRM not PSD: v'*GRM*v = {} < 0", vgv);
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Monomorphic markers should not produce NaN/Inf
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_monomorphic_marker_no_nan(
        n in 10usize..30,
        dosage_val in prop::sample::select(vec![0.0f64, 1.0, 2.0]),
    ) {
        let mu: Vec<f64> = (0..n).map(|i| 0.2 + 0.6 * (i as f64 / n as f64)).collect();
        let mu2: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
        let residuals: Vec<f64> = mu.iter().map(|m| 0.5 - m).collect();

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
            ploidy: saige_geno::traits::PloidyMode::Diploid,
            y: None,
        };

        // All-same genotype (monomorphic)
        let g = vec![dosage_val; n];
        let result = engine.test_marker(&g, "rs_mono", "1", 100, "A", "T").unwrap();

        prop_assert!(!result.pvalue.is_nan(), "p-value is NaN for monomorphic marker");
        prop_assert!(!result.beta.is_nan() || result.var_t < 1e-20,
            "beta is NaN for monomorphic marker (var_t={})", result.var_t);
        prop_assert!(!result.tstat.is_nan(), "tstat is NaN for monomorphic marker");
    }
}

// ---------------------------------------------------------------------------
// 9. Allele order invariance: flipping g -> (2-g) should give same p-value
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_allele_flip_invariance(
        n in 15usize..30,
        seed in 0u64..500,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let mu: Vec<f64> = (0..n).map(|_| 0.1 + rng.gen::<f64>() * 0.8).collect();
        let mu2: Vec<f64> = mu.iter().map(|m| m * (1.0 - m)).collect();
        let residuals: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

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
            ploidy: saige_geno::traits::PloidyMode::Diploid,
            y: None,
        };

        // Generate a polymorphic genotype and its flipped version
        let g: Vec<f64> = (0..n).map(|_| {
            let r: f64 = rng.gen();
            if r < 0.5 { 0.0 } else if r < 0.8 { 1.0 } else { 2.0 }
        }).collect();
        let g_flip: Vec<f64> = g.iter().map(|gi| 2.0 - gi).collect();

        let result_orig = engine.test_marker(&g, "rs1", "1", 100, "A", "T").unwrap();
        let result_flip = engine.test_marker(&g_flip, "rs1", "1", 100, "T", "A").unwrap();

        // p-values should be identical (score^2/var is invariant to sign of g_tilde)
        let p_diff = (result_orig.pvalue - result_flip.pvalue).abs();
        prop_assert!(p_diff < 1e-10,
            "p-values differ after allele flip: {} vs {} (diff={})",
            result_orig.pvalue, result_flip.pvalue, p_diff);

        // |beta| should be the same (sign flips)
        let beta_diff = (result_orig.beta.abs() - result_flip.beta.abs()).abs();
        prop_assert!(beta_diff < 1e-10,
            "|beta| differs after allele flip: {} vs {}",
            result_orig.beta, result_flip.beta);
    }
}

// ---------------------------------------------------------------------------
// 10. Firth convergence for non-degenerate data
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn prop_firth_convergence(
        seed in 0u64..500,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let n = 50;
        // Generate balanced binary outcome
        let y: Vec<f64> = (0..n).map(|_| if rng.gen::<f64>() < 0.5 { 1.0 } else { 0.0 }).collect();

        // Intercept + one covariate
        let mut x_data = vec![0.0; n * 2];
        for i in 0..n {
            x_data[i] = 1.0; // intercept
            x_data[n + i] = rng.gen::<f64>() * 2.0 - 1.0; // covariate in [-1, 1]
        }
        let x = DenseMatrix::from_col_major(n, 2, x_data);

        let config = FirthConfig {
            max_iter: 50,
            tol: 1e-5,
            l2_penalty: 0.0,
        };

        let result = firth_logistic(&y, &x, &config).unwrap();

        prop_assert!(result.converged, "Firth did not converge (seed={})", seed);
        prop_assert!(result.beta[0].is_finite(), "intercept not finite");
        prop_assert!(result.beta[1].is_finite(), "slope not finite");
        prop_assert!(result.pvalue >= 0.0 && result.pvalue <= 1.0,
            "Firth p-value out of range: {}", result.pvalue);
    }
}

// ---------------------------------------------------------------------------
// 11. OnTheFlyGrm mat_vec consistency with explicit GRM
// ---------------------------------------------------------------------------
proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn prop_on_the_fly_grm_matches_explicit(
        seed in 0u64..300,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        let n = 8;
        let m = 15;
        let mut dosages = Vec::new();
        let mut afs = Vec::new();

        for _ in 0..m {
            let af = 0.1 + rng.gen::<f64>() * 0.4;
            let g: Vec<f64> = (0..n).map(|_| {
                let r: f64 = rng.gen();
                if r < (1.0 - af).powi(2) { 0.0 }
                else if r < (1.0 - af).powi(2) + 2.0 * af * (1.0 - af) { 1.0 }
                else { 2.0 }
            }).collect();
            dosages.push(g);
            afs.push(af);
        }

        // Compute GRM * v both ways
        let grm_explicit = compute_grm_from_dosages(&dosages, &afs);
        let grm_otf = OnTheFlyGrm::new(&dosages, &afs);

        let v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();

        let result_explicit = grm_explicit.mat_vec(&v);
        let result_otf = grm_otf.mat_vec(&v);

        for i in 0..n {
            let diff = (result_explicit[i] - result_otf[i]).abs();
            // OnTheFlyGrm uses f32 internally (matching R SAIGE's float32 precision),
            // so tolerance must account for f32 rounding (~1e-6 relative).
            prop_assert!(diff < 1e-5,
                "GRM*v mismatch at [{}]: explicit={}, on-the-fly={} (diff={})",
                i, result_explicit[i], result_otf[i], diff);
        }
    }
}
