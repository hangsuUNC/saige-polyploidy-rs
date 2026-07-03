# SAIGE-CNV: Diagnosis of Inflation, Sign-Flipped BETA, and Extreme BETA

**Date:** 2026-07-02
**Context:** Applying `saige-polyploidy-rs` to CNV / tandem-repeat / mtDNA copy-number dosages
(dosage range e.g. `[0, 8]`, most samples at 2). Symptoms observed in the R SAIGE runs and
reproduced/traceable in the Rust port during work with Linfeng Hu & Wei.

Reference failing marker (ESRRA, R SAIGE):

```
BETA=-6.82478  SE=0.438323  Tstat=14.4404  var=1.37846e-13  p=3.4e-109
AC_Allele2=38081.4  AF_Allele2=2.04673  N_case=220  N_ctrl=9083
```

---

## TL;DR

The three symptoms are **not primarily caused by Firth or SPA**. Both are essentially
scale-agnostic and largely correct. The root causes sit **upstream** of them:

1. **Opposite BETA** → the diploid `2 − g` allele flip triggered because `AF = sum/(2N) > 0.5`.
2. **Extreme BETA (1e13)** → the score-test formula `beta = S / var`, whose variance collapses
   when the dosage has tiny dispersion (near-constant predictor).
3. **Inflation** → variance-ratio miscalibration: a negative/garbage MAC selects the wrong
   categorical variance ratio, and the VR is estimated on a hard-coded `[0, 2]` scale.

Firth and SPA mostly *inherit* or *mask* the damage.

---

## 1. Opposite direction of BETA — the `2 − g` flip

- Diploid allele flip is baked into variance-ratio estimation:
  `saige-core/src/glmm/variance_ratio.rs:133-134`

  ```rust
  let g0: Vec<f64> = if af > 0.5 {
      g_raw.iter().map(|&gi| 2.0 - gi).collect()
  ...
  ```

  (Header comment `variance_ratio.rs:11` states the rule explicitly.)
- For a CNV, `af = sum / (2N)` is **guaranteed > 0.5** (observed `AF_Allele2 = 2.047`), so every
  such marker is sign-flipped → negative beta.
- This is the exact "implicit diploid assumption" that Wei/Linfeng found in R SAIGE.
- **Firth does not cause the sign flip.** `firth/logistic.rs:227-238` fits
  `y ~ [covariates | genotype]` and reports the true coefficient sign. The `-6.8` Firth beta in R
  was negative only because the genotype was *already flipped* before Firth saw it. The Rust
  single-variant path has no flip, so its score-beta sign is correct — meaning R and this port can
  disagree on sign.

## 2. Extreme BETA (1e13) when SD is small — the `S / var` formula

- `saige-core/src/score_test/single_variant.rs:211` → `beta = score / var_t_star`, with
  `var_t_star = (Σ g̃ᵢ² · μ(1−μ)) · τₑ · vr` (lines 160-169).
- When the dosage is nearly constant (std 0.377, most samples exactly 2), `g̃ ≈ 0` after projecting
  out the intercept, so the variance **collapses** (`var = 1.38e-13`) and `beta = S/var` explodes.
  This is inherent ill-conditioning of the score-test beta for a low-dispersion predictor — not a
  Firth/SPA bug.
- Firth normally *hides* this by estimating the coefficient as a regularized MLE. Two catches in the
  Rust Firth:
  - Genotype is fed **unstandardized** (`firth/logistic.rs:234`); a near-constant column makes
    `X'WX` near-singular in the genotype term, triggering the ridge fallback
    (`firth/logistic.rs:118-123`) → unstable estimate.
  - If Firth doesn't converge in 25 iterations, `assoc_test.rs:204-209` only overrides beta when
    `fr.converged` is true — otherwise it **keeps the exploded `S/var` beta**.

## 3. Inflation — miscalibrated variance ratio

- `saige-geno/src/traits.rs:52` → `mac = sum.min(2N − sum)` goes **negative** when mean dosage > 2
  (AC=38081 > 2N=18606). That garbage MAC then:
  - drives `select_variance_ratio(mac, …)` (`variance_ratio.rs:289`) into the **wrong categorical
    bin** → wrong `vr` → miscalibrated statistic (Linfeng's "variance ratio might just be off");
  - causes markers to be silently dropped by filters at `assoc_test.rs:169`
    (`mac < min_mac`) and `assoc_test.rs:173` (`af > 1 − min_maf`).
- The variance ratio itself is estimated on a `[0, 2]` scale (plus the flip in §1), so it does not
  calibrate a dosage-in-`[0,8]` marker.

## What is actually fine

- **SPA CGF** (`spa/binary.rs:15`): `K(t) = Σ log(1 − μ + μ·e^{g·t})` models the *phenotype*
  randomness (`y ~ Bernoulli(μ)`) with `g` a fixed weight. It does **not** assume diploid and is
  valid for any dosage scale. The earlier note claiming "SPA needs Binomial(CNᵢ, p)" is a
  misconception — SPA conditions on the genotype.
- **Firth's algorithm** is correct and scale-agnostic; its only issues are the unstandardized
  predictor and the non-convergence fallback above.

---

## Diploid-assumption inventory (code locations)

| # | Location | Issue | Symptom |
|---|----------|-------|---------|
| 1 | `saige-geno/src/traits.rs:51` | `af = sum / (2N)` → AF > 1 for CNV | filtering, flip trigger |
| 2 | `saige-geno/src/traits.rs:52` | `mac = sum.min(2N − sum)` → negative | wrong VR bin, dropped markers |
| 3 | `saige-geno/src/traits.rs:58` | impute missing with `2·af` | biased imputation |
| 4 | `saige-cli/src/commands/assoc_test.rs:169,173` | MAF/MAC filters assume `[0,2]` | CNV markers silently skipped |
| 5 | `saige-core/src/glmm/variance_ratio.rs:133-134` | `g0 = 2 − g` flip if `af>0.5` | opposite BETA |
| 6 | `saige-core/src/glmm/variance_ratio.rs:289` | `select_variance_ratio` bins by MAC | inflation |
| 7 | `saige-core/src/score_test/single_variant.rs:211` | `beta = S / var` blows up at low dispersion | extreme BETA |
| 8 | `saige-core/src/score_test/single_variant.rs:195` | SPA `q = score/√vr + m1` inherits bad `vr` | miscalibration |
| 9 | `saige-cli/src/commands/assoc_test.rs:204-209` | keeps exploded beta if Firth not converged | extreme BETA |

---

## Implementation status (branch `cnv-polyploidy-fixes`)

Design: ploidy is now an **explicit mode** (`PloidyMode` in `saige-geno/src/traits.rs`), selectable
per run via `--ploidy`:

| `--ploidy` | Denominator | Use case |
|-----------|-------------|----------|
| `diploid` (default) | 2 | normal SNP GWAS — byte-for-byte unchanged |
| `haploid` | 1 | **mitochondrial** homoplasmic (0/1) & heteroplasmic (VAF) variants, chrY |
| `auto` | per-marker `max(2, max dosage)` | CNV / tandem-repeat dosages |
| `<number>` | fixed N | known fixed copy number |

This replaced the earlier data-inferred `max(2.0, max dosage)` floor, which was wrong for haploid
mtDNA (it would halve the allele frequency). All AF/MAC/flip logic routes through
`PloidyMode::resolve(dosages)`.

Done in this branch:

- **`PloidyMode` enum** with `resolve()` and `parse()` — `saige-geno/src/traits.rs`.
- **AF / MAC** normalized by the resolved ploidy — `saige-geno/src/traits.rs` (`compute_af`, new
  `ploidy` field) threaded through all readers (`plink.rs`, `vcf.rs`, `bgen.rs`, plus `set_ploidy`
  on the `GenotypeReader` trait and the `sav`/`pgen` stubs). AF ∈ [0,1], MAC ≥ 0.
- **CLI `--ploidy`** on `test` (`assoc_test.rs`) and `fit-null` (`fit_null.rs`); wired to the
  reader, the `ScoreTestEngine`, and `VarianceRatioConfig`.
- **Score-test path** recomputes AF/MAC with inferred ploidy and normalizes case/control AF —
  `saige-core/src/score_test/single_variant.rs`. This also feeds a sane MAC into variance-ratio
  bin selection.
- **VR flip** generalized `2 - g` → `ploidy - g` via `VarianceRatioConfig.ploidy` —
  `saige-core/src/glmm/variance_ratio.rs`.
- **Score engine** resolves AF/MAC/case-control AF via `ScoreTestEngine.ploidy` —
  `saige-core/src/score_test/single_variant.rs`.
- **Firth robustness** — `saige-core/src/firth/logistic.rs`: genotype column standardized before
  fitting (back-transformed after), and SE/p-value now computed at the final iterate so a finite
  Firth beta is returned even without full convergence.
- **Firth reporting** — `saige-cli/src/commands/assoc_test.rs`: adopt the (finite) Firth beta/SE
  even from a non-converged fit and **keep the score/SPA p-value** (matching `is_Firth_beta`
  semantics), instead of silently retaining the exploded `S/var` beta.
- **Tests** added for diploid-unchanged, rare-SNP floor, CNV AF∈[0,1]/MAC≥0, CNV imputation, and a
  CNV score-test producing a finite beta.

### Build / verify locally (not run in this environment)

```bash
cd saige-rs
cargo fmt --all
cargo clippy --all-targets --all-features   # must be warning-clean (CI uses -D warnings)
cargo test --workspace
# targeted:
cargo test -p saige-geno traits::tests
cargo test -p saige-core single_variant::tests::test_cnv_dosage_valid_af_and_finite_beta
# end-to-end PheWAS CLI (runs the built `saige phewas` on fixtures; skips if absent):
cargo test -p saige-cli --test phewas_e2e
```

### Usage examples

Mitochondrial (haploid) single-variant PheWAS — Step 2 per phecode:

```bash
# Step 1 (nuclear GRM is diploid → leave --ploidy at default)
saige fit-null --plink-file nuclear_grm --pheno-file pheno.tsv --pheno-col <phecode> \
  --trait-type binary --sparse-grm sparseGRM.mtx ...

# Step 2: mtDNA variants are haploid (homoplasmic 0/1 or heteroplasmic VAF)
saige test --vcf-file mtdna.vcf.gz --model-file <phecode>.saige.model \
  --ploidy haploid --is-firth true ...
```

CNV / tandem-repeat dosages (0..N): use `--ploidy auto` (or a fixed number) in Step 2.

### SAIGE-PheWAS (`saige phewas`) — implemented

A `phewas` subcommand runs Step 1 + Step 2 for many phenotypes in one process,
**parallelized across phecodes** (rayon), so the per-phenotype null fit — the dominant cost — scales
with cores. `--skip-vr` (default **on** for PheWAS) skips variance-ratio estimation per Wei's
suggestion and sidesteps VR-miscalibration inflation. Output is a single phecode × variant table.

Shared refactor: the Step-1 core (`fit_null_model`) and score-engine builder (`build_engine`) now
live in `saige-cli/src/commands/pipeline.rs` and are reused by `fit-null`, `test`, and `phewas`.

Example — mitochondrial (haploid) single-variant PheWAS across phecodes:

```bash
saige phewas \
  --plink-file nuclear_grm \                 # Step 1 GRM (diploid nuclear SNPs)
  --pheno-file base_pheno.tsv \
  --phecodes P1,P2,P3            (or --phecode-file phecodes.txt) \
  --covar-cols age,age2,sex,PC1,PC2,mPC1,mPC2 \
  --test-vcf mtdna.vcf.gz --test-ploidy haploid \   # haploid mtDNA variants
  --is-firth true \
  --output-file phewas_results.tsv
```

For CNV/TR variants use `--test-ploidy auto`; `--test-plink` / `--test-bgen` are also accepted.

Note: per-phenotype valid-sample sets differ (phenotype/covariate missingness), so each phecode fits
its own GRM subset — matching running `fit-null` separately per phenotype, just faster and in one
process. Not yet done: sharing a single in-memory genotype matrix across phecodes (memory-bound),
and a sparse-GRM fast path for `phewas`.

Still open (not yet done): P2 variance-ratio recalibration for the dosage scale and the
skip-VR / PheWAS workflow; P3 per-sample missing imputation with a real CN source, BGEN per-sample
ploidy, and R-SAIGE numerical validation on a CNV marker set.

## TO-DO

### P0 — Correctness (unblock CNV results) — DONE on branch `cnv-polyploidy-fixes`

- [x] **Introduce a per-marker ploidy / max-dosage model** — inferred as `max(2.0, max dosage)`.
      (`traits.rs:infer_ploidy`, `MarkerData.ploidy`)
- [x] **Fix AF:** `af = sum / (ploidy·N)`, AF ∈ [0,1]. (`traits.rs:compute_af`,
      `single_variant.rs`)
- [x] **Fix MAC so it cannot go negative:** `mac = min(AC, ploidy·N − AC)`. (`traits.rs`,
      `single_variant.rs`)
- [x] **Generalize the `af > 0.5` flip** to `ploidy − g` (diploid unchanged since ploidy = 2).
      (`variance_ratio.rs`)
- [x] **Re-examine MAF/MAC filters** — now correct once AF ∈ [0,1] and MAC ≥ 0.
      (`assoc_test.rs`)

### P1 — Beta stability & reporting — DONE on branch `cnv-polyploidy-fixes`

- [x] **Report Firth beta even on non-convergence** (finite check + warn), keep score/SPA p-value.
      (`assoc_test.rs`)
- [x] **Standardize the genotype column before Firth**, back-transform beta/SE.
      (`firth/logistic.rs:firth_test_variant`)
- [x] **SE/p-value computed at final iterate** via shared `firth_se_pvalue` so non-converged fits
      still yield finite SE. (`firth/logistic.rs`)
- [ ] (Optional) Guard the score-test beta against `var → 0` with an explicit `NA`/warning; left
      as-is for now since the Firth path supplies the sane beta. (`single_variant.rs`)
- [ ] (Optional) Consider the **penalized** information matrix for Firth SE. (`firth/logistic.rs`)

### P2 — Calibration

- [ ] **Recalibrate variance ratio for the actual dosage scale** (not `[0,2]`); confirm categorical
      VR bins are chosen with the corrected MAC. (`variance_ratio.rs`)
- [ ] Add option to **skip variance-ratio estimation** for small numbers of test variants
      (per Wei's suggestion) — useful for PheWAS-style single-marker CNV tests.
- [ ] **Force the exact test — always `is_fastTest = FALSE`; then remove the fast path.**
      R SAIGE must be run with `--is_fastTest=FALSE` for accurate variance estimation; the default
      `--is_fastTest=TRUE` uses an approximate variance formula that can produce **inflated variance
      for high-AF variants**. The Rust analog is `--is-fast-spa` (default `true`, `assoc_test.rs:56`),
      which routes to `saige-core/src/spa/fast.rs` — that path approximates the zero-genotype block's
      CGF with a normal (`NAmu`/`NAsigma`), the same approximation source.
      Sub-tasks:
    - [ ] **Investigate/test why** the fast approximation inflates variance for high-AF variants
          (few zero-dosage samples → the normal approximation of the zeroed block is poor; quantify
          against the exact CGF on high-AF and CNV/haploid markers).
    - [ ] Change the default to `false` and confirm concordance with the exact path.
    - [ ] **Remove `is_fast_spa` / the fast-SPA code path entirely** (always exact): drop the flag in
          `assoc_test.rs`, the `use_fast_spa` field in `ScoreTestEngine`, the `spa_binary_fast`
          dispatch in `single_variant.rs`, and retire `saige-core/src/spa/fast.rs`.

### P3 — Imputation / readers / validation

- [ ] **Per-sample missing imputation:** fill with `CNᵢ · p̂` instead of `2·af`. (`traits.rs:58`)
- [ ] **BGEN per-sample ploidy:** relax the `min_ploidy == max_ploidy == 2` guard and read the
      per-sample ploidy byte. (`saige-geno/src/bgen.rs`)
- [ ] **Add CNV test fixtures** (dosage > 2) and a regression test asserting: correct BETA sign,
      finite BETA magnitude, and AF ∈ [0,1].
- [ ] **Validate against R SAIGE** with `--is_fastTest=FALSE` on a matched CNV marker set.

### Verification

- [ ] After each P0/P1 fix, re-run the ESRRA marker and confirm: BETA sign matches raw association,
      `|BETA|` is finite/interpretable, `AF ∈ [0,1]`, MAC ≥ 0, and QQ/λ shows no residual inflation.

---

## Sources

- Slack DM with Linfeng — ESRRA output & diploid-flip discussion.
- Slack DM — turning off SPA/Firth: direction fixed but BETA still ~1e13; small-SD explanation.
