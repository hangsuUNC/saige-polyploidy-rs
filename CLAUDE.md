# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of [saigegit/SAIGE](https://github.com/saigegit/SAIGE) (the R/Rcpp GWAS package, v1.5.1)
that **adds a from-scratch Rust rewrite in `saige-rs/`**. There are two parallel codebases here:

1. **Upstream R/C++ SAIGE** — `R/` (R sources), `src/` (Rcpp/RcppArmadillo C++), `extdata/`
   (the canonical step-1/2/3 driver scripts), `man/`, `DESCRIPTION`, built as an R package.
   Treat this as the **reference implementation** — it defines the behavior `saige-rs` must match.
2. **`saige-rs/`** — the actively developed Rust port (a Cargo workspace). Recent git history is
   almost entirely `saige-rs` work. **This is where new development happens.**

When a task says "SAIGE" without qualification, ask which side; default to `saige-rs/` given the
active work, but read the matching R/C++ source to confirm intended behavior before changing Rust.

## saige-rs: the Rust workspace

`cd saige-rs` first — it is a self-contained Cargo workspace with its own `Cargo.toml`/lock,
separate from the R package.

```bash
cd saige-rs
cargo build --release            # binary: target/release/saige
cargo test --workspace           # unit + integration + property tests
cargo test -p saige-core <name>  # single test in one crate
cargo clippy --all-targets --all-features   # CI runs this with -D warnings
cargo fmt --all                  # CI checks `cargo fmt --all -- --check`
```

CI (`.github/workflows/saige-rs.yml`, triggered only on `saige-rs/**` changes) gates on **fmt,
clippy, tests on ubuntu+macos (debug and release), and a release build**, all with
`RUSTFLAGS=-D warnings`. Keep clippy clean — warnings fail the build.

### Crate layout (dependency order: linalg → geno → core → cli)

| Crate | Role |
|-------|------|
| `saige-linalg` | Dense/sparse matrix wrappers (`faer`, `sprs`), decompositions, the **PCG solver**. Foundation, no SAIGE logic. |
| `saige-geno` | Genotype/phenotype I/O. The `GenotypeReader` trait + readers for PLINK bed/bim/fam, BGEN v1.2, VCF/BCF (`noodles`), SAV, PGEN; plus `phenotype`, `sample`, `group_file`, `sparse_grm_io`. |
| `saige-core` | All statistics: `glmm` (AI-REML, PCG, variance ratio, LOCO, family/link), `grm` (dense/sparse), `spa` (binary/fast/survival saddlepoint), `score_test` (single_variant, region, CCT, exact, permutation), `firth`, `ld`, `model` (NullModel + serialization). |
| `saige-cli` | The `saige` binary. Thin clap wrapper; subcommands live in `src/commands/`. |

### CLI subcommands ↔ SAIGE three-step workflow ↔ R driver scripts

The Rust subcommands mirror the standard SAIGE pipeline and the R scripts in `extdata/`:

| `saige` subcommand | SAIGE step | R reference script |
|--------------------|-----------|--------------------|
| `fit-null`          | Step 1: fit null GLMM, estimate variance ratios | `extdata/step1_fitNULLGLMM.R` |
| `test`              | Step 2: single-variant / region association (SPA) | `extdata/step2_SPAtests.R` |
| `ld-matrix`         | Step 3: LD matrix for region tests | `extdata/step3_LDmat.R` |
| `create-sparse-grm` | Sparse GRM construction | `extdata/createSparseGRM.R` |
| `get-neff`          | Effective sample size | `extdata/extractNglmm.R` |

The fitted null model is serialized with `bincode` to `<prefix>.saige.model` (magic bytes `SGMD`),
with an optional JSON sidecar for inspection. `saige-core::model::serialization` is the entry point.

### Dispatch convention

`GenotypeReader` is used via **static dispatch (generics) in hot loops** and **dynamic dispatch
(`Box<dyn GenotypeReader>`) at the CLI level** — preserve this when adding readers or test paths.

## Matching R SAIGE (the central constraint)

`saige-rs` is validated against R SAIGE for numerical concordance (current: 0.995 correlation on
−log₁₀ p across ~2K variants). Most recent commits are bug-fixes that align Rust behavior with
subtle R/C++ details — e.g. **separate GRM vs variance-ratio marker pools**, AI-REML `tau`
initialization, SPA variance direction, P-projection. When changing core statistics:

- Read the corresponding `R/`/`src/` code first; the C++ in `src/` (e.g. `SAIGE_test.cpp`,
  `SPA*.cpp`, `SAIGE_fitGLMM_fast.cpp`) is the ground truth for formulas.
- **Validate against R, not `is_fastTest=TRUE`.** R SAIGE's default fast test uses an approximate
  variance formula that inflates variance for high-AF variants; benchmark/compare with
  `--is_fastTest=FALSE`.
- Test fixtures live in `extdata/input/` (small PLINK/BGEN/VCF marker sets); `saige-core` and
  `saige-geno` integration tests skip gracefully when fixtures are absent.

## Building / running the upstream R package

Dependencies are managed with **pixi** (conda-forge + bioconda) — see `pixi.toml` (R, Rcpp,
RcppArmadillo, RcppEigen, SKAT, MetaSKAT, savvy, openblas, etc.). The C++ is compiled via
`src/Makevars`. A Docker build is also provided:

```bash
make docker          # docker build -f docker/Dockerfile .
```

Do not edit `src/RcppExports.cpp` or `R/RcppExports.R` by hand — they are generated by Rcpp.
