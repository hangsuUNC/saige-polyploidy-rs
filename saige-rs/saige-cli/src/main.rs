//! saige-rs: A Rust implementation of SAIGE for GWAS.
//!
//! CLI entry point using clap for argument parsing.

mod commands;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "saige",
    version,
    about = "SAIGE-RS: Scalable and Accurate Implementation of GEneralized mixed model",
    long_about = "A Rust implementation of SAIGE for genome-wide association studies.\n\
                   Supports binary, quantitative, and survival traits with mixed models."
)]
struct Cli {
    /// Number of threads to use
    #[arg(long, default_value = "1", global = true)]
    threads: usize,

    /// Verbosity level (can be repeated: -v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Step 1: Fit the null GLMM and estimate variance ratios
    FitNull(commands::fit_null::FitNullArgs),

    /// Step 2: Run single-variant or region-based association tests
    Test(commands::assoc_test::AssocTestArgs),

    /// PheWAS: test a variant set against many phenotypes (Step 1 + Step 2)
    Phewas(commands::phewas::PheWasArgs),

    /// Step 3: Compute LD matrix for region-based tests
    LdMatrix(commands::ld_matrix::LdMatrixArgs),

    /// Create a sparse GRM from genotype data
    CreateSparseGrm(commands::create_sparse_grm::CreateSparseGrmArgs),

    /// Compute effective sample size from a fitted model
    GetNeff(commands::get_neff::GetNeffArgs),
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Set up logging
    let filter = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();

    // Set up thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(cli.threads)
        .build_global()
        .ok();

    tracing::info!("SAIGE-RS v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Using {} threads", cli.threads);

    match cli.command {
        Commands::FitNull(args) => commands::fit_null::run(args),
        Commands::Test(args) => commands::assoc_test::run(args),
        Commands::Phewas(args) => commands::phewas::run(args),
        Commands::LdMatrix(args) => commands::ld_matrix::run(args),
        Commands::CreateSparseGrm(args) => commands::create_sparse_grm::run(args),
        Commands::GetNeff(args) => commands::get_neff::run(args),
    }
}
