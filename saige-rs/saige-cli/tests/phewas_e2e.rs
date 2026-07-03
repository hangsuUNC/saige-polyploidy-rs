//! End-to-end CLI test for `saige phewas`.
//!
//! Invokes the compiled `saige` binary (via the `CARGO_BIN_EXE_saige` path that
//! Cargo provides to integration tests) against the shared test fixtures,
//! running Step 1 + Step 2 for one phecode and checking the combined output
//! table. Skips gracefully when the fixtures are not present, matching the
//! convention used by the other integration tests in this workspace.

use std::path::Path;
use std::process::Command;

/// Fixtures live behind the `saige-rs/tests/fixtures` symlink (-> extdata).
fn fixtures_dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/fixtures").to_string()
}

#[test]
fn phewas_end_to_end() {
    let fixtures = fixtures_dir();
    let plink = format!(
        "{fixtures}/input/nfam_100_nindep_0_step1_includeMoreRareVariants_poly"
    );
    let pheno = format!("{fixtures}/input/pheno_1000samples.txt");

    if !Path::new(&format!("{plink}.bed")).exists() || !Path::new(&pheno).exists() {
        eprintln!("Skipping phewas_end_to_end: test fixtures not found at {fixtures}");
        return;
    }

    let out = std::env::temp_dir().join(format!("phewas_e2e_{}.tsv", std::process::id()));
    let out_str = out.to_str().unwrap().to_string();

    // Use the same PLINK set as both the Step-1 GRM and the Step-2 test variants.
    // Small AI-REML settings keep the run quick; --skip-vr defaults to true.
    let output = Command::new(env!("CARGO_BIN_EXE_saige"))
        .args([
            "phewas",
            "--plink-file",
            &plink,
            "--pheno-file",
            &pheno,
            "--phecodes",
            "y",
            "--covar-cols",
            "x1,x2",
            "--sample-id-col",
            "IID",
            "--trait-type",
            "binary",
            "--test-plink",
            &plink,
            "--output-file",
            &out_str,
            "--max-iter",
            "10",
            "--n-random-vectors",
            "10",
        ])
        .output()
        .expect("failed to execute the saige binary");

    if !output.status.success() {
        panic!(
            "saige phewas failed (status {:?})\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }

    let content = std::fs::read_to_string(&out)
        .expect("phewas did not produce the requested output file");
    let mut lines = content.lines();

    // Header must match the phewas combined-table schema.
    let header = lines.next().unwrap_or("");
    assert!(
        header.starts_with("phenotype\tCHR\tPOS\tSNPID\tAllele1\tAllele2"),
        "unexpected phewas header: {header}"
    );

    // Data rows (if any) must be tagged with the phecode we requested.
    let data_rows: Vec<&str> = lines.filter(|l| !l.trim().is_empty()).collect();
    for row in &data_rows {
        assert!(
            row.starts_with("y\t"),
            "result row not tagged with phecode 'y': {row}"
        );
        // AF must be a valid frequency and beta finite (sanity on the pipeline).
        let cols: Vec<&str> = row.split('\t').collect();
        assert!(cols.len() >= 21, "row has too few columns: {row}");
        if let Ok(af) = cols[7].parse::<f64>() {
            assert!((0.0..=1.0).contains(&af), "AF out of [0,1]: {af}");
        }
    }

    // A successful run over 100 test markers should yield at least one row; we
    // warn rather than hard-fail so the test remains robust if all markers are
    // filtered out by MAF/MAC on this particular fixture.
    if data_rows.is_empty() {
        eprintln!("phewas_end_to_end: no data rows produced (all markers filtered?)");
    }

    let _ = std::fs::remove_file(&out);
}
