//! TSV phenotype and covariate file parser.
//!
//! Reads tab/space-delimited files with sample IDs and phenotype/covariate columns.
//! Supports missing value handling and sample ID matching.

use std::path::Path;

use anyhow::{bail, Context, Result};

/// Parsed phenotype data for all samples.
#[derive(Debug, Clone)]
pub struct PhenotypeData {
    /// Sample IDs in file order.
    pub sample_ids: Vec<String>,
    /// Phenotype values (NaN for missing).
    pub phenotype: Vec<f64>,
    /// Covariate matrix: covariates[i][j] = sample i, covariate j.
    pub covariates: Vec<Vec<f64>>,
    /// Covariate column names.
    pub covariate_names: Vec<String>,
}

/// Parse a phenotype/covariate file.
///
/// # Arguments
/// - `path`: Path to the TSV/CSV file
/// - `pheno_col`: Name of the phenotype column
/// - `covar_cols`: Names of covariate columns (may be empty)
/// - `sample_id_col`: Name of the sample ID column (default: "IID")
/// - `delimiter`: Column delimiter (default: auto-detect tab or space)
pub fn parse_phenotype_file(
    path: &Path,
    pheno_col: &str,
    covar_cols: &[String],
    sample_id_col: &str,
) -> Result<PhenotypeData> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read phenotype file: {}", path.display()))?;

    let mut lines = contents.lines();
    let header_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("Empty phenotype file"))?;

    // Detect delimiter
    let delim = if header_line.contains('\t') {
        '\t'
    } else {
        ' '
    };

    let headers: Vec<&str> = header_line.split(delim).map(|s| s.trim()).collect();

    // Find column indices
    let id_idx = headers
        .iter()
        .position(|&h| h == sample_id_col)
        .ok_or_else(|| {
            anyhow::anyhow!("Sample ID column '{}' not found in header", sample_id_col)
        })?;

    let pheno_idx = headers
        .iter()
        .position(|&h| h == pheno_col)
        .ok_or_else(|| anyhow::anyhow!("Phenotype column '{}' not found in header", pheno_col))?;

    let covar_indices: Vec<usize> = covar_cols
        .iter()
        .map(|name| {
            headers
                .iter()
                .position(|&h| h == name.as_str())
                .ok_or_else(|| anyhow::anyhow!("Covariate column '{}' not found in header", name))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut sample_ids = Vec::new();
    let mut phenotype = Vec::new();
    let mut covariates = Vec::new();

    for (line_num, line) in lines.enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(delim).map(|s| s.trim()).collect();
        if fields.len() <= id_idx.max(pheno_idx) {
            bail!(
                "Line {} has too few fields (expected at least {})",
                line_num + 2,
                id_idx.max(pheno_idx) + 1
            );
        }

        sample_ids.push(fields[id_idx].to_string());

        let pheno_val = parse_value(fields[pheno_idx]);
        phenotype.push(pheno_val);

        let mut row_covars = Vec::with_capacity(covar_indices.len());
        for &ci in &covar_indices {
            if ci < fields.len() {
                row_covars.push(parse_value(fields[ci]));
            } else {
                row_covars.push(f64::NAN);
            }
        }
        covariates.push(row_covars);
    }

    Ok(PhenotypeData {
        sample_ids,
        phenotype,
        covariates,
        covariate_names: covar_cols.to_vec(),
    })
}

/// Parse a string value to f64.
///
/// Missing tokens map to NaN. Boolean labels (as written by pandas/R, e.g.
/// `True`/`False`) map to 1.0/0.0 so a phenotype or covariate coded as a logical
/// is usable directly without pre-recoding.
fn parse_value(s: &str) -> f64 {
    match s {
        "NA" | "na" | "Na" | "." | "" | "-" | "NaN" | "nan" => f64::NAN,
        "True" | "TRUE" | "true" | "T" => 1.0,
        "False" | "FALSE" | "false" | "F" => 0.0,
        _ => s.parse().unwrap_or(f64::NAN),
    }
}

/// Build a design matrix (intercept + covariates) for the given samples.
/// Returns (X matrix as flat col-major vec, n_samples, n_covariates+1).
pub fn build_design_matrix(pheno_data: &PhenotypeData) -> (Vec<f64>, usize, usize) {
    let n = pheno_data.sample_ids.len();
    let p = pheno_data.covariate_names.len() + 1; // +1 for intercept
    let mut x = vec![0.0; n * p];

    // Column 0: intercept
    for xi in x.iter_mut().take(n) {
        *xi = 1.0;
    }

    // Remaining columns: covariates
    for (j, _) in pheno_data.covariate_names.iter().enumerate() {
        for i in 0..n {
            let val = if j < pheno_data.covariates[i].len() {
                pheno_data.covariates[i][j]
            } else {
                0.0
            };
            x[(j + 1) * n + i] = val;
        }
    }

    (x, n, p)
}

/// Filter samples to only those with non-missing phenotype and covariates.
/// Returns indices of valid samples.
pub fn valid_sample_indices(pheno_data: &PhenotypeData) -> Vec<usize> {
    let mut valid = Vec::new();
    for i in 0..pheno_data.sample_ids.len() {
        if pheno_data.phenotype[i].is_nan() {
            continue;
        }
        let mut all_covars_valid = true;
        for j in 0..pheno_data.covariate_names.len() {
            if j < pheno_data.covariates[i].len() && pheno_data.covariates[i][j].is_nan() {
                all_covars_valid = false;
                break;
            }
        }
        if all_covars_valid {
            valid.push(i);
        }
    }
    valid
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_value() {
        assert_eq!(parse_value("1.5"), 1.5);
        assert_eq!(parse_value("0"), 0.0);
        assert!(parse_value("NA").is_nan());
        assert!(parse_value(".").is_nan());
        assert!(parse_value("").is_nan());
        // Boolean labels (pandas/R) map to 1.0 / 0.0.
        assert_eq!(parse_value("True"), 1.0);
        assert_eq!(parse_value("TRUE"), 1.0);
        assert_eq!(parse_value("False"), 0.0);
        assert_eq!(parse_value("false"), 0.0);
    }

    #[test]
    fn test_parse_phenotype_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("pheno.tsv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "IID\ty\tage\tsex").unwrap();
        writeln!(f, "S1\t1\t45\t1").unwrap();
        writeln!(f, "S2\t0\t50\t2").unwrap();
        writeln!(f, "S3\tNA\t55\t1").unwrap();

        let result =
            parse_phenotype_file(&path, "y", &["age".to_string(), "sex".to_string()], "IID")
                .unwrap();

        assert_eq!(result.sample_ids, vec!["S1", "S2", "S3"]);
        assert_eq!(result.phenotype[0], 1.0);
        assert_eq!(result.phenotype[1], 0.0);
        assert!(result.phenotype[2].is_nan());
        assert_eq!(result.covariates[0], vec![45.0, 1.0]);
    }

    #[test]
    fn test_valid_sample_indices() {
        let data = PhenotypeData {
            sample_ids: vec!["S1".into(), "S2".into(), "S3".into()],
            phenotype: vec![1.0, 0.0, f64::NAN],
            covariates: vec![vec![1.0], vec![2.0], vec![3.0]],
            covariate_names: vec!["x".into()],
        };
        let valid = valid_sample_indices(&data);
        assert_eq!(valid, vec![0, 1]);
    }
}
