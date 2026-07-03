//! VCF/BCF reader via noodles crate.
//!
//! Reads genotype data from VCF and BCF files. Supports both
//! hard-call (GT) and dosage (DS) fields.

use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

use crate::traits::{GenotypeReader, MarkerData, MarkerInfo};

/// Reader for VCF/BCF files.
pub struct VcfReader {
    /// Path to the VCF/BCF file.
    path: PathBuf,
    /// Sample IDs.
    sample_ids: Vec<String>,
    /// Pre-scanned variant info.
    variants: Vec<VcfVariantInfo>,
    /// Sample subset indices.
    sample_subset: Option<Vec<usize>>,
    /// Whether to prefer DS (dosage) over GT (genotype) field.
    prefer_dosage: bool,
}

#[derive(Debug, Clone)]
struct VcfVariantInfo {
    chrom: String,
    pos: u64,
    id: String,
    ref_allele: String,
    alt_allele: String,
    /// File offset or record index for seeking.
    record_index: usize,
}

impl VcfReader {
    /// Open a VCF file and scan variant metadata.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Read header and scan variants
        let contents = std::fs::read_to_string(&path)?;
        let mut sample_ids = Vec::new();
        let mut variants = Vec::new();
        let mut record_index = 0;

        for line in contents.lines() {
            if line.starts_with("##") {
                continue;
            }
            if line.starts_with('#') {
                // Header line with sample IDs
                let fields: Vec<&str> = line.split('\t').collect();
                if fields.len() > 9 {
                    sample_ids = fields[9..].iter().map(|s| s.to_string()).collect();
                }
                continue;
            }

            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 8 {
                continue;
            }

            let alt_alleles: Vec<&str> = fields[4].split(',').collect();
            variants.push(VcfVariantInfo {
                chrom: fields[0].to_string(),
                pos: fields[1].parse().unwrap_or(0),
                id: if fields[2] == "." {
                    format!("{}:{}", fields[0], fields[1])
                } else {
                    fields[2].to_string()
                },
                ref_allele: fields[3].to_string(),
                alt_allele: alt_alleles.first().unwrap_or(&".").to_string(),
                record_index,
            });
            record_index += 1;
        }

        Ok(Self {
            path,
            sample_ids,
            variants,
            sample_subset: None,
            prefer_dosage: true,
        })
    }

    /// Set whether to prefer DS (dosage) field over GT (genotype) field.
    pub fn set_prefer_dosage(&mut self, prefer: bool) {
        self.prefer_dosage = prefer;
    }

    /// Parse genotype dosages from a VCF record line.
    fn parse_record_dosages(&self, record_idx: usize) -> Result<Vec<f64>> {
        let contents = std::fs::read_to_string(&self.path)?;
        let mut current_idx = 0;

        for line in contents.lines() {
            if line.starts_with('#') {
                continue;
            }
            if current_idx == record_idx {
                return self.parse_vcf_line_dosages(line);
            }
            current_idx += 1;
        }

        bail!("Record index {} not found", record_idx);
    }

    fn parse_vcf_line_dosages(&self, line: &str) -> Result<Vec<f64>> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            bail!("VCF line has fewer than 10 fields");
        }

        let format_str = fields[8];
        let format_fields: Vec<&str> = format_str.split(':').collect();

        // Find DS and GT field indices
        let ds_idx = format_fields.iter().position(|&f| f == "DS");
        let gt_idx = format_fields.iter().position(|&f| f == "GT");

        let n_samples = fields.len() - 9;
        let mut dosages = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let sample_data = fields[9 + i];
            let sample_fields: Vec<&str> = sample_data.split(':').collect();

            let dosage = if let (true, Some(ds_i)) = (self.prefer_dosage, ds_idx) {
                // Use DS field
                if ds_i < sample_fields.len() {
                    sample_fields[ds_i].parse().unwrap_or(f64::NAN)
                } else {
                    f64::NAN
                }
            } else if let Some(gt_i) = gt_idx {
                // Parse GT field
                if gt_i < sample_fields.len() {
                    parse_gt_dosage(sample_fields[gt_i])
                } else {
                    f64::NAN
                }
            } else {
                f64::NAN
            };

            dosages.push(dosage);
        }

        Ok(dosages)
    }
}

/// Parse a GT field (e.g., "0/1", "1|0", "./.") to dosage of ALT allele.
fn parse_gt_dosage(gt: &str) -> f64 {
    let sep = if gt.contains('|') { '|' } else { '/' };
    let alleles: Vec<&str> = gt.split(sep).collect();
    let mut dosage = 0.0;
    for allele in &alleles {
        match *allele {
            "." => return f64::NAN,
            "0" => {}
            _ => dosage += 1.0,
        }
    }
    dosage
}

impl GenotypeReader for VcfReader {
    fn n_markers(&self) -> usize {
        self.variants.len()
    }

    fn n_samples(&self) -> usize {
        match &self.sample_subset {
            Some(indices) => indices.len(),
            None => self.sample_ids.len(),
        }
    }

    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn set_sample_subset(&mut self, ids: &[String]) -> Result<()> {
        let mut indices = Vec::new();
        let mut new_ids = Vec::new();
        for id in ids {
            if let Some(pos) = self.sample_ids.iter().position(|s| s == id) {
                indices.push(pos);
                new_ids.push(id.clone());
            }
        }
        self.sample_subset = Some(indices);
        self.sample_ids = new_ids;
        Ok(())
    }

    fn read_marker(&mut self, index: u64) -> Result<MarkerData> {
        let idx = index as usize;
        if idx >= self.variants.len() {
            bail!(
                "Variant index {} out of range ({})",
                idx,
                self.variants.len()
            );
        }

        let record_idx = self.variants[idx].record_index;
        let all_dosages = self.parse_record_dosages(record_idx)?;

        let dosages = match &self.sample_subset {
            Some(indices) => indices.iter().map(|&i| all_dosages[i]).collect(),
            None => all_dosages,
        };

        let (af, mac, n_valid, ploidy) = MarkerData::compute_af(&dosages);

        let v = &self.variants[idx];
        Ok(MarkerData {
            info: MarkerInfo {
                chrom: v.chrom.clone(),
                pos: v.pos,
                id: v.id.clone(),
                ref_allele: v.ref_allele.clone(),
                alt_allele: v.alt_allele.clone(),
            },
            dosages,
            af,
            mac,
            n_valid,
            ploidy,
            is_imputed: false,
            info_score: None,
        })
    }

    fn marker_info(&self, index: u64) -> Result<MarkerInfo> {
        let idx = index as usize;
        if idx >= self.variants.len() {
            bail!("Variant index {} out of range", idx);
        }
        let v = &self.variants[idx];
        Ok(MarkerInfo {
            chrom: v.chrom.clone(),
            pos: v.pos,
            id: v.id.clone(),
            ref_allele: v.ref_allele.clone(),
            alt_allele: v.alt_allele.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gt_dosage() {
        assert_eq!(parse_gt_dosage("0/0"), 0.0);
        assert_eq!(parse_gt_dosage("0/1"), 1.0);
        assert_eq!(parse_gt_dosage("1/1"), 2.0);
        assert_eq!(parse_gt_dosage("0|1"), 1.0);
        assert_eq!(parse_gt_dosage("1|0"), 1.0);
        assert!(parse_gt_dosage("./.").is_nan());
        assert!(parse_gt_dosage(".|.").is_nan());
    }
}
