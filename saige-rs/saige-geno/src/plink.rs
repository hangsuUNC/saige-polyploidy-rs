//! PLINK bed/bim/fam reader using memory-mapped files.
//!
//! PLINK binary format consists of three files:
//! - .bed: Binary genotype data (2 bits per genotype, packed)
//! - .bim: Variant information (chrom, id, cm, pos, a1, a2)
//! - .fam: Sample information (fid, iid, father, mother, sex, pheno)
//!
//! Reference: https://www.cog-genomics.org/plink/1.9/formats#bed

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;

use crate::traits::{GenotypeReader, MarkerData, MarkerInfo};

/// PLINK BIM file entry (one per variant).
#[derive(Debug, Clone)]
pub struct BimEntry {
    pub chrom: String,
    pub id: String,
    pub cm: f64,
    pub pos: u64,
    pub allele1: String, // Usually ALT / minor allele
    pub allele2: String, // Usually REF / major allele
}

/// PLINK FAM file entry (one per sample).
#[derive(Debug, Clone)]
pub struct FamEntry {
    pub fid: String,
    pub iid: String,
    pub father: String,
    pub mother: String,
    pub sex: i32,
    pub pheno: f64,
}

/// Reader for PLINK bed/bim/fam files.
pub struct PlinkReader {
    /// Memory-mapped .bed file.
    mmap: Mmap,
    /// Variant information from .bim file.
    bim: Vec<BimEntry>,
    /// Sample information from .fam file.
    fam: Vec<FamEntry>,
    /// Number of samples (total, before subsetting).
    n_samples_total: usize,
    /// Number of bytes per marker in the bed file.
    bytes_per_marker: usize,
    /// Sample IDs (IID).
    sample_ids: Vec<String>,
    /// Indices of selected samples (None = all samples).
    sample_subset: Option<Vec<usize>>,
    /// Base path (without extension).
    _base_path: PathBuf,
}

impl PlinkReader {
    /// Open PLINK files from a base path (without extension).
    /// Will look for .bed, .bim, .fam files.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Result<Self> {
        let base = base_path.as_ref();
        let bed_path = base.with_extension("bed");
        let bim_path = base.with_extension("bim");
        let fam_path = base.with_extension("fam");

        // Parse .fam file
        let fam = Self::parse_fam(&fam_path)?;
        let n_samples = fam.len();

        // Parse .bim file
        let bim = Self::parse_bim(&bim_path)?;

        // Memory-map .bed file
        let bed_file = std::fs::File::open(&bed_path)
            .with_context(|| format!("Failed to open bed file: {}", bed_path.display()))?;
        let mmap = unsafe { Mmap::map(&bed_file)? };

        // Validate bed file magic number
        if mmap.len() < 3 {
            bail!("Bed file too small");
        }
        if mmap[0] != 0x6C || mmap[1] != 0x1B {
            bail!("Invalid PLINK bed file magic number");
        }
        if mmap[2] != 0x01 {
            bail!("Only SNP-major bed files are supported (mode byte = 0x01)");
        }

        let bytes_per_marker = n_samples.div_ceil(4);
        let expected_size = 3 + bytes_per_marker * bim.len();
        if mmap.len() < expected_size {
            bail!(
                "Bed file too small: expected at least {} bytes, got {}",
                expected_size,
                mmap.len()
            );
        }

        let sample_ids: Vec<String> = fam.iter().map(|f| f.iid.clone()).collect();

        Ok(Self {
            mmap,
            bim,
            fam,
            n_samples_total: n_samples,
            bytes_per_marker,
            sample_ids,
            sample_subset: None,
            _base_path: base.to_path_buf(),
        })
    }

    /// Parse a .fam file.
    fn parse_fam(path: &Path) -> Result<Vec<FamEntry>> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read fam file: {}", path.display()))?;
        let mut entries = Vec::new();
        for (line_num, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 6 {
                bail!("Fam file line {} has fewer than 6 fields", line_num + 1);
            }
            entries.push(FamEntry {
                fid: fields[0].to_string(),
                iid: fields[1].to_string(),
                father: fields[2].to_string(),
                mother: fields[3].to_string(),
                sex: fields[4].parse().unwrap_or(0),
                pheno: fields[5].parse().unwrap_or(f64::NAN),
            });
        }
        Ok(entries)
    }

    /// Parse a .bim file.
    fn parse_bim(path: &Path) -> Result<Vec<BimEntry>> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read bim file: {}", path.display()))?;
        let mut entries = Vec::new();
        for (line_num, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 6 {
                bail!("Bim file line {} has fewer than 6 fields", line_num + 1);
            }
            entries.push(BimEntry {
                chrom: fields[0].to_string(),
                id: fields[1].to_string(),
                cm: fields[2].parse().unwrap_or(0.0),
                pos: fields[3].parse().unwrap_or(0),
                allele1: fields[4].to_string(),
                allele2: fields[5].to_string(),
            });
        }
        Ok(entries)
    }

    /// Decode a single genotype from the bed file.
    /// Returns dosage of allele1 (usually ALT): 0, 1, 2, or NaN for missing.
    #[inline]
    fn decode_genotype(byte: u8, offset: usize) -> f64 {
        let bits = (byte >> (offset * 2)) & 0x03;
        match bits {
            0b00 => 2.0,      // Homozygous A1/A1 (ALT/ALT in PLINK convention)
            0b01 => f64::NAN, // Missing
            0b10 => 1.0,      // Heterozygous A1/A2
            0b11 => 0.0,      // Homozygous A2/A2 (REF/REF)
            _ => unreachable!(),
        }
    }

    /// Read raw genotypes for a marker (all samples, no subsetting).
    fn read_marker_raw(&self, index: u64) -> Vec<f64> {
        let marker_idx = index as usize;
        let offset = 3 + marker_idx * self.bytes_per_marker;
        let mut dosages = Vec::with_capacity(self.n_samples_total);

        for sample_idx in 0..self.n_samples_total {
            let byte_idx = offset + sample_idx / 4;
            let bit_offset = sample_idx % 4;
            let byte = self.mmap[byte_idx];
            dosages.push(Self::decode_genotype(byte, bit_offset));
        }
        dosages
    }

    /// Get FAM entries.
    pub fn fam(&self) -> &[FamEntry] {
        &self.fam
    }

    /// Get BIM entries.
    pub fn bim(&self) -> &[BimEntry] {
        &self.bim
    }
}

impl GenotypeReader for PlinkReader {
    fn n_markers(&self) -> usize {
        self.bim.len()
    }

    fn n_samples(&self) -> usize {
        match &self.sample_subset {
            Some(indices) => indices.len(),
            None => self.n_samples_total,
        }
    }

    fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    fn set_sample_subset(&mut self, ids: &[String]) -> Result<()> {
        let mut indices = Vec::new();
        let mut new_ids = Vec::new();
        for id in ids {
            if let Some(pos) = self.fam.iter().position(|f| &f.iid == id) {
                indices.push(pos);
                new_ids.push(id.clone());
            }
        }
        self.sample_subset = Some(indices);
        self.sample_ids = new_ids;
        Ok(())
    }

    fn read_marker(&mut self, index: u64) -> Result<MarkerData> {
        if index as usize >= self.bim.len() {
            bail!("Marker index {} out of range ({})", index, self.bim.len());
        }

        let all_dosages = self.read_marker_raw(index);

        let dosages = match &self.sample_subset {
            Some(indices) => indices.iter().map(|&i| all_dosages[i]).collect(),
            None => all_dosages,
        };

        let (af, mac, n_valid, ploidy) = MarkerData::compute_af(&dosages);

        let bim = &self.bim[index as usize];
        Ok(MarkerData {
            info: MarkerInfo {
                chrom: bim.chrom.clone(),
                pos: bim.pos,
                id: bim.id.clone(),
                ref_allele: bim.allele2.clone(),
                alt_allele: bim.allele1.clone(),
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
        if index as usize >= self.bim.len() {
            bail!("Marker index {} out of range ({})", index, self.bim.len());
        }
        let bim = &self.bim[index as usize];
        Ok(MarkerInfo {
            chrom: bim.chrom.clone(),
            pos: bim.pos,
            id: bim.id.clone(),
            ref_allele: bim.allele2.clone(),
            alt_allele: bim.allele1.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_genotype() {
        assert_eq!(PlinkReader::decode_genotype(0b00_00_00_00, 0), 2.0); // Hom ALT
        assert!(PlinkReader::decode_genotype(0b00_00_00_01, 0).is_nan()); // Missing
        assert_eq!(PlinkReader::decode_genotype(0b00_00_00_10, 0), 1.0); // Het
        assert_eq!(PlinkReader::decode_genotype(0b00_00_00_11, 0), 0.0); // Hom REF
    }

    #[test]
    fn test_decode_genotype_offsets() {
        let byte: u8 = 0b11_10_01_00; // sample3=HOM_REF, sample2=HET, sample1=MISSING, sample0=HOM_ALT
        assert_eq!(PlinkReader::decode_genotype(byte, 0), 2.0);
        assert!(PlinkReader::decode_genotype(byte, 1).is_nan());
        assert_eq!(PlinkReader::decode_genotype(byte, 2), 1.0);
        assert_eq!(PlinkReader::decode_genotype(byte, 3), 0.0);
    }
}
