//! Core traits for genotype reading.

use anyhow::{bail, Result};

/// How to determine the copy-number maximum ("ploidy") used to normalize
/// allele frequency and count for a marker.
///
/// This replaces the hard-coded diploid `2` and lets one code path serve
/// diploid GWAS, haploid analyses (e.g. mitochondrial variants), and
/// CNV / polyploid dosages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PloidyMode {
    /// Standard diploid: denominator = 2. Default; SNP behavior unchanged.
    Diploid,
    /// Haploid: denominator = 1 (e.g. mitochondrial homoplasmic/heteroplasmic
    /// variants, chrY). AF then reflects true carrier/allele fraction in [0, 1].
    Haploid,
    /// Fixed user-specified copy number (denominator = n).
    Fixed(f64),
    /// Infer per marker as `max(2.0, max non-missing dosage)` — for CNV /
    /// tandem-repeat dosages whose scale varies by locus.
    Auto,
}

impl PloidyMode {
    /// Resolve the numeric ploidy (per-sample denominator) for a marker's dosages.
    pub fn resolve(&self, dosages: &[f64]) -> f64 {
        match self {
            PloidyMode::Diploid => 2.0,
            PloidyMode::Haploid => 1.0,
            PloidyMode::Fixed(n) => *n,
            PloidyMode::Auto => {
                let mut m = 2.0_f64;
                for &d in dosages {
                    if !d.is_nan() && d > m {
                        m = d;
                    }
                }
                m
            }
        }
    }

    /// Parse a CLI string: "diploid" (2), "haploid" (1), "auto", or a number.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "diploid" | "2" => Ok(PloidyMode::Diploid),
            "haploid" | "1" => Ok(PloidyMode::Haploid),
            "auto" => Ok(PloidyMode::Auto),
            other => match other.parse::<f64>() {
                Ok(n) if n > 0.0 => Ok(PloidyMode::Fixed(n)),
                _ => bail!(
                    "invalid --ploidy '{s}'; expected diploid|haploid|auto or a positive number"
                ),
            },
        }
    }
}

impl Default for PloidyMode {
    fn default() -> Self {
        PloidyMode::Diploid
    }
}

/// Information about a genetic marker (variant).
#[derive(Debug, Clone)]
pub struct MarkerInfo {
    /// Chromosome (e.g. "1", "22", "X").
    pub chrom: String,
    /// Position in base pairs.
    pub pos: u64,
    /// Marker/variant ID (e.g. rsID).
    pub id: String,
    /// Reference allele.
    pub ref_allele: String,
    /// Alternative allele.
    pub alt_allele: String,
}

/// Data for a single marker across all samples.
#[derive(Debug, Clone)]
pub struct MarkerData {
    /// Marker metadata.
    pub info: MarkerInfo,
    /// Dosage values for each sample (0.0 to 2.0).
    /// Missing values represented as NaN.
    pub dosages: Vec<f64>,
    /// Allele frequency of the alt allele.
    pub af: f64,
    /// Minor allele count.
    pub mac: f64,
    /// Number of non-missing samples.
    pub n_valid: usize,
    /// Per-marker copy-number maximum ("ploidy") used to normalize AF/MAC.
    /// Equals `max(2.0, max non-missing dosage)`; 2.0 for ordinary diploid markers.
    pub ploidy: f64,
    /// Whether the marker is imputed (true) or genotyped (false).
    pub is_imputed: bool,
    /// Imputation quality (info score), if available.
    pub info_score: Option<f64>,
}

impl MarkerData {
    /// Compute allele frequency, minor allele count, valid N, and the resolved
    /// ploidy for a marker under the given [`PloidyMode`].
    ///
    /// AF and MAC are normalized by the resolved ploidy (per-sample denominator)
    /// instead of a hard-coded diploid `2`, so haploid (mtDNA) and CNV /
    /// polyploid dosages yield AF in [0, 1] and a non-negative MAC.
    pub fn compute_af(dosages: &[f64], ploidy_mode: PloidyMode) -> (f64, f64, usize, f64) {
        let ploidy = ploidy_mode.resolve(dosages);
        let mut sum = 0.0;
        let mut n = 0usize;
        for &d in dosages {
            if !d.is_nan() {
                sum += d;
                n += 1;
            }
        }
        let total = ploidy * n as f64;
        let af = if n > 0 { sum / total } else { 0.0 };
        // MAC is clamped at 0: for Diploid/Auto `total >= sum` always holds; for
        // Haploid/Fixed a stray dosage above the ploidy could otherwise make
        // `total - sum` negative.
        let mac = if n > 0 {
            sum.min(total - sum).max(0.0)
        } else {
            0.0
        };
        (af, mac, n, ploidy)
    }

    /// Impute missing dosages with the expected dosage (`ploidy * af`, i.e. the
    /// non-missing mean), generalizing the diploid `2 * af` rule.
    pub fn impute_missing(&mut self) {
        let impute_val = self.ploidy * self.af;
        for d in &mut self.dosages {
            if d.is_nan() {
                *d = impute_val;
            }
        }
    }
}

/// Trait for reading genotype data from various file formats.
///
/// Each format (PLINK, BGEN, VCF, SAV, PGEN) implements this trait.
/// Static dispatch via generics in hot loops; dynamic dispatch
/// (`Box<dyn GenotypeReader>`) at the CLI level.
pub trait GenotypeReader: Send {
    /// Total number of markers in the file.
    fn n_markers(&self) -> usize;

    /// Total number of samples in the file.
    fn n_samples(&self) -> usize;

    /// Get the list of sample IDs.
    fn sample_ids(&self) -> &[String];

    /// Set a sample subset for reading. Only these samples will be
    /// included in subsequent `read_marker` calls.
    fn set_sample_subset(&mut self, ids: &[String]) -> Result<()>;

    /// Set the ploidy mode used to normalize AF/MAC in `read_marker`.
    /// Readers default to [`PloidyMode::Diploid`] until this is called.
    fn set_ploidy(&mut self, mode: PloidyMode);

    /// Read genotype data for marker at the given index.
    fn read_marker(&mut self, index: u64) -> Result<MarkerData>;

    /// Get marker info without reading genotype data.
    fn marker_info(&self, index: u64) -> Result<MarkerInfo>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diploid_af_mac_unchanged() {
        // Ordinary diploid marker: default mode -> ploidy 2, behaves as before.
        let g = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let (af, mac, n, ploidy) = MarkerData::compute_af(&g, PloidyMode::Diploid);
        assert_eq!(n, 5);
        assert_eq!(ploidy, 2.0);
        // sum = 4, 2N = 10 -> af 0.4, mac min(4, 6) = 4
        assert!((af - 0.4).abs() < 1e-12);
        assert!((mac - 4.0).abs() < 1e-12);
    }

    #[test]
    fn haploid_af_uses_denominator_one() {
        // Haploid (e.g. mtDNA homoplasmic): 0/1 dosages. AF must be sum/N, not
        // sum/2N — a variant carried by everyone should read AF = 1.0.
        let g = vec![1.0, 1.0, 1.0, 0.0, 1.0];
        let (af, mac, _n, ploidy) = MarkerData::compute_af(&g, PloidyMode::Haploid);
        assert_eq!(ploidy, 1.0);
        // sum = 4, N = 5 -> af 0.8 (would be 0.4 under the diploid denominator).
        assert!((af - 0.8).abs() < 1e-12, "af={af}");
        // mac = min(4, 5-4) = 1
        assert!((mac - 1.0).abs() < 1e-12, "mac={mac}");
    }

    #[test]
    fn haploid_heteroplasmy_continuous_vaf() {
        // Heteroplasmic mtDNA variant: continuous VAF in [0, 1].
        let g = vec![0.1, 0.0, 0.9, 0.3, 0.0];
        let (af, mac, _n, ploidy) = MarkerData::compute_af(&g, PloidyMode::Haploid);
        assert_eq!(ploidy, 1.0);
        assert!((0.0..=1.0).contains(&af), "af={af}");
        assert!(mac >= 0.0, "mac={mac}");
    }

    #[test]
    fn cnv_auto_af_in_unit_interval_and_mac_nonnegative() {
        // CNV / copy-number dosages in [0, 8], most samples at 2, Auto mode.
        let g = vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0, 2.0, 6.0, 2.0, 2.0];
        let (af, mac, n, ploidy) = MarkerData::compute_af(&g, PloidyMode::Auto);
        assert_eq!(n, 10);
        assert_eq!(ploidy, 8.0);
        // AF must be a valid frequency, not > 1 as with the old 2N denominator.
        assert!((0.0..=1.0).contains(&af), "af={af}");
        // MAC must never go negative (the old formula gave 2N - sum < 0 here).
        assert!(mac >= 0.0, "mac={mac}");
    }

    #[test]
    fn ploidy_parse() {
        assert_eq!(PloidyMode::parse("diploid").unwrap(), PloidyMode::Diploid);
        assert_eq!(PloidyMode::parse("haploid").unwrap(), PloidyMode::Haploid);
        assert_eq!(PloidyMode::parse("AUTO").unwrap(), PloidyMode::Auto);
        assert_eq!(PloidyMode::parse("4").unwrap(), PloidyMode::Fixed(4.0));
        assert!(PloidyMode::parse("nonsense").is_err());
    }

    #[test]
    fn cnv_missing_imputed_with_expected_dosage() {
        let g = vec![2.0, 4.0, f64::NAN, 6.0, 2.0];
        let (af, _mac, _n, ploidy) = MarkerData::compute_af(&g, PloidyMode::Auto);
        let mut md = MarkerData {
            info: MarkerInfo {
                chrom: "chrM".into(),
                pos: 1,
                id: "cnv1".into(),
                ref_allele: "A".into(),
                alt_allele: "G".into(),
            },
            dosages: g,
            af,
            mac: 0.0,
            n_valid: 4,
            ploidy,
            is_imputed: false,
            info_score: None,
        };
        md.impute_missing();
        let filled = md.dosages[2];
        // Expected dosage = ploidy * af = non-missing mean, and within [0, ploidy].
        assert!(filled.is_finite());
        assert!((0.0..=ploidy).contains(&filled), "filled={filled}");
    }
}
