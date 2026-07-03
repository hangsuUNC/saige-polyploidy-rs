//! Core traits for genotype reading.

use anyhow::Result;

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
    /// Infer the per-marker copy-number maximum ("ploidy") used to normalize
    /// allele frequency and count.
    ///
    /// Uses the maximum observed non-missing dosage for the marker, floored at
    /// 2.0. The floor makes ordinary diploid markers (dosage in [0, 2]) behave
    /// exactly as before, while genuine CNV / polyploid markers (dosage > 2)
    /// get a denominator matching their actual copy-number range.
    pub fn infer_ploidy(dosages: &[f64]) -> f64 {
        let mut max_d = 0.0_f64;
        for &d in dosages {
            if !d.is_nan() && d > max_d {
                max_d = d;
            }
        }
        max_d.max(2.0)
    }

    /// Compute allele frequency, minor allele count, valid N, and inferred ploidy.
    ///
    /// AF and MAC are normalized by the per-marker copy-number maximum (`ploidy`)
    /// instead of a hard-coded diploid `2`, so CNV / polyploid dosages (> 2)
    /// yield AF in [0, 1] and a non-negative MAC.
    pub fn compute_af(dosages: &[f64]) -> (f64, f64, usize, f64) {
        let ploidy = Self::infer_ploidy(dosages);
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
        // Guard against negative MAC: `total >= sum` always holds because
        // `ploidy >= max dosage`, so `total - sum >= 0`.
        let mac = if n > 0 { sum.min(total - sum) } else { 0.0 };
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
        // Ordinary diploid marker: max dosage 2 -> ploidy 2, behaves as before.
        let g = vec![0.0, 1.0, 2.0, 1.0, 0.0];
        let (af, mac, n, ploidy) = MarkerData::compute_af(&g);
        assert_eq!(n, 5);
        assert_eq!(ploidy, 2.0);
        // sum = 4, 2N = 10 -> af 0.4, mac min(4, 6) = 4
        assert!((af - 0.4).abs() < 1e-12);
        assert!((mac - 4.0).abs() < 1e-12);
    }

    #[test]
    fn rare_diploid_snp_floors_ploidy_at_two() {
        // No homozygous-alt observed (max dosage 1) must still use ploidy 2,
        // otherwise AF would double and diverge from standard SAIGE.
        let g = vec![0.0, 1.0, 0.0, 0.0, 1.0];
        let (af, _mac, _n, ploidy) = MarkerData::compute_af(&g);
        assert_eq!(ploidy, 2.0);
        // sum = 2, 2N = 10 -> af 0.2 (not 0.4)
        assert!((af - 0.2).abs() < 1e-12);
    }

    #[test]
    fn cnv_af_in_unit_interval_and_mac_nonnegative() {
        // CNV / copy-number dosages in [0, 8], most samples at 2.
        let g = vec![2.0, 2.0, 2.0, 2.0, 8.0, 2.0, 2.0, 6.0, 2.0, 2.0];
        let (af, mac, n, ploidy) = MarkerData::compute_af(&g);
        assert_eq!(n, 10);
        assert_eq!(ploidy, 8.0);
        // AF must be a valid frequency, not > 1 as with the old 2N denominator.
        assert!((0.0..=1.0).contains(&af), "af={af}");
        // MAC must never go negative (the old formula gave 2N - sum < 0 here).
        assert!(mac >= 0.0, "mac={mac}");
    }

    #[test]
    fn cnv_missing_imputed_with_expected_dosage() {
        let g = vec![2.0, 4.0, f64::NAN, 6.0, 2.0];
        let (af, _mac, _n, ploidy) = MarkerData::compute_af(&g);
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
