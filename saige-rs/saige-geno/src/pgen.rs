//! PGEN format reader (placeholder for FFI to plink2 C lib).
//!
//! PGEN is the PLINK 2.0 binary genotype format, which is more
//! efficient than PLINK 1.x bed files. Full implementation requires
//! linking against the pgenlib C library.

use anyhow::{bail, Result};

use crate::traits::{GenotypeReader, MarkerData, MarkerInfo, PloidyMode};

/// Reader for PGEN files (stub - requires pgenlib C FFI).
pub struct PgenReader {
    _path: std::path::PathBuf,
}

impl PgenReader {
    pub fn new<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        bail!("PGEN reader not yet implemented (requires pgenlib C FFI)")
    }
}

impl GenotypeReader for PgenReader {
    fn n_markers(&self) -> usize {
        0
    }
    fn n_samples(&self) -> usize {
        0
    }
    fn sample_ids(&self) -> &[String] {
        &[]
    }
    fn set_sample_subset(&mut self, _ids: &[String]) -> Result<()> {
        Ok(())
    }
    fn set_ploidy(&mut self, _mode: PloidyMode) {}
    fn read_marker(&mut self, _index: u64) -> Result<MarkerData> {
        bail!("PGEN reader not yet implemented")
    }
    fn marker_info(&self, _index: u64) -> Result<MarkerInfo> {
        bail!("PGEN reader not yet implemented")
    }
}
