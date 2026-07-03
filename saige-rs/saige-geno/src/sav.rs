//! SAV format reader (placeholder for FFI to savvy C++ lib).
//!
//! SAV is the format used by the savvy library for efficient
//! genotype storage. Full implementation requires linking against
//! the savvy C++ library.

use anyhow::{bail, Result};

use crate::traits::{GenotypeReader, MarkerData, MarkerInfo, PloidyMode};

/// Reader for SAV files (stub - requires FFI to savvy C++ library).
pub struct SavReader {
    _path: std::path::PathBuf,
}

impl SavReader {
    pub fn new<P: AsRef<std::path::Path>>(_path: P) -> Result<Self> {
        bail!("SAV reader not yet implemented (requires savvy C++ FFI)")
    }
}

impl GenotypeReader for SavReader {
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
        bail!("SAV reader not yet implemented")
    }
    fn marker_info(&self, _index: u64) -> Result<MarkerInfo> {
        bail!("SAV reader not yet implemented")
    }
}
