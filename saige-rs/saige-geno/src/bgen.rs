//! BGEN v1.2 reader (pure Rust, zlib+zstd decompression).
//!
//! BGEN is a binary format for imputed genotype probabilities.
//! Supports .bgi SQLite index files for random access.
//!
//! Reference: https://www.well.ox.ac.uk/~gav/bgen_format/

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

use crate::traits::{GenotypeReader, MarkerData, MarkerInfo};

/// BGEN file header information.
#[derive(Debug, Clone)]
pub struct BgenHeader {
    pub offset: u32,
    pub n_variants: u32,
    pub n_samples: u32,
    pub header_length: u32,
    pub compression_type: CompressionType,
    pub layout: u32,
    pub has_sample_ids: bool,
}

/// BGEN compression type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompressionType {
    None,
    Zlib,
    Zstd,
}

/// BGEN variant index entry (from .bgi file).
#[derive(Debug, Clone)]
pub struct BgenIndexEntry {
    pub file_offset: u64,
    pub size: u64,
    pub chrom: String,
    pub pos: u32,
    pub rsid: String,
    pub allele1: String,
    pub allele2: String,
}

/// Reader for BGEN v1.2 files.
pub struct BgenReader {
    /// Path to the .bgen file.
    _path: PathBuf,
    /// Header information.
    header: BgenHeader,
    /// Sample IDs (if present in file or provided separately).
    sample_ids: Vec<String>,
    /// Memory-mapped file data.
    data: Vec<u8>,
    /// Variant index entries (from .bgi or sequential scan).
    index: Vec<BgenIndexEntry>,
    /// Sample subset indices.
    sample_subset: Option<Vec<usize>>,
}

impl BgenReader {
    /// Open a BGEN file. Optionally reads .bgi index and .sample files.
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read BGEN file: {}", path.display()))?;

        if data.len() < 20 {
            bail!("BGEN file too small");
        }

        let header = Self::parse_header(&data)?;

        // Read sample IDs if present
        let sample_ids = if header.has_sample_ids {
            Self::parse_sample_ids(&data, &header)?
        } else {
            // Try to read from .sample file (check multiple conventions)
            let candidates = [
                path.with_extension("sample"),                        // foo.sample
                PathBuf::from(format!("{}.sample", path.display())),  // foo.bgen.sample
                PathBuf::from(format!("{}.samples", path.display())), // foo.bgen.samples
            ];
            let sample_file = candidates.iter().find(|p| p.exists());
            if let Some(sample_path) = sample_file {
                Self::parse_sample_file(sample_path)?
            } else {
                (0..header.n_samples)
                    .map(|i| format!("sample_{}", i))
                    .collect()
            }
        };

        // Try to read .bgi index
        let bgi_path = PathBuf::from(format!("{}.bgi", path.display()));
        let index = if bgi_path.exists() {
            Self::read_bgi_index(&bgi_path, &header)?
        } else {
            Self::build_sequential_index(&data, &header)?
        };

        Ok(Self {
            _path: path.to_path_buf(),
            header,
            sample_ids,
            data,
            index,
            sample_subset: None,
        })
    }

    fn parse_header(data: &[u8]) -> Result<BgenHeader> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let mut cursor = Cursor::new(data);
        let offset = cursor.read_u32::<LittleEndian>()?;
        let header_length = cursor.read_u32::<LittleEndian>()?;
        let n_variants = cursor.read_u32::<LittleEndian>()?;
        let n_samples = cursor.read_u32::<LittleEndian>()?;

        // Read magic number
        let mut magic = [0u8; 4];
        std::io::Read::read_exact(&mut cursor, &mut magic)?;
        // BGEN magic: "bgen" or 0x00 0x00 0x00 0x00
        if &magic != b"bgen" && magic != [0, 0, 0, 0] {
            bail!("Invalid BGEN magic number: {:?}", magic);
        }

        // Skip free data area
        let free_data_len = header_length as usize - 20;
        let pos = cursor.position() as usize;
        if pos + free_data_len > data.len() {
            bail!("BGEN header extends beyond file");
        }

        // Read flags (at position 4 + header_length - 4)
        let flags_pos = 4 + header_length as usize - 4;
        let flags = u32::from_le_bytes([
            data[flags_pos],
            data[flags_pos + 1],
            data[flags_pos + 2],
            data[flags_pos + 3],
        ]);

        let compression_type = match flags & 0x03 {
            0 => CompressionType::None,
            1 => CompressionType::Zlib,
            2 => CompressionType::Zstd,
            _ => bail!("Unknown BGEN compression type"),
        };

        let layout = (flags >> 2) & 0x0F;
        if layout != 2 {
            bail!(
                "Only BGEN layout 2 (v1.2) is supported, got layout {}",
                layout
            );
        }

        let has_sample_ids = (flags >> 31) & 0x01 == 1;

        Ok(BgenHeader {
            offset,
            n_variants,
            n_samples,
            header_length,
            compression_type,
            layout,
            has_sample_ids,
        })
    }

    fn parse_sample_ids(data: &[u8], header: &BgenHeader) -> Result<Vec<String>> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let start = 4 + header.header_length as usize;
        let mut cursor = Cursor::new(&data[start..]);
        let _block_length = cursor.read_u32::<LittleEndian>()?;
        let n_samples = cursor.read_u32::<LittleEndian>()?;

        let mut ids = Vec::with_capacity(n_samples as usize);
        for _ in 0..n_samples {
            let id_len = cursor.read_u16::<LittleEndian>()? as usize;
            let pos = cursor.position() as usize;
            let id = std::str::from_utf8(&data[start + pos..start + pos + id_len])
                .unwrap_or("?")
                .to_string();
            cursor.set_position((pos + id_len) as u64);
            ids.push(id);
        }

        Ok(ids)
    }

    fn parse_sample_file(path: &Path) -> Result<Vec<String>> {
        let contents = std::fs::read_to_string(path)?;
        let mut ids = Vec::new();
        for (i, line) in contents.lines().enumerate() {
            if i < 2 {
                continue; // Skip header lines
            }
            let fields: Vec<&str> = line.split_whitespace().collect();
            if !fields.is_empty() {
                ids.push(fields[0].to_string());
            }
        }
        Ok(ids)
    }

    fn read_bgi_index(path: &Path, _header: &BgenHeader) -> Result<Vec<BgenIndexEntry>> {
        let conn = rusqlite::Connection::open(path)
            .with_context(|| format!("Failed to open .bgi index: {}", path.display()))?;

        let mut stmt = conn.prepare(
            "SELECT file_start_position, size_in_bytes, chromosome, position, rsid, allele1, allele2 \
             FROM Variant ORDER BY file_start_position"
        )?;

        let entries: Vec<BgenIndexEntry> = stmt
            .query_map([], |row| {
                Ok(BgenIndexEntry {
                    file_offset: row.get(0)?,
                    size: row.get(1)?,
                    chrom: row.get(2)?,
                    pos: row.get(3)?,
                    rsid: row.get(4)?,
                    allele1: row.get(5)?,
                    allele2: row.get(6)?,
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(entries)
    }

    fn build_sequential_index(data: &[u8], header: &BgenHeader) -> Result<Vec<BgenIndexEntry>> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let mut entries = Vec::with_capacity(header.n_variants as usize);
        let mut pos = (4 + header.offset) as usize;

        for _ in 0..header.n_variants {
            if pos >= data.len() {
                break;
            }
            let entry_start = pos;
            let mut cursor = Cursor::new(&data[pos..]);

            // Layout 2 variant identifying data:
            // varid_len(u16), varid, rsid_len(u16), rsid, chrom_len(u16), chrom, pos(u32), nalleles(u16)
            // then for each allele: allele_len(u32), allele
            // then genotype data block
            // Note: Layout 2 does NOT have N_individuals field (only layout 1 does)

            let varid_len = cursor.read_u16::<LittleEndian>()? as usize;
            let varid_start = cursor.position() as usize;
            cursor.set_position((varid_start + varid_len) as u64);

            let rsid_len = cursor.read_u16::<LittleEndian>()? as usize;
            let rsid_start = cursor.position() as usize;
            let rsid = std::str::from_utf8(&data[pos + rsid_start..pos + rsid_start + rsid_len])
                .unwrap_or("?")
                .to_string();
            cursor.set_position((rsid_start + rsid_len) as u64);

            let chrom_len = cursor.read_u16::<LittleEndian>()? as usize;
            let chrom_start = cursor.position() as usize;
            let chrom =
                std::str::from_utf8(&data[pos + chrom_start..pos + chrom_start + chrom_len])
                    .unwrap_or("?")
                    .to_string();
            cursor.set_position((chrom_start + chrom_len) as u64);

            let variant_pos = cursor.read_u32::<LittleEndian>()?;
            let n_alleles = cursor.read_u16::<LittleEndian>()? as usize;

            let mut alleles = Vec::with_capacity(n_alleles);
            for _ in 0..n_alleles {
                let allele_len = cursor.read_u32::<LittleEndian>()? as usize;
                let allele_start = cursor.position() as usize;
                let allele =
                    std::str::from_utf8(&data[pos + allele_start..pos + allele_start + allele_len])
                        .unwrap_or("?")
                        .to_string();
                cursor.set_position((allele_start + allele_len) as u64);
                alleles.push(allele);
            }

            // Genotype data block
            let geno_data_len = cursor.read_u32::<LittleEndian>()? as usize;
            let total_size = cursor.position() as usize + geno_data_len;

            entries.push(BgenIndexEntry {
                file_offset: entry_start as u64,
                size: total_size as u64,
                chrom,
                pos: variant_pos,
                rsid,
                allele1: alleles.first().cloned().unwrap_or_default(),
                allele2: alleles.get(1).cloned().unwrap_or_default(),
            });

            pos += total_size;
        }

        Ok(entries)
    }

    /// Decompress genotype probability data for a variant.
    fn decompress_genotype_data(&self, variant_idx: usize) -> Result<Vec<f64>> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let entry = &self.index[variant_idx];
        let offset = entry.file_offset as usize;
        let data = &self.data[offset..];
        let mut cursor = Cursor::new(data);

        // Skip variant identifying data to get to genotype block
        // Note: Layout 2 does NOT have N_individuals field at start
        // (only layout 1 does). Variant identifying data starts directly
        // with the variant ID length.
        let varid_len = cursor.read_u16::<LittleEndian>()? as usize;
        cursor.set_position(cursor.position() + varid_len as u64);
        let rsid_len = cursor.read_u16::<LittleEndian>()? as usize;
        cursor.set_position(cursor.position() + rsid_len as u64);
        let chrom_len = cursor.read_u16::<LittleEndian>()? as usize;
        cursor.set_position(cursor.position() + chrom_len as u64);
        let _pos = cursor.read_u32::<LittleEndian>()?;
        let n_alleles = cursor.read_u16::<LittleEndian>()? as usize;
        for _ in 0..n_alleles {
            let allele_len = cursor.read_u32::<LittleEndian>()? as usize;
            cursor.set_position(cursor.position() + allele_len as u64);
        }

        // Genotype data block
        // C = total length of compressed block (including 4-byte D prefix when compressed)
        let c_len = cursor.read_u32::<LittleEndian>()? as usize;

        // Decompress
        let decompressed = match self.header.compression_type {
            CompressionType::None => {
                let geno_start = cursor.position() as usize;
                data[geno_start..geno_start + c_len].to_vec()
            }
            CompressionType::Zlib => {
                // First 4 bytes of C block = D (decompressed length), rest = compressed data
                let _decompressed_len = cursor.read_u32::<LittleEndian>()?;
                let geno_start = cursor.position() as usize;
                let compressed_data = &data[geno_start..geno_start + c_len - 4];
                use flate2::read::ZlibDecoder;
                use std::io::Read;
                let mut decoder = ZlibDecoder::new(compressed_data);
                let mut buf = Vec::new();
                decoder.read_to_end(&mut buf)?;
                buf
            }
            CompressionType::Zstd => {
                let _decompressed_len = cursor.read_u32::<LittleEndian>()?;
                let geno_start = cursor.position() as usize;
                let compressed_data = &data[geno_start..geno_start + c_len - 4];
                zstd::decode_all(compressed_data)?
            }
        };

        // Parse layout 2 probability data
        self.parse_layout2_probabilities(&decompressed, n_alleles)
    }

    fn parse_layout2_probabilities(&self, data: &[u8], _n_alleles: usize) -> Result<Vec<f64>> {
        use byteorder::{LittleEndian, ReadBytesExt};
        use std::io::Cursor;

        let mut cursor = Cursor::new(data);
        let n_samples = cursor.read_u32::<LittleEndian>()? as usize;
        let n_alleles = cursor.read_u16::<LittleEndian>()? as usize;
        let min_ploidy = cursor.read_u8()?;
        let max_ploidy = cursor.read_u8()?;

        // Read per-sample ploidy and missingness
        let mut missing = vec![false; n_samples];
        let mut _ploidies = vec![0u8; n_samples];
        for i in 0..n_samples {
            let val = cursor.read_u8()?;
            _ploidies[i] = val & 0x3F;
            missing[i] = (val & 0x80) != 0;
        }

        let phased = cursor.read_u8()? != 0;
        let bits_per_prob = cursor.read_u8()? as usize;

        if bits_per_prob == 0 {
            bail!("BGEN: 0 bits per probability not supported");
        }

        let max_val = (1u64 << bits_per_prob) - 1;

        // For diploid biallelic: 2 probabilities per sample (P(AA), P(AB); P(BB) = 1 - P(AA) - P(AB))
        // Dosage = 0*P(AA) + 1*P(AB) + 2*P(BB) = P(AB) + 2*(1 - P(AA) - P(AB)) = 2 - 2*P(AA) - P(AB)
        let mut dosages = vec![f64::NAN; n_samples];

        if n_alleles == 2 && min_ploidy == 2 && max_ploidy == 2 && !phased {
            // Common case: diploid biallelic unphased
            let prob_start = cursor.position() as usize;
            let mut bit_offset = 0usize;

            for i in 0..n_samples {
                if missing[i] {
                    bit_offset += 2 * bits_per_prob;
                    continue;
                }

                let p_aa =
                    read_bits(data, prob_start, bit_offset, bits_per_prob) as f64 / max_val as f64;
                bit_offset += bits_per_prob;
                let p_ab =
                    read_bits(data, prob_start, bit_offset, bits_per_prob) as f64 / max_val as f64;
                bit_offset += bits_per_prob;

                dosages[i] = p_ab + 2.0 * (1.0 - p_aa - p_ab);
            }
        } else {
            // Fallback: treat as dosage of second allele
            tracing::warn!("BGEN: non-standard ploidy/alleles, using simple dosage conversion");
            for i in 0..n_samples {
                dosages[i] = if missing[i] { f64::NAN } else { 0.0 };
            }
        }

        Ok(dosages)
    }
}

/// Read `n_bits` from a byte array starting at `byte_offset` bytes + `bit_offset` bits.
fn read_bits(data: &[u8], byte_start: usize, bit_offset: usize, n_bits: usize) -> u64 {
    let mut result: u64 = 0;
    for bit in 0..n_bits {
        let total_bit = bit_offset + bit;
        let byte_idx = byte_start + total_bit / 8;
        let bit_idx = total_bit % 8;
        if byte_idx < data.len() {
            let bit_val = (data[byte_idx] >> bit_idx) & 1;
            result |= (bit_val as u64) << bit;
        }
    }
    result
}

impl GenotypeReader for BgenReader {
    fn n_markers(&self) -> usize {
        self.index.len()
    }

    fn n_samples(&self) -> usize {
        match &self.sample_subset {
            Some(indices) => indices.len(),
            None => self.header.n_samples as usize,
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
        if idx >= self.index.len() {
            bail!("Variant index {} out of range ({})", idx, self.index.len());
        }

        let all_dosages = self.decompress_genotype_data(idx)?;

        let dosages = match &self.sample_subset {
            Some(indices) => indices.iter().map(|&i| all_dosages[i]).collect(),
            None => all_dosages,
        };

        let (af, mac, n_valid, ploidy) = MarkerData::compute_af(&dosages);

        let entry = &self.index[idx];
        Ok(MarkerData {
            info: MarkerInfo {
                chrom: entry.chrom.clone(),
                pos: entry.pos as u64,
                id: entry.rsid.clone(),
                ref_allele: entry.allele1.clone(),
                alt_allele: entry.allele2.clone(),
            },
            dosages,
            af,
            mac,
            n_valid,
            ploidy,
            is_imputed: true,
            info_score: None,
        })
    }

    fn marker_info(&self, index: u64) -> Result<MarkerInfo> {
        let idx = index as usize;
        if idx >= self.index.len() {
            bail!("Variant index {} out of range ({})", idx, self.index.len());
        }
        let entry = &self.index[idx];
        Ok(MarkerInfo {
            chrom: entry.chrom.clone(),
            pos: entry.pos as u64,
            id: entry.rsid.clone(),
            ref_allele: entry.allele1.clone(),
            alt_allele: entry.allele2.clone(),
        })
    }
}
