/// Organism-specific biological constraints on gene structure.
pub struct OrganismConstraints {
    pub min_exon_len: usize,
    pub max_exon_len: usize,
    pub min_intron_len: usize,
    pub max_intron_len: usize,
    pub allow_gc_donor: bool,
}

impl OrganismConstraints {
    pub fn for_organism(organism: &str) -> Self {
        match organism {
            "vertebrate" => Self {
                min_exon_len: 3,
                max_exon_len: 11_000,
                min_intron_len: 60,
                max_intron_len: 500_000,
                allow_gc_donor: true,
            },
            "insect" => Self {
                min_exon_len: 3,
                max_exon_len: 5_000,
                min_intron_len: 40,
                max_intron_len: 100_000,
                allow_gc_donor: false,
            },
            "plant" => Self {
                min_exon_len: 3,
                max_exon_len: 5_000,
                min_intron_len: 60,
                max_intron_len: 10_000,
                allow_gc_donor: true,
            },
            "fungus" => Self {
                min_exon_len: 3,
                max_exon_len: 3_000,
                min_intron_len: 40,
                max_intron_len: 5_000,
                allow_gc_donor: false,
            },
            _ => Self::for_organism("vertebrate"),
        }
    }

    pub fn valid_intron_length(&self, len: usize) -> bool {
        len >= self.min_intron_len && len <= self.max_intron_len
    }

    pub fn valid_exon_length(&self, len: usize) -> bool {
        len >= self.min_exon_len && len <= self.max_exon_len
    }

    /// Check GT-AG (canonical) or GC-AG (non-canonical) splice site in sequence.
    pub fn valid_donor(&self, seq: &[u8], pos: usize) -> bool {
        if pos + 2 > seq.len() {
            return false;
        }
        let di = &seq[pos..pos + 2];
        matches!(di, b"GT" | b"gt")
            || (self.allow_gc_donor && matches!(di, b"GC" | b"gc"))
    }

    pub fn valid_acceptor(&self, seq: &[u8], pos: usize) -> bool {
        if pos < 2 {
            return false;
        }
        let di = &seq[pos - 2..pos];
        matches!(di, b"AG" | b"ag")
    }

    /// Start codon ATG occupying [pos, pos+3); pos is the first coding base.
    pub fn valid_start(&self, seq: &[u8], pos: usize) -> bool {
        if pos + 3 > seq.len() {
            return false;
        }
        matches!(&seq[pos..pos + 3], b"ATG" | b"atg")
    }

    /// Stop codon occupying [end-3, end); end is the exclusive exon end (first
    /// intronic base), so the stop codon is the last three coding bases.
    pub fn valid_stop(&self, seq: &[u8], end: usize) -> bool {
        if end < 3 {
            return false;
        }
        matches!(
            &seq[end - 3..end],
            b"TAA" | b"TAG" | b"TGA" | b"taa" | b"tag" | b"tga"
        )
    }
}
