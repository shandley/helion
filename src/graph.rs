use pyo3::prelude::*;

use crate::constraints::OrganismConstraints;

#[derive(Clone, Debug)]
pub struct ExonNode {
    pub start: usize,   // 0-based genomic
    pub end: usize,
    pub frame: u8,      // 0, 1, 2
    pub score: f32,
    pub homology: f32,
}

/// A directed acyclic graph of candidate exon nodes.
/// Edges represent valid intron transitions (stored implicitly via adjacency list).
pub struct Dag {
    pub nodes: Vec<ExonNode>,
    /// adjacency[i] = list of node indices that can follow node i
    pub adjacency: Vec<Vec<usize>>,
}

#[pyclass]
pub struct PyDag(pub Dag);

#[allow(clippy::too_many_arguments)]
pub fn build(
    donor_scores: &[f32],
    acceptor_scores: &[f32],
    start_scores: &[f32],
    stop_scores: &[f32],
    coding_scores: &[Vec<f32>],
    intergenic_scores: Option<&[f32]>,
    homology_scores: Option<&[f32]>,
    constraints: &OrganismConstraints,
    threshold: f32,
) -> Dag {
    let seq_len = donor_scores.len();

    // Candidate exon start positions (strong acceptor or start codon)
    let starts: Vec<usize> = (0..seq_len)
        .filter(|&i| acceptor_scores[i] > threshold || start_scores[i] > threshold)
        .collect();

    // Candidate exon end positions (strong donor or stop codon)
    let ends: Vec<usize> = (0..seq_len)
        .filter(|&i| donor_scores[i] > threshold || stop_scores[i] > threshold)
        .collect();

    let mut nodes: Vec<ExonNode> = Vec::new();

    for &s in &starts {
        // For each start, only consider ends that are within valid exon length range
        let min_end = s + constraints.min_exon_len;
        let max_end = s + constraints.max_exon_len;

        // Binary search for first end >= min_end
        let first = ends.partition_point(|&e| e < min_end);

        for &e in &ends[first..] {
            if e > max_end {
                break;
            }
            let exon_len = e - s;

            for frame in 0u8..3 {
                let coding_score = coding_scores[s..e]
                    .iter()
                    .map(|c| c[frame as usize])
                    .sum::<f32>()
                    / exon_len as f32;

                if coding_score < threshold {
                    continue;
                }

                let homology = homology_scores
                    .map(|h| h[s..e].iter().sum::<f32>() / exon_len as f32)
                    .unwrap_or(0.0);

                let mean_intergenic = intergenic_scores
                    .map(|ig| ig[s..e].iter().sum::<f32>() / exon_len as f32)
                    .unwrap_or(0.0);

                let score = donor_scores[e.saturating_sub(1)]
                    + acceptor_scores[s]
                    + coding_score
                    + homology
                    - mean_intergenic;

                nodes.push(ExonNode { start: s, end: e, frame, score, homology });
            }
        }
    }

    let n = nodes.len();
    let mut adjacency = vec![Vec::new(); n];

    if n == 0 {
        return Dag { nodes, adjacency };
    }

    // Sort node indices by start position for O(n log n) adjacency building
    let mut sorted_by_start: Vec<usize> = (0..n).collect();
    sorted_by_start.sort_by_key(|&i| nodes[i].start);

    for i in 0..n {
        let min_next_start = nodes[i].end + constraints.min_intron_len;
        let max_next_start = nodes[i].end + constraints.max_intron_len;

        // Binary search for the first sorted node whose start >= min_next_start
        let first = sorted_by_start.partition_point(|&j| nodes[j].start < min_next_start);

        for &j in &sorted_by_start[first..] {
            if nodes[j].start > max_next_start {
                break;
            }
            if i == j {
                continue;
            }
            // Frame compatibility: output frame of exon i must equal input frame of exon j
            let next_frame = (nodes[i].frame as usize + (nodes[i].end - nodes[i].start)) % 3;
            if next_frame == nodes[j].frame as usize {
                adjacency[i].push(j);
            }
        }
    }

    Dag { nodes, adjacency }
}
