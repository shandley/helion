use pyo3::prelude::*;
use std::collections::HashSet;

use crate::graph::{Dag, ExonNode};

#[pyclass]
#[derive(Clone)]
pub struct PyExon {
    #[pyo3(get)]
    pub start: usize,
    #[pyo3(get)]
    pub end: usize,
    #[pyo3(get)]
    pub frame: u8,
    #[pyo3(get)]
    pub score: f32,
}

#[pyclass]
pub struct PyGeneModel(pub GeneModel);

#[pymethods]
impl PyGeneModel {
    #[getter]
    fn seqid(&self) -> &str {
        &self.0.seqid
    }
    #[getter]
    fn start(&self) -> usize {
        self.0.start
    }
    #[getter]
    fn end(&self) -> usize {
        self.0.end
    }
    #[getter]
    fn strand(&self) -> &str {
        &self.0.strand
    }
    #[getter]
    fn score(&self) -> f32 {
        self.0.score
    }
    #[getter]
    fn exons(&self) -> Vec<PyExon> {
        self.0.exons.iter().map(|e| PyExon {
            start: e.start,
            end: e.end,
            frame: e.frame,
            score: e.score,
        }).collect()
    }

    fn with_offset(&self, offset: usize) -> Self {
        let mut m = self.0.clone();
        m.start += offset;
        m.end += offset;
        for exon in &mut m.exons {
            exon.start += offset;
            exon.end += offset;
        }
        PyGeneModel(m)
    }

    fn with_seqid(&self, seqid: String) -> Self {
        let mut m = self.0.clone();
        m.seqid = seqid;
        PyGeneModel(m)
    }

    /// Flip coordinates from RC-window space back to genomic sense-strand space.
    ///
    /// RC exon [s, e) maps to genomic [window_len - e, window_len - s).
    /// Exon order is reversed so exons remain sorted by genomic start.
    fn with_rc_flip(&self, window_len: usize) -> Self {
        let mut m = self.0.clone();
        m.start = window_len - self.0.end;
        m.end = window_len - self.0.start;
        let n = m.exons.len();
        for i in 0..n {
            let old_start = self.0.exons[i].start;
            let old_end = self.0.exons[i].end;
            m.exons[i].start = window_len - old_end;
            m.exons[i].end = window_len - old_start;
        }
        m.exons.reverse();
        PyGeneModel(m)
    }
}

#[derive(Clone)]
pub struct GeneModel {
    pub seqid: String,
    pub start: usize,
    pub end: usize,
    pub strand: String,
    pub score: f32,
    pub exons: Vec<ExonNode>,
}

/// Run one pass of Viterbi DP over the exon DAG, skipping excluded nodes.
/// Returns the best path as a vec of node indices, plus the path score.
/// Returns None if no valid path exists (all nodes excluded or no positive score).
fn best_path(
    dag: &Dag,
    excluded: &HashSet<usize>,
    order: &[usize],
) -> Option<(Vec<usize>, f32)> {
    let n = dag.nodes.len();
    let mut dp = vec![f32::NEG_INFINITY; n];
    let mut prev = vec![usize::MAX; n];

    for &i in order {
        if excluded.contains(&i) {
            continue;
        }
        dp[i] = dag.nodes[i].score;
    }

    for &i in order {
        if excluded.contains(&i) {
            continue;
        }
        for &j in &dag.adjacency[i] {
            if excluded.contains(&j) {
                continue;
            }
            let candidate = dp[i] + dag.nodes[j].score;
            if candidate > dp[j] {
                dp[j] = candidate;
                prev[j] = i;
            }
        }
    }

    // Best terminal: highest dp value among non-excluded nodes
    let best_end = order
        .iter()
        .filter(|&&i| !excluded.contains(&i) && dp[i] > 0.0)
        .copied()
        .max_by(|&a, &b| dp[a].partial_cmp(&dp[b]).unwrap_or(std::cmp::Ordering::Equal));

    let end_node = best_end?;

    let mut path = vec![end_node];
    let mut cur = end_node;
    while prev[cur] != usize::MAX {
        cur = prev[cur];
        path.push(cur);
    }
    path.reverse();

    Some((path, dp[end_node]))
}

/// Decode multiple non-overlapping gene models from the exon DAG.
///
/// Iteratively finds the best-scoring path, adds it as a gene model, then
/// excludes all nodes that overlap the called gene's genomic span before
/// searching for the next model. Stops when no remaining path scores above 0
/// or when the model cap is reached.
pub fn decode(dag: &Dag, strand: &str) -> Vec<GeneModel> {
    let n = dag.nodes.len();
    if n == 0 {
        return vec![];
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| dag.nodes[i].start);

    let mut excluded: HashSet<usize> = HashSet::new();
    let mut models: Vec<GeneModel> = Vec::new();
    const MAX_MODELS: usize = 50;

    while models.len() < MAX_MODELS {
        let Some((path, score)) = best_path(dag, &excluded, &order) else {
            break;
        };

        let exons: Vec<ExonNode> = path.iter().map(|&i| dag.nodes[i].clone()).collect();
        let model_start = exons.first().map(|e| e.start).unwrap_or(0);
        let model_end = exons.last().map(|e| e.end).unwrap_or(0);

        models.push(GeneModel {
            seqid: String::new(), // set by Python via with_seqid()
            start: model_start,
            end: model_end,
            strand: strand.to_string(),
            score,
            exons,
        });

        // Exclude all nodes whose genomic span overlaps this model
        for i in 0..n {
            if dag.nodes[i].start < model_end && dag.nodes[i].end > model_start {
                excluded.insert(i);
            }
        }
    }

    models
}
