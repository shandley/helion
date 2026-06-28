use pyo3::prelude::*;

mod constraints;
mod graph;
mod viterbi;

pub use graph::{Dag, ExonNode};
pub use viterbi::GeneModel;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    donor_scores,
    acceptor_scores,
    start_scores,
    stop_scores,
    coding_scores,
    intergenic_scores = None,
    homology_scores = None,
    sequence = None,
    organism = "vertebrate",
    threshold = 0.1,
    boundary_contrast = 0.0,
    splice_weight = 1.0,
    length_penalty = 0.0,
))]
fn build_dag(
    donor_scores: Vec<f32>,
    acceptor_scores: Vec<f32>,
    start_scores: Vec<f32>,
    stop_scores: Vec<f32>,
    coding_scores: Vec<Vec<f32>>, // (seq_len, 3)
    intergenic_scores: Option<Vec<f32>>,
    homology_scores: Option<Vec<f32>>,
    sequence: Option<String>, // sense-oriented window sequence for consensus checks
    organism: &str,
    threshold: f32,
    boundary_contrast: f32,
    splice_weight: f32,
    length_penalty: f32,
) -> PyResult<graph::PyDag> {
    let constraints = constraints::OrganismConstraints::for_organism(organism);
    let seq_bytes = sequence.as_ref().map(|s| s.as_bytes());
    let dag = graph::build(
        &donor_scores,
        &acceptor_scores,
        &start_scores,
        &stop_scores,
        &coding_scores,
        intergenic_scores.as_deref(),
        homology_scores.as_deref(),
        seq_bytes,
        &constraints,
        threshold,
        boundary_contrast,
        splice_weight,
        length_penalty,
    );
    Ok(graph::PyDag(dag))
}

#[pyfunction]
#[pyo3(signature = (dag, strand = "+"))]
fn viterbi_decode(dag: &graph::PyDag, strand: &str) -> PyResult<Vec<viterbi::PyGeneModel>> {
    let models = viterbi::decode(&dag.0, strand);
    Ok(models.into_iter().map(viterbi::PyGeneModel).collect())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_dag, m)?)?;
    m.add_function(wrap_pyfunction!(viterbi_decode, m)?)?;
    m.add_class::<graph::PyDag>()?;
    m.add_class::<viterbi::PyGeneModel>()?;
    m.add_class::<viterbi::PyExon>()?;
    Ok(())
}
