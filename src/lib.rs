use pyo3::prelude::*;

mod constraints;
mod graph;
mod viterbi;

pub use graph::{Dag, ExonNode};
pub use viterbi::GeneModel;

#[pyfunction]
#[pyo3(signature = (
    donor_scores,
    acceptor_scores,
    start_scores,
    stop_scores,
    coding_scores,
    homology_scores = None,
    organism = "vertebrate",
    threshold = 0.1,
))]
fn build_dag(
    donor_scores: Vec<f32>,
    acceptor_scores: Vec<f32>,
    start_scores: Vec<f32>,
    stop_scores: Vec<f32>,
    coding_scores: Vec<Vec<f32>>, // (seq_len, 3)
    homology_scores: Option<Vec<f32>>,
    organism: &str,
    threshold: f32,
) -> PyResult<graph::PyDag> {
    let constraints = constraints::OrganismConstraints::for_organism(organism);
    let dag = graph::build(
        &donor_scores,
        &acceptor_scores,
        &start_scores,
        &stop_scores,
        &coding_scores,
        homology_scores.as_deref(),
        &constraints,
        threshold,
    );
    Ok(graph::PyDag(dag))
}

#[pyfunction]
fn viterbi_decode(dag: &graph::PyDag) -> PyResult<Vec<viterbi::PyGeneModel>> {
    let models = viterbi::decode(&dag.0);
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
