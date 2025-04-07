use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod ops;
mod model;








#[pyfunction]
fn relu(x: f32) -> PyResult<f32> {
    Ok(ops::activation::relu(x))
}


#[pyfunction]
fn sigmoid(x: f32) -> PyResult<f32> {
    Ok(ops::activation::sigmoid(x))
}


#[pyfunction]
fn dot_matrice(a:Vec<Vec<f32>>, b:Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    Ok(ops::linear_alg::dot_matrice(a,b))
}
   
#[pyfunction]
fn cosh(x:f32) ->PyResult<f32>{
    Ok(ops::activation::cosh(x))
}


#[pyfunction]
fn tanh(x:f32) ->PyResult<f32>{
    Ok(ops::activation::tanh(x))
}




#[pymodule]
fn smithy(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(relu,m)?)?; 
    m.add_function(wrap_pyfunction!(sigmoid,m)?)?;
    m.add_function(wrap_pyfunction!(cosh,m)?)?;
    m.add_function(wrap_pyfunction!(tanh,m)?)?;
    m.add_function(wrap_pyfunction!(dot_matrice,m)?)?;
    Ok(())
}