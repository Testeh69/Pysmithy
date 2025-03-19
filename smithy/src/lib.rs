use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
mod ops;



#[pyfunction]
fn relu(x: f64) -> PyResult<f64> {
    Ok(ops::activation::relu(x))
}


#[pyfunction]
fn sigmoid(x: f64) -> PyResult<f64> {
    Ok(ops::activation::sigmoid(x))
}


#[pyfunction]
fn dot_matrice(a:Vec<Vec<f64>>, b:Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
    Ok(ops::linear::dot_matrice(a,b))
}
   
#[pyfunction]
fn cosh(x:f64) ->PyResult<f64>{
    Ok(ops::activation::cosh(x))
}


#[pyfunction]
fn tanh(x:f64) ->PyResult<f64>{
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