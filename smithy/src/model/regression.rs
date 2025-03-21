extern crate ndarray;
extern crate ndarray_linalg; 

use ndarray::{Array2, Array1, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_linalg::Inverse;
use pyo3::prelude::*;
use ndarray_rand::RandomExt;



#[pyclass]
pub struct linear_regression {
    pub methode_resolution:i32,
    pub weight:Option<Array2<f64>>,
}



impl linear_regression{

    pub fn new(resolution:i32)-> linear_regression{
        linear_regression{
            methode_resolution: resolution,
            weight:None,
        }
    }

    pub fn fit(&mut self, x_train:&Array2<f32>, y_train:&Array2<f32>, learning_rate:f32, epochs:usize)-> Array2<f64>{
        let (n_samples,n_featues) = x_train.dim();
        let mut weight  =  Array2::random((n_featues,1), Uniform::new(0.0, 1.0));
        let mut result = x_train.t().dot(x_train);
        if (self.methode_resolution == -1){
            weight = result.pinv().dot(x_train.t().dot(y_train))
        }
        self.weight = weight;
        return self.weight;
    }

    pub fn predict(&self,x_test:&Array2){
        return x_test.dot(self.weight);
    }

}