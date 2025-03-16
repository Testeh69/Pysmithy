use rayon::prelude::*;
use core::panic;

pub fn dot_matrice(a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
   
    let a_line_len = a.len();
    let a_col_len = a[0].len();       
    let b_line_len = b.len();          
    let b_col_len = b[0].len();   
    if a_col_len != b_line_len {
        panic!("Incompatible dimensions: a's columns must equal b's rows.");
    }
    
    let mut matrice = vec![vec![0.0; b_col_len]; a_line_len];

    matrice.par_iter_mut().enumerate().for_each(|(i, row)| {
        for col in 0..b_col_len { 
            let mut result = 0.0;
            for k in 0..a_col_len {  
                result += a[line][k] * b[k][col]; 
            }
            matrice[line][col] = result; 
        }
    });
    
    matrice  
}