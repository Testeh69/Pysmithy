use rayon::prelude::*;
use core::panic;
use ndarray::{Array2,Array1};
use rand::thread_rng;
use rand_distr::{Normal, Distribution};




#[derive(Debug, Clone)]
pub enum IngotType {
    Vector(Array1<f32>),
    Matrice(Array2<f32>),
}



pub trait OperationMatrice {
    fn dot(&self, other: &Self) -> Option<Self>
    where
        Self: Sized;
    
    fn transpose(&self) -> Self;

  
}


impl OperationMatrice for IngotType {
    fn dot(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (IngotType::Matrice(a), IngotType::Matrice(b)) => {
                if a.shape()[1] == b.shape()[0] {
                    Some(IngotType::Matrice(a.dot(b)))
                } else {
                    None
                }
            }
            (IngotType::Vector(a), IngotType::Vector(b)) => {
                if a.len() == b.len() {
                    Some(IngotType::Vector(Array1::from_elem(1, a.dot(b))))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn transpose(&self) -> Self {
        match self {
            IngotType::Matrice(a) => IngotType::Matrice(a.t().to_owned()),
            IngotType::Vector(a) => IngotType::Vector(a.clone()), // Un vecteur 1D reste le même
        }
    }

}



impl IngotType {

    fn iter_mut(&mut self) -> Box<dyn Iterator<Item = &mut f32> + '_> {
        match self {
            IngotType::Vector(v) => Box::new(v.iter_mut()),
            IngotType::Matrice(m) => Box::new(m.iter_mut()),
        }
    }

    pub fn generate_random_matrice(&mut self) {
      for i in self.iter_mut(){
        *i = generate_random_number();
      }
    }

    pub fn inverse(&self) -> Option<IngotType> {
        match self {
            IngotType::Matrice(m) => {
                let n = m.nrows();
                if m.ncols() != n {
                    return None; 
                }

                let mut augmented = m.clone();
                let mut identity = Array2::<f32>::eye(n);

                for i in 0..n {
                    let pivot = augmented[[i, i]];

                    if pivot == 0.0 {
                        return None; 
                    }

                    for j in 0..n {
                        augmented[[i, j]] /= pivot;
                        identity[[i, j]] /= pivot;
                    }

                    for k in 0..n {
                        if k != i {
                            let factor = augmented[[k, i]];
                            for j in 0..n {
                                augmented[[k, j]] -= factor * augmented[[i, j]];
                                identity[[k, j]] -= factor * identity[[i, j]];
                            }
                        }
                    }
                }

                Some(IngotType::Matrice(identity))
            }
            _ => None, 
        }
    }
}

pub fn generate_random_number()->f32{
    let mut rng = thread_rng();
    let normal : Normal<f32> = Normal::new(0.0, 0.3).unwrap();
    let mut x = normal.sample(&mut rng);
    while x < -1.0 || x > 1.0 {
        x = normal.sample(&mut rng);
    }
    return x

}

pub fn generate_random_matrice(array: Option<IngotType>) -> Option<IngotType> {
    array.map(|mut arr| {
        for i in arr.iter_mut() {
            *i = generate_random_number();
        }
        arr
    })
}





pub fn dot_matrice(a: Vec<Vec<f32>>, b: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
   
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
                result += a[i][k] * b[k][col]; 
            }
            row[col] = result; 
        }
    });
    
    matrice  
}