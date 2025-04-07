



pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp()) 
}

pub fn cosh(x: f32) -> f32 {
    x.cosh() 
}

pub fn tanh(x:f32) -> f32 {
    x.tanh()
}