use serde::Serialize;

use super::num::Num;

#[derive(Serialize, Clone)]
pub struct Matrix {
    pub entries: Vec<Num>,
    pub width: usize,
    pub augmented: bool,
}

impl Matrix {
   
}