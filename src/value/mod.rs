use std::ops::Add;

use serde::Serialize;

mod num;
mod matrix;
use matrix::Matrix;

pub trait Value: Serialize {
    fn abs(&self) -> Option<Any> { None }
    fn add(&self, rhs: Any) -> Option<Any> { None }
    fn mul(&self, rhs: Any) -> Option<Any> { None }
    fn div(&self, rhs: Any) -> Option<Any> { None }
    fn dot(&self, rhs: Any) -> Option<Any> { None }
}

pub enum Any {
    Num(num::Num),
    Matrix(Matrix)
}
