use std::ops::{Add, Mul, Div};

use num::{complex::{Complex64, ComplexFloat}, BigInt, ToPrimitive, Integer, Signed};
use serde::Serialize;

use super::{Value, Any};

#[derive(Serialize, Clone, PartialEq)]
pub enum Num { 
    Complex(Complex64),
    Real(f64),
    Int(BigInt)
}

impl Add for Num {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a + b),
            (Num::Real(a), Num::Int(b)) | (Num::Int(b), Num::Real(a)) => {
                match b.to_f64() {
                    Some(b) => Num::Real(a+b),
                    None => Num::Real(f64::NAN)
                }
            },
            (Num::Real(a), Num::Real(b)) => Num::Real(a+b),
            (Num::Int(a), Num::Complex(b)) | (Num::Complex(b), Num::Int(a)) => {
                match a.to_f64() {
                    Some(a) => Num::Complex(b + a),
                    None => Num::Real(f64::NAN)
                }
            },
            (Num::Real(a), Num::Complex(b)) | (Num::Complex(b), Num::Real(a)) => Num::Complex(a+b),
            (Num::Complex(a), Num::Complex(b)) => Num::Complex(a+b)
        }
    }
}

impl Mul for Num {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => Num::Int(a * b),
            (Num::Real(a), Num::Int(b)) | (Num::Int(b), Num::Real(a)) => {
                match b.to_f64() {
                    Some(b) => Num::Real(a * b),
                    None => Num::Real(f64::NAN)
                }
            },
            (Num::Real(a), Num::Real(b)) => Num::Real(a * b),
            (Num::Int(a), Num::Complex(b)) | (Num::Complex(b), Num::Int(a)) => {
                match a.to_f64() {
                    Some(a) => Num::Complex(b * a),
                    None => Num::Real(f64::NAN)
                }
            },
            (Num::Real(a), Num::Complex(b)) | (Num::Complex(b), Num::Real(a)) => Num::Complex(a * b),
            (Num::Complex(a), Num::Complex(b)) => Num::Complex(a * b)
        }
    }
}

impl Div for Num {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (Num::Int(a), Num::Int(b)) => {
                if a.is_multiple_of(&b) {
                    Num::Int(a/b)
                } else {
                    Num::Real(a.to_f64().unwrap_or(f64::NAN) / b.to_f64().unwrap_or(f64::NAN))
                }
            },
            (Num::Int(a), Num::Real(b)) => Num::Real(a.to_f64().unwrap_or(f64::NAN) / b),
            (Num::Real(a), Num::Int(b)) => Num::Real(a / b.to_f64().unwrap_or(f64::NAN)),
            (Num::Real(a), Num::Real(b)) => Num::Real(a / b),

            (Num::Int(a), Num::Complex(b))  => Num::Complex(a.to_f64().unwrap_or(f64::NAN) / b),
            (Num::Complex(a), Num::Int(b)) => Num::Complex(a / b.to_f64().unwrap_or(f64::NAN)),
            (Num::Real(a), Num::Complex(b)) => Num::Complex(a/b),
            (Num::Complex(a), Num::Real(b)) => Num::Complex(a/b),
            (Num::Complex(a), Num::Complex(b)) => Num::Complex(a/ b)
        }
    }   
}


impl Value for Num {
    fn abs(&self) -> Option<Any> {
        Some(Any::Num(match self {
            Num::Int(a) => Num::Int(a.abs().into()),
            Num::Real(a) => Num::Real(a.abs()),
            Num::Complex(a) => Num::Real(a.abs())
        }))
    }

    fn add(&self, rhs: super::Any) -> Option<super::Any> {
        match rhs {
            Any::Num(n) => Some(Any::Num(self.clone()+n)),
            _ => None
        }
    }

    fn mul(&self, rhs: super::Any) -> Option<super::Any> {
        match rhs {
            Any::Num(n) => Some(Any::Num(self.clone()*n)),
            _ => None
        }
    }

    fn div(&self, rhs: super::Any) -> Option<super::Any> {
        match rhs {
            Any::Num(n) => Some(Any::Num(self.clone()/n)),
            _ => None
        }
    }
}