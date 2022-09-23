use std::collections::HashMap;
use serde::{Serialize, Serializer};

use crate::{ast::Node, tokenizer::Tokenizer};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "lowercase")]
pub enum Value {
    Undefined,
    Num(f64),
    Matrix(Matrix),
}

#[derive(Debug, Clone, Serialize)]
pub struct Matrix {
    pub entries: Vec<f64>,
    pub width: usize,
    pub augmented: bool,
}

impl Matrix {
    pub fn height(&self) -> usize {
        return self.entries.len() / self.width
    }
}

pub type SymbolTable = HashMap<String, Value>;

pub fn eval(node: &Node, symbols: &mut SymbolTable) -> Result<Value, String> {
    Ok(match node {
        Node::Matrix(matrix) => {
            let mut entries = Vec::new();
            for node in matrix.entries.iter() {
                let x = match eval(node, symbols)? {
                    Value::Num(x) => x,
                    _ => Err("non-number as matrix entry".to_string())?  
                };
                entries.push(x);
            }

            Value::Matrix(Matrix {
                entries,
                width: matrix.width,
                augmented: matrix.augmented
            })
        },
        Node::Num(x) => Value::Num(*x),
        Node::Ident(x) => symbols.get(x).ok_or_else(|| format!("variable {} not defined", x)).map(|x| x.clone())?,
        Node::Assign(ident, value) => {
            if symbols.contains_key(ident) {
                Err(format!("variable {} is already assigned", ident))?
            } else {
                let res = eval(value, symbols)?;
                symbols.insert(ident.clone(), res.clone());
                res
            }
        },
        Node::UnaryOp(op, inner) => {
            let inner = eval(inner, symbols)?;
            match inner {
                Value::Matrix(mut m) => {
                    match op {
                        '+' => Value::Matrix(m),
                        '-' => {
                            for entry in m.entries.iter_mut() {
                                *entry *= -1.0;
                            }
                            Value::Matrix(m)
                        },
                        _ => Err(format!("cannot apply unary {} to matrix", op))?
                    }
                },
                Value::Num(n) => {
                    match op {
                        '+' => Value::Num(n),
                        '-' => Value::Num(-n),
                        _ => Err(format!("cannot apply unary {} to real", op))?
                    }
                },
                Value::Undefined => Value::Undefined
            }   
        }
        Node::BinaryOp(lhs, op, rhs) => {
            let lhs = eval(lhs, symbols)?;
            let rhs = eval(rhs, symbols)?;
            match (lhs, rhs) {
                (Value::Num(a), Value::Num(b)) => {
                    Value::Num(match *op {
                        '+' => a + b,
                        '-' => a - b,
                        '*' => a * b,
                        '/' => if b == 0.0 { return Ok(Value::Undefined) } else { a / b },
                        _ => Err(format!("cannot {} two numbers", *op))?
                    })
                }
                (Value::Num(a), Value::Matrix(mut v)) => {
                    if *op != '*' {
                        Err(format!("cannot {} number and matrix", *op))?
                    }
                    for entry in v.entries.iter_mut() {
                        *entry *= a
                    }
                    Value::Matrix(v)
                },
                (Value::Matrix(mut v), Value::Matrix(w)) => {
                    match *op {
                        '*' => {
                            if v.width == w.height() {
                                let mut m = Matrix {
                                    entries: vec![0.0; v.height() * w.width],
                                    width: w.width,
                                    augmented: false,
                                };
                                for i in 0..v.height() {
                                    for j in 0..w.width {
                                        let mut sum = 0.0;
                                        for k in 0..v.width {
                                            sum += v.entries[i * v.width + k] * w.entries[k * w.width + j];
                                        }
                                        m.entries[i * w.width + j] = sum;
                                    }
                                }
                                Value::Matrix(m)
                            } else if v.width == 1 && w.width == 1 && v.height() == 3 && w.height() == 3 {
                                let m1 = v.entries[1] * w.entries[2] - v.entries[2] * w.entries[1];
                                let m2 = v.entries[2] * w.entries[3] - v.entries[3] * w.entries[2];
                                let m3 = v.entries[0] * w.entries[1] - v.entries[1] * w.entries[0];
                                Value::Matrix(Matrix{
                                    entries: vec![m1, m2, m3],
                                    width: 1,
                                    augmented: false
                                })
                            } else {
                                Err(format!("cannot multiply {}x{} with {}x{} matrix", v.height(), v.width, w.height(), w.width))?
                            }
                        },
                        '+' => {
                            if v.width != w.width || v.height() != w.height() {
                                Err(format!("cannot add {}x{} with {}x{} matrix", v.height(), v.width, w.height(), w.width))?
                            }
                            
                            for (idx, entry) in v.entries.iter_mut().enumerate() {
                                *entry += w.entries[idx]
                            }

                            Value::Matrix(v)
                        },
                        '.' => {
                            if v.width != 1 || w.width != 1 || v.height() != w.height() {
                                Err(format!("cannot dot {}x{} with {}x{} matrix", v.height(), v.width, w.height(), w.width))?
                            }
                            
                            let mut sum = 0.0;
                            for i in 0..v.height() {
                                sum += v.entries[i] * w.entries[i];
                            }
                            Value::Num(sum)
                        }
                        _ =>  Err(format!("cannot {} matrix and matrix", *op))?
                    }
                },
                (Value::Undefined, _) => Value::Undefined,
                (_, Value::Undefined) => Value::Undefined,
                (lhs, rhs) => Err(format!("cannot {} lhs {:?} rhs {:?}", *op, lhs, rhs))?
            }
        },
    })
}