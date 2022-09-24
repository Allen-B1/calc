use std::{collections::HashMap, fmt::Display};
use num::{Zero, complex::Complex64};
use serde::{Serialize, Serializer, ser::{SerializeTuple, SerializeSeq}};

use crate::{ast::Node, tokenizer::Tokenizer};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum Class {
    Any,
    Num(NumClass),
    Vector(NumClass, usize),
    Matrix(NumClass, usize, usize),
    Func(Vec<Class>, Box<Class>)
}

impl Display for Class {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "todo!()")
    }
}

#[derive(Debug, Clone, Serialize, Copy, PartialEq, Eq)]
pub enum NumClass {
    Int,
    Real,
    Complex
}

impl NumClass {
    /** Returns the result type when you combine two numbers */
    pub fn combine(self, b: NumClass) -> NumClass {
        match (self, b) {
            (NumClass::Int, NumClass::Int) => NumClass::Int,
            (NumClass::Real | NumClass::Int, NumClass::Real | NumClass::Int) => NumClass::Real,
            (_, _) => NumClass::Complex
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "lowercase")]
pub enum Value {
    Undefined,
    #[serde(serialize_with="serialize_num")]
    Num(Complex64),
    Matrix(Matrix),
    Func(Vec<(String, Class)>, Box<Expr>)
}

impl Value {
    pub fn class(&self) -> Class {
        match self {
            Value::Undefined => Class::Any,
            Value::Num(c) => {
                if c.im != 0.0 {
                    Class::Num(NumClass::Complex)
                } else if c.re as i64 as f64 == c.re { 
                    Class::Num(NumClass::Int)
                } else {
                    Class::Num(NumClass::Real)
                }
            },
            Value::Matrix(c) => {
                let class = if c.entries.iter().all(|n| n.im == 0.0) {
                    NumClass::Real
                } else {
                    NumClass::Complex
                };

                if c.width == 1 {
                    Class::Vector(class, c.height())
                } else {
                    Class::Matrix(class, c.height(), c.width)
                }
            },
            Value::Func(params, expr) => {
                Class::Func(params.iter().map(|x| x.1.clone()).collect(), Box::new(expr.class.clone()))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Matrix {
    #[serde(serialize_with="serialize_matrix")]
    pub entries: Vec<Complex64>,
    pub width: usize,
    pub augmented: bool,
}

fn serialize_num<S: Serializer>(num: &Complex64, s: S) -> Result<S::Ok, S::Error> {
    let mut t = s.serialize_tuple(2)?;
    t.serialize_element(&num.re)?;
    t.serialize_element(&num.im)?;
    t.end()
}

fn serialize_matrix<S: Serializer>(entries: &Vec<Complex64>, s: S) -> Result<S::Ok, S::Error> {
    let mut seq = s.serialize_seq(Some(entries.len()))?;
    for num in entries.iter() {
        seq.serialize_element(&(num.re, num.im))?;
    }
    seq.end()
}

impl Matrix {
    pub fn height(&self) -> usize {
        return self.entries.len() / self.width
    }

    pub fn dot(&self, rhs: &Matrix) -> Option<Complex64> {
        if self.width != 1 || rhs.width != 1 || self.height() != rhs.height() {
            return None
        }
        
        let mut sum = Complex64::zero();
        for i in 0..self.height() {
            sum += self.entries[i] * rhs.entries[i].conj();
        }
        Some(sum)
    }

    pub fn cross(&self, rhs: &Matrix) -> Option<Matrix> {
        if self.width == 1 && rhs.width == 1 && self.height() == 3 && rhs.height() == 3 {
            return None
        }
        if !self.entries.iter().all(|x| x.im == 0.0) || !rhs.entries.iter().all(|x| x.im == 0.0) {
            return None
        }

        let (a1, a2, a3, b1, b2, b3) = (
            self.entries[0].re, self.entries[1].re, self.entries[2].re, rhs.entries[0].re, rhs.entries[1].re, rhs.entries[2].re);

        let m1 = a2*b3 - a3*b2;
        let m2 = a3*b1 - a1*b3;
        let m3 = a1*b2 - a2*b1;
        Some(Matrix{
            entries: vec![Complex64::new(m1, 0.0), Complex64::new(m2, 0.0), Complex64::new(m3, 0.0)],
            width: 1,
            augmented: false
        })
    }

    pub fn mul(&self, rhs: &Matrix) -> Option<Matrix> {
        if self.width == rhs.height() {
            return None
        }

        // matrix multiplication
        let mut m = Matrix {
            entries: vec![Complex64::zero(); self.height() * rhs.width],
            width: rhs.width,
            augmented: false,
        };
        for i in 0..self.height() {
            for j in 0..rhs.width {
                let mut sum = Complex64::zero();
                for k in 0..self.width {
                    sum += self.entries[i * self.width + k] * rhs.entries[k * rhs.width + j];
                }
                m.entries[i * rhs.width + j] = sum;
            }
        }
        Some(m)
    }

    pub fn scalar_mul(&self, scalar: Complex64) -> Matrix {
        let mut clone = self.clone();
        for entry in clone.entries.iter_mut() {
            *entry *= scalar;
        }
        clone
    }

    pub fn add(&self, rhs: &Matrix) -> Option<Matrix> {
        if self.width != rhs.width || self.height() != rhs.height() {
            return None;
        }
        
        let mut m = Matrix {
            entries: self.entries.iter().enumerate().map(|(i, v)| rhs.entries[i] + *v).collect(),
            width: self.width,
            augmented: false
        };

        Some(m)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Expr {
    class: Class,
    data: ExprData,
}

impl Expr {
    pub fn new_add(class: Class, lhs: Expr, rhs: Expr) -> Expr {
        Expr { class, data: ExprData::Add(Box::new(lhs), Box::new(rhs))}
    }
    pub fn new_mul(class: Class, lhs: Expr, rhs: Expr) -> Expr {
        Expr { class, data: ExprData::Mul(Box::new(lhs), Box::new(rhs))}
    }
    pub fn new_sub(class: Class, scalar_class: NumClass, lhs: Expr, rhs: Expr) -> Expr {
        Self::new_add(class.clone(), lhs, 
            Self::new_mul(class, Expr { class: Class::Num(scalar_class), data: ExprData::Num(-1.0) }, rhs))
    }
    pub fn new_div(class: Class, lhs: Expr, rhs: Expr) -> Expr {
        Expr { class, data: ExprData::Div(Box::new(lhs), Box::new(rhs)) }
    }
    pub fn new_dot(class: Class, lhs: Expr, rhs: Expr) -> Expr {
        Expr { class, data: ExprData::Dot(Box::new(lhs), Box::new(rhs)) }
    }

    pub fn opaque(class: Class) -> Expr {
        Expr { class, data: ExprData::Num(5346.0) }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", content = "data_data")]
pub enum ExprData {
    Ident(String),
    Num(f64),
    Matrix {
        entries: Vec<Expr>,
        width: usize,
        augmented: bool,
    },
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Dot(Box<Expr>, Box<Expr>),
    Func(Vec<String>, Box<Expr>)
}

pub type SymbolTable = HashMap<String, Value>;
pub type Error = String;

pub fn eval_expr(expr: &Expr, symbols: &SymbolTable) -> Result<Value, Error> {
    Ok(match &expr.data {
        ExprData::Ident(s) => symbols.get(s).ok_or(format!("variable {} is not defined", s))?.clone(),
        ExprData::Num(n) => Value::Num(Complex64::new(*n, 0.0)),
        ExprData::Add(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;
            
            match (lhs, rhs) {
                (Value::Matrix(lhs), Value::Matrix(rhs)) => {
                    Value::Matrix(lhs.add(&rhs).ok_or(format!("can't add {}x{} and {}x{} matrices", lhs.height(), lhs.width, rhs.height(), rhs.width))?)
                },
                (Value::Num(lhs), Value::Num(rhs)) => {
                    Value::Num(lhs + rhs)
                },
                (lhs, rhs) => Err(format!("cannot add {} and {}", lhs.class(), rhs.class()))?
            }
        },
        ExprData::Mul(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;

            match (lhs, rhs) {
                (Value::Matrix(lhs), Value::Matrix(rhs)) => {
                    if let Some(cross) = lhs.cross(&rhs) {
                        Value::Matrix(cross)
                    } else {
                        Value::Matrix(lhs.mul(&rhs).ok_or(format!("can't multiply {}x{} and {}x{} matrices", lhs.height(), lhs.width, rhs.height(), rhs.width))?)
                    }
                },
                (Value::Num(lhs), Value::Num(rhs)) => {
                    Value::Num(lhs*rhs)
                },
                (Value::Num(scalar), Value::Matrix(vector)) => {
                    Value::Matrix(vector.scalar_mul(scalar))
                },
                (lhs, rhs) => Err(format!("cannot multiply {} and {}", lhs.class(), rhs.class()))?
            }
        },
        ExprData::Div(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;

            match (lhs, rhs) {
                (Value::Num(lhs), Value::Num(rhs)) => {
                    if rhs.is_zero() {
                        Value::Undefined
                    } else {
                        Value::Num(lhs/rhs)
                    }
                },
                (Value::Matrix(lhs), Value::Num(rhs)) => {
                    if rhs.is_zero() {
                        Value::Undefined
                    } else {
                        Value::Matrix(lhs.scalar_mul(1.0/rhs))
                    }
                },
                (lhs, rhs) => Err(format!("cannot divide {} and {}", lhs.class(), rhs.class()))?
            }
        },
        ExprData::Dot(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;

            match (lhs, rhs) {
                (Value::Matrix(lhs), Value::Matrix(rhs)) => {
                    Value::Num(lhs.dot(&rhs).ok_or(format!("can't multiply {}x{} and {}x{} matrices", lhs.height(), lhs.width, rhs.height(), rhs.width))?)
                },
                (lhs, rhs) => Err(format!("cannot dot {} and {}", lhs.class(), rhs.class()))?
            }
        },
        ExprData::Matrix { entries, width, augmented } => {
            let entries: Vec<Complex64> = entries.iter().map(|entry| match eval_expr(entry, symbols)? {
                Value::Num(n) => Ok(n),
                _ => Err(format!("matrix cannot contain non-number entries: {:?}", entry))
            }).collect::<Result<_,_>>()?;
            Value::Matrix(Matrix {
                entries,
                width: *width, augmented: *augmented
            })
        },
        ExprData::Func(params, output) => {
            let param_classes = match &expr.class {
                Class::Func(params, _) => params,
                _ => unreachable!()
            };

            Value::Func(params.iter().map(Clone::clone).zip(param_classes.iter().map(Clone::clone)).collect(), output.clone())
        }
    })
}

pub type SymbolExprTable = HashMap<String, Expr>;

pub fn create_expr(node: &Node, symbols: &mut SymbolExprTable, undefined_variable: &mut impl FnMut(&str, &mut SymbolExprTable)) -> Result<Expr, Error> {
    Ok(match node {
        Node::Assign(ident, params, rhs) => {
            if symbols.contains_key(ident) {
                Err(format!("variable {} already defined", ident))?
            }

            let mut param_classes = Vec::new();
            let expr = if params.len() !=0 { 
                let mut func_symbols = symbols.clone();
                for param in params.iter() {
                    func_symbols.insert(param.clone(), Expr::opaque(Class::Any));
                    param_classes.push(Class::Any);
                }

                create_expr(rhs, &mut func_symbols, undefined_variable)?
            } else if ident == "y" && !symbols.contains_key("x") {
                let mut func_symbols = symbols.clone();
                func_symbols.insert("x".to_string(), Expr::opaque(Class::Num(NumClass::Real)));
                param_classes.push(Class::Num(NumClass::Real));
                create_expr(rhs, symbols, undefined_variable)?
            } else {
                create_expr(rhs, symbols, undefined_variable)?
            };

            symbols.insert(ident.clone(), expr.clone());
            if params.len() == 0 {
                expr
            } else {
                Expr {
                    class: Class::Func(param_classes, Box::new(expr.class.clone())),
                    data: ExprData::Func(params.clone(), Box::new(expr))
                }
            }
        },
        Node::BinaryOp(lhs, op, rhs) => {
            let lhs = create_expr(lhs, symbols, undefined_variable)?;
            let rhs = create_expr(rhs, symbols, undefined_variable)?;

            let class = match (&lhs.class, &rhs.class) {
                (Class::Num(a), Class::Num(b)) => {
                    let class = a.combine(*b);
                    Class::Num(class)
                },
                (Class::Num(a), Class::Matrix(n, row, col)) => {
                    let class = a.combine(*n);
                    Class::Matrix(class, *row, *col)
                },
                (Class::Matrix(a, arow, acol), Class::Matrix(b, brow, bcol)) => {
                    let class = a.combine(*b);

                    match op {
                        '+' => {
                            if arow == brow && acol == bcol {
                                Class::Matrix(class, *arow, *acol)
                            } else {
                                Class::Any
                            }
                        },
                        '*' => {
                            if acol == brow {
                                Class::Matrix(class, *arow, *bcol)
                            } else if *arow == 3 && *brow == 3&& *acol == 1 && *bcol == 1 {
                                Class::Matrix(class, 3, 1)
                            } else {
                                Class::Any
                            }
                        },
                        _ => Class::Any
                    }
                },
                _ => Class::Any
            };
            match op {
                '+' => Expr::new_add(class, lhs, rhs),
                '-' => Expr::new_sub(class, NumClass::Int, lhs, rhs),
                '*' => Expr::new_mul(class,lhs, rhs),
                '/' => Expr::new_div(class,lhs, rhs),
                '.' => Expr::new_dot(class,lhs, rhs),
                _ => Err(format!("unknown operation: {}", op))?
            }
        },
        Node::UnaryOp(op, inner) => {
            let inner = create_expr(inner, symbols, undefined_variable)?;
            let class = inner.class.clone();
            match  op {
                '+' => Expr::new_mul(class, Expr { class: Class::Num(NumClass::Int), data: ExprData::Num(1.0) }, inner),
                '-' => Expr::new_mul(class, Expr { class: Class::Num(NumClass::Int), data: ExprData::Num(-1.0) }, inner),
                _ => Err(format!("unknown unary operation: {}", *op))?,
            }
        }
        Node::Ident(ident) => {
            let expr = symbols.get(ident);
            if expr.is_none() {
                undefined_variable(ident.as_str(), symbols);
            }
            let expr = symbols.get(ident).ok_or(format!("variable {} not defined", ident))?;
            Expr { class: expr.class.clone(), data: ExprData::Ident(ident.clone()) }
        }
        Node::Matrix(mdata) => {
            let entries: Vec<Expr> = mdata.entries.iter().map(|node| create_expr(node, symbols, undefined_variable)).collect::<Result<_, _>>()?;
            let class = entries.iter().fold(Some(NumClass::Int), |class, entry| match (class, &entry.class) {
                (Some(nc1), Class::Num(nc2)) => Some(nc1.combine(*nc2)),
                _ => None
            });
            Expr { class: class.map(Class::Num).unwrap_or(Class::Any), data: ExprData::Matrix {
                entries,
                width: mdata.width,
                augmented: mdata.augmented
            } } 
        },
        Node::Num(n) => {
            let class = if *n as i64 as f64 == *n {
                Class::Num(NumClass::Int)
            } else {
                Class::Num(NumClass::Real)
            };

            Expr { class, data: ExprData::Num(*n) }
        },
    })
}
/*
#[cfg(test)]
mod tests {
    use crate::{ast::parser::{parse, Cursor}, tokenizer::*, eval::SymbolTable};

    use super::eval;

    fn create_toklist(src: &str) -> (Vec<Token>, Cursor) {
        (Tokenizer::new(src).map(Result::unwrap).collect(), 0)
    }

    #[test]
    fn test_matrix() {
        let (tokens, mut cur) = create_toklist("M = [1, 5 ; 2,  4 ; 3, 5]");
        let node= parse(&tokens, &mut cur).unwrap();
        let matrix = eval(&node, &mut SymbolTable::new()).unwrap();
        dbg!(matrix);
    }
}*/