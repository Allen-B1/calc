use std::{collections::{HashMap, HashSet}, fmt::Display};
use num::{Zero, complex::Complex64};
use serde::{Serialize, Serializer, ser::{SerializeTuple, SerializeSeq}, Deserialize};

use crate::{ast::Node, tokenizer::Tokenizer};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

impl Class {
    pub fn combine(&self, rhs: &Class, op: char) -> Class {
        let lhs = self;
        match (&lhs, &rhs) {
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
            (Class::Func(params2, out2), Class::Func(params1, out1)) => {
                match op {
                    '.' => {
                        if params2.len() == 1 && params2[0] == **out1 {
                            Class::Func(params1.clone(), out2.clone())
                        } else {
                            Class::Any
                        }     
                    },
                    '+' | '*' => {
                        if params2.len() == params1.len() {
                            Class::Func(
                                params2.iter().zip(params1.iter()).map(|(a, b)| if a == b { a.clone() } else { Class::Any }).collect(),
                                Box::new(out2.combine(out1, op))
                            )
                        } else {
                            Class::Any
                        }
                    }
                    _ => Class::Any
                }
            },  
            _ => Class::Any
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Copy, PartialEq, Eq)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "lowercase")]
pub enum Value {
    Undefined,
    Num(Complex64),
    Matrix(Matrix),
    Func(Vec<String>, Box<Expr>)
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
                Class::Func(vec![Class::Any; params.len()], Box::new(Class::Any))
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Matrix {
    pub entries: Vec<Complex64>,
    pub width: usize,
    pub augmented: bool,
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
        if !(self.width == 1 && rhs.width == 1 && self.height() == 3 && rhs.height() == 3) {
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
        if self.width != rhs.height() {
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
        
        let m = Matrix {
            entries: self.entries.iter().enumerate().map(|(i, v)| rhs.entries[i] + *v).collect(),
            width: self.width,
            augmented: false
        };

        Some(m)
    }
}

impl Expr {
    pub fn new_add(lhs: Expr, rhs: Expr) -> Expr {
        Expr::Add(Box::new(lhs), Box::new(rhs))
    }
    pub fn new_mul(lhs: Expr, rhs: Expr) -> Expr {
        Expr::Mul(Box::new(lhs), Box::new(rhs))
    }
    pub fn new_sub(lhs: Expr, rhs: Expr) -> Expr {
        Self::new_add(lhs, 
            Self::new_mul(Expr::Num(-1.0), rhs))
    }
    pub fn new_div(lhs: Expr, rhs: Expr) -> Expr {
        Expr::Div(Box::new(lhs), Box::new(rhs))
    }
    pub fn new_dot(lhs: Expr, rhs: Expr) -> Expr {
        Expr::Dot(Box::new(lhs), Box::new(rhs))
    }

    pub fn opaque(class: Class) -> Expr {
        Expr::Num(5346.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Expr {
    Ident(String),
    Num(f64),
    Matrix {
        entries: Vec<Expr>,
        width: usize,
        augmented: bool,
    },
    Call(Box<Expr>, Vec<Expr>),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Dot(Box<Expr>, Box<Expr>),
    Func(Vec<String>, Box<Expr>)
}

pub type SymbolTable = HashMap<String, Value>;
pub type Error = String;

pub fn eval_expr(expr: &Expr, symbols: &SymbolTable) -> Result<Value, Error> {
    Ok(match &expr {
        Expr::Ident(s) => symbols.get(s).ok_or(format!("variable {} is not defined", s))?.clone(),
        Expr::Num(n) => Value::Num(Complex64::new(*n, 0.0)),
        Expr::Call(lhs, params) => {
            let lhs = eval_expr(lhs, symbols)?;
            let (param_names, expr) = match lhs {
                Value::Func(params, expr) => (params, expr),
                _ => Err(format!("cannot call a non-function of type {}", lhs.class()))?
            };
            
            let mut new_symbols = symbols.clone();
            for (i, param) in params.iter().enumerate() {
                new_symbols.insert(param_names[i].clone(), eval_expr(param, symbols)?);
            }
            eval_expr(&expr, &new_symbols)?
        },
        Expr::Add(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;
            
            match (lhs, rhs) {
                (Value::Matrix(lhs), Value::Matrix(rhs)) => {
                    Value::Matrix(lhs.add(&rhs).ok_or(format!("can't add {}x{} and {}x{} matrices", lhs.height(), lhs.width, rhs.height(), rhs.width))?)
                },
                (Value::Num(lhs), Value::Num(rhs)) => {
                    Value::Num(lhs + rhs)
                },
                (Value::Func(params1, expr1), Value::Func(params2, expr2)) => {
                    if params1.len() != params2.len() {
                        Err(format!("cannot add functions with {} and {} parameters", params1.len(), params2.len()))?
                    }
                    Value::Func(params1.clone(),
                        Box::new(Expr::new_add( *expr1, *expr2))
                    )
                },
                (lhs, rhs) => Err(format!("cannot add {} and {}", lhs.class(), rhs.class()))?
            }
        },
        Expr::Mul(a, b) => {
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
                (Value::Func(params1, expr1), Value::Func(params2, expr2)) => {
                    if params1.len() != params2.len() {
                        Err(format!("cannot multiply functions with {} and {} parameters", params1.len(), params2.len()))?
                    }
                    Value::Func(params1.clone(),
                        Box::new(Expr::new_mul(*expr1, *expr2))
                    )
                },
                (lhs, rhs) => Err(format!("cannot multiply {} and {}", lhs.class(), rhs.class()))?
            }
        },
        Expr::Div(a, b) => {
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
        Expr::Dot(a, b) => {
            let lhs = eval_expr(a, symbols)?;
            let rhs = eval_expr(b, symbols)?;

            match (lhs, rhs) {
                (Value::Matrix(lhs), Value::Matrix(rhs)) => {
                    Value::Num(lhs.dot(&rhs).ok_or(format!("can't multiply {}x{} and {}x{} matrices", lhs.height(), lhs.width, rhs.height(), rhs.width))?)
                },
                (Value::Func(params2, expr2), Value::Func(params1, expr1)) => {
                    if params2.len() != 1 {
                        Err(format!("cannot compose function with multivariable input"))?;
                    }

                    Value::Func(params1.clone(), Box::new(Expr::Call(
                            // f
                            Box::new(Expr::Func(
                                params2.iter().map(|x| x.clone()).collect(),
                                expr2,
                            )),
                            // input to f
                            vec![Expr::Call(
                                    // g
                                Box::new(Expr::Func(
                                        params1.iter().map(|x| x.clone()).collect(),
                                        expr1,
                                    )
                                ),
                                params1.iter()
                                    .map(|x| Expr::Ident(x.clone()))
                                    .collect()
                            )]
                        )
                    ))
                },
                (lhs, rhs) => Err(format!("cannot dot {} and {}", lhs.class(), rhs.class()))?
            }
        },
        Expr::Matrix { entries, width, augmented } => {
            let entries: Vec<Complex64> = entries.iter().map(|entry| match eval_expr(entry, symbols)? {
                Value::Num(n) => Ok(n),
                value => Err(format!("matrix cannot contain non-number entries: {}", value.class()))
            }).collect::<Result<_,_>>()?;
            Value::Matrix(Matrix {
                entries,
                width: *width, augmented: *augmented
            })
        },
        Expr::Func(params, output) => {
            Value::Func(params.clone(), output.clone())
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

            let expr = if params.len() !=0 { 
                let mut func_symbols = symbols.clone();
                let mut param_classes = Vec::new();
                for param in params.iter() {
                    func_symbols.insert(param.clone(), Expr::opaque(Class::Any));
                    param_classes.push(Class::Any);
                }

                let inner_expr = create_expr(rhs, &mut func_symbols, undefined_variable)?;
                Expr::Func(params.clone(), Box::new(inner_expr))
            } else if ident.eq("y") && !symbols.contains_key("x") {
                let mut func_symbols = symbols.clone();
                func_symbols.insert("x".to_string(), Expr::opaque(Class::Num(NumClass::Real)));
                let inner_expr = create_expr(rhs, &mut func_symbols, undefined_variable)?;
                Expr::Func(vec!["x".to_owned()], Box::new(inner_expr))
            } else {
                create_expr(rhs, symbols, undefined_variable)?
            };

            symbols.insert(ident.clone(), expr.clone());
            expr
        },
        Node::BinaryOp(lhs, op, rhs) => {
            let lhs = create_expr(lhs, symbols, undefined_variable)?;
            let rhs = create_expr(rhs, symbols, undefined_variable)?;

            match op {
                '+' => Expr::new_add(lhs, rhs),
                '-' => Expr::new_sub(lhs, rhs),
                '*' => Expr::new_mul(lhs, rhs),
                '/' => Expr::new_div(lhs, rhs),
                '.' => Expr::new_dot(lhs, rhs),
                _ => Err(format!("unknown operation: {}", op))?
            }
        },
        Node::UnaryOp(op, inner) => {
            let inner = create_expr(inner, symbols, undefined_variable)?;
            match  op {
                '+' => Expr::new_mul(Expr::Num(1.0), inner),
                '-' => Expr::new_mul(Expr::Num(-1.0), inner),
                _ => Err(format!("unknown unary operation: {}", *op))?,
            }
        }
        Node::Ident(ident) => {
            let expr = symbols.get(ident);
            if expr.is_none() {
                undefined_variable(ident.as_str(), symbols);
            }
            Expr::Ident(ident.clone())
        }
        Node::Matrix(mdata) => {
            let entries: Vec<Expr> = mdata.entries.iter().map(|node| create_expr(node, symbols, undefined_variable)).collect::<Result<_, _>>()?;
            Expr::Matrix {
                entries,
                width: mdata.width,
                augmented: mdata.augmented
            }
        },
        Node::Num(n) => {
            Expr::Num(*n)
        },
        Node::Call(lhs, params) => {
            let lhs = create_expr(lhs, symbols, undefined_variable)?;
            let mut param_exprs = Vec::new();
            for param in params {
                param_exprs.push(create_expr(param, symbols, undefined_variable)?);
            }
            Expr::Call(Box::new(lhs), param_exprs)
        }
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