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
    })
}