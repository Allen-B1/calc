use std::iter::Peekable;

use serde_json::json;
use wasm_bindgen::{prelude::*, JsCast};

use crate::{eval::{SymbolTable, Value, SymbolExprTable}, ast::Node};

mod tokenizer;
mod ast;
mod eval;

/// Evaluates a JSON-encoded array of expressions,
/// and returns an array of values or errors.
#[wasm_bindgen]
pub fn eval(input: String) -> String {
    fn inner(input: Vec<String>) -> Vec<Result<Value, String>> {
        // execute assignments first
        // TODO: more advanced dependency graph
        let mut assignments = Vec::new();
        let mut exprs = Vec::new();
        let mut errors = Vec::new();

        for (idx, str) in input.iter().enumerate() {
            let tokens: Result<Vec<_>, _> = tokenizer::Tokenizer::new(str).collect();
            let node = tokens.and_then(|tokens| ast::parser::parse(&tokens, &mut 0)); 
            match node {
                Ok(node @ Node::Assign(_, _, _)) => assignments.push((idx, node)),
                Ok(node) => exprs.push((idx, node)),
                Err(err) => errors.push((idx, err))
            }
        }

        let mut expr_table = SymbolExprTable::new();
        let mut table = SymbolTable::new();
        let mut values = vec![Err("".to_string()); input.len()];
        for (idx, node) in assignments.iter() {
            let expr = eval::create_expr(node, &mut expr_table, &mut |_, _|{});
            let ident = match node {
                Node::Assign(ident, _, _) => ident,
                _ => unreachable!()
            };
            let value = expr.and_then(|expr| eval::eval_expr(&expr, &table));
            if value.is_ok() {
                table.insert(ident.clone(), value.clone().unwrap());
            }
            values[*idx] = value;
        }
        for (idx, node) in exprs.iter() {
            let expr = eval::create_expr(node, &mut expr_table, &mut |_, _|{});
            values[*idx] = expr.and_then(|expr| eval::eval_expr(&expr, &mut table));
        }
        for (idx, err) in errors {
            values[idx] = Err(err);
        }
        values
    }

    let vec = inner(match serde_json::from_str(&input) {
        Ok(v) => v,
        Err(e) => return format!("{:?}", e)
    });
    let mut new_value = Vec::new();
    for item in vec {
        match item {
            Ok(v) => new_value.push(serde_json::value::to_value(v).unwrap_or(serde_json::Value::Null)),
            Err(err) => new_value.push(json!({"error": err}))
        }
    }

    match serde_json::to_string(&new_value) {
        Ok(v) => v,
        Err(e) => return format!("{:?}", e)
    }
}

#[wasm_bindgen]
pub fn get_ident(expr: String) -> Option<String> {
    let tokens: Result<Vec<_>, _> = tokenizer::Tokenizer::new(&expr).collect();
    let node = tokens.and_then(|tokens| ast::parser::parse(&tokens, &mut 0));
    
    match node {
        Ok(Node::Assign(name, _, _)) => Some(name),
        _ => None
    }
}