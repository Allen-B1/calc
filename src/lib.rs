use std::{iter::Peekable, collections::{HashMap, HashSet}};

use serde_json::json;
use wasm_bindgen::{prelude::*, JsCast};

use crate::{eval::{SymbolTable, Value, SymbolExprTable, eval_expr}, ast::Node};

mod tokenizer;
mod ast;
mod eval;
mod value;

/// Evaluates a JSON-encoded array of expressions,
/// and returns an array of values or errors.
#[wasm_bindgen]
pub fn eval(input: String) -> String {
    fn inner(input: Vec<String>) -> Vec<Result<Value, String>> {
        // execute assignments first
        // TODO: more advanced dependency graph
        let mut assignments = HashMap::new();

        let mut exprs = Vec::new();
        let mut errors = Vec::new();

        for (idx, str) in input.iter().enumerate() {
            let tokens: Result<Vec<_>, _> = tokenizer::Tokenizer::new(str).collect();
            let node = tokens.and_then(|tokens| ast::parser::parse(&tokens, &mut 0)); 
            match node {
                Ok(Node::Assign(ident, b, c)) => {assignments.insert(ident.clone(), (idx, Node::Assign(ident, b, c)));},
                Ok(node) => exprs.push((idx, node)),
                Err(err) => errors.push((idx, err))
            }
        }

        let mut ordering = Vec::new();

        let mut mark_perm = HashSet::new();
        let mut mark_temp = HashSet::new();

        fn visit(ident:&str, node: &Node, assignments: &HashMap<String, (usize, Node)>, ordering: &mut Vec<String>, mark_perm: &mut HashSet<String>, mark_temp: &mut HashSet<String>) {
            if mark_perm.contains(ident) {
                return;
            }
            if mark_temp.contains(ident) {
                ordering.push(ident.to_owned());
                return;
            }

            mark_temp.insert(ident.to_owned());

            let mut idents = HashSet::new();
            node.idents(&mut idents);
            for dep in idents {
                if assignments.contains_key(&dep) {
                    visit(&dep, &assignments[&dep].1, assignments, ordering, mark_perm, mark_temp);
                }
            }

            mark_temp.remove(ident);
            mark_perm.insert(ident.to_owned());
            ordering.push(ident.to_owned());
        }

        loop {
            for (ident, (idx, node)) in assignments.iter() {
                if !mark_temp.contains(ident) && !mark_perm.contains(ident) {
                    visit(&ident, &node, &assignments, &mut ordering, &mut mark_perm, &mut mark_temp);
                    break;
                }
            }

            if mark_temp.len() + mark_perm.len() >= assignments.len() {
                break
            }
        };

        let mut exprtable = SymbolExprTable::new();
        let mut symbols = SymbolTable::new();
        let mut values = vec![None; input.len()];
        for ident in ordering {
            let idx =assignments[&ident].0;
            let expr = eval::create_expr(&assignments[&ident].1, &mut exprtable, &mut |_, _| {});
            let value = expr.and_then(|expr| eval::eval_expr(&expr, &symbols));
            if let Ok(value) = &value {
                symbols.insert(ident, value.clone());
            }
            values[idx] = Some(value);
        }
        for (idx, node) in exprs {
            let expr = eval::create_expr(&node, &mut exprtable, &mut |_, _| {});
            let value = expr.and_then(|expr| eval::eval_expr(&expr, &symbols));
            values[idx] = Some(value);
        }

        values.into_iter().map(|x| x.unwrap_or(Err(format!("internal error 2042")))).collect()
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

#[wasm_bindgen]
pub fn eval_func(value: String, inputs: String) -> String {
    fn inner(func: Value, inputs: Vec<Vec<Value>>) -> Result<Vec<Value>, String> {
        let (params, expr) = match func {
            Value::Func(params, expr) => (params, *expr.clone()),
            _ => Err(format!("value is not a function"))?
        };

        let mut outputs = Vec::new();
        for inputset in inputs {
            if params.len() != inputset.len() {
                Err(format!("expected {} arguments, got {} arguments", params.len(), inputset.len()))?
            }
    
            let mut symbols = SymbolTable::new();
            for (i, input) in inputset.into_iter().enumerate() {
                symbols.insert(params[i].clone(), input);
            }
            outputs.push(eval_expr(&expr, &symbols)?);
        }

        Ok(outputs)
    }

    let vec = inner(
        match serde_json::from_str(&value) {
            Ok(e) => e,
            Err(e) => return format!("{:?}", e)
        },
        match serde_json::from_str(&inputs) {
            Ok(e) => e,
            Err(e) => return format!("{:?}", e)
        }
    );

    match vec {
        Err(e) => format!("{}", e),
        Ok(v) => serde_json::to_string(&v).unwrap_or_else(|e| e.to_string())
    }
}