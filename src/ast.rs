
#[derive(Debug)]
pub enum Node {
    Matrix(Matrix),
    Num(f64),
    Ident(String),
    Assign(String, Vec<String>, Box<Node>),
    BinaryOp(Box<Node>, char, Box<Node>),
    UnaryOp(char, Box<Node>)
}

#[derive(Debug)]
pub struct Matrix {
    pub entries: Vec<Node>,
    pub width: usize,
    pub augmented: bool,
}

impl Matrix {
    pub fn height(&self) -> usize {
        return self.entries.len() / self.width
    }
}

pub mod parser {
    use crate::tokenizer::{Tokenizer, Token};
    use std::iter::Peekable;
    use super::*;

    pub type TokenListRef<'a> = &'a [Token];
    pub type Error = String;

    /// A cursor is the index of the next token.
    /// To consume, increase the cursor by one.
    pub type Cursor = usize;
    fn get<'a>(r: TokenListRef<'a>, cur: Cursor) -> Option<&'a Token> {
        if cur >= r.len() {
            None
        } else {
            Some(&r[cur])
        }
    }

    /// Consumes the token matching the given token if and only if it is present,
    /// otherwise errors
    fn expect<'a>(r: TokenListRef<'a>, cur: &mut Cursor, token: Token) -> Result<&'a Token, Error> {
        let tok = get(r, *cur);
        if tok.is_none() {
            return Err(format!("expected {:?}, found EOF", token));
        }
        let tok = tok.unwrap();
        if !token.eq(tok) {
            return Err(format!("expected {:?}, found {:?}", token, tok));
        }
        *cur += 1;
        Ok(tok)
    }
    fn expect_pred<'a>(r: TokenListRef<'a>, cur: &mut Cursor, pred: impl FnOnce(&Token) -> bool) -> Result<&'a Token, Error> {
        let tok = get(r, *cur);
        if tok.is_none() {
            return Err("expected pattern, found EOF".to_string());
        }
        let tok = tok.unwrap();
        if !pred(tok) {
            return Err(format!("expected pattern, found {:?}", tok));
        }
        *cur += 1;
        Ok(tok)
    }

    fn or<T>(r: TokenListRef<'_>, cur: &mut Cursor, path1: impl FnOnce(TokenListRef<'_>, &mut Cursor) -> Result<T, Error>, 
    path2: impl FnOnce(TokenListRef<'_>, &mut Cursor) -> Result<T, Error>) -> Result<T, Error> {
        let mut cur1 = *cur;
        let mut cur2 = *cur;
        let err1 = match path1(r, &mut cur1) {
            Ok(n) => { *cur = cur1; return Ok(n) }
            Err(e) => e
        };
        let err2 = match path2(r, &mut cur2) {
            Ok(n) => { *cur = cur2; return Ok(n) }
            Err(e) => e
        };
        Err(format!("tried two paths [{} | {}]", err1, err2))
    }


    pub fn parse_matrix<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Matrix, Error> {
        expect(r, cur, Token::LeftBracket)?;

        let mut vec = Vec::new();
        let mut current_row = Vec::new();
        let mut size: Option<usize> = None;
        // none if no bar, true if the bar was the last one
        let mut bar_is_recent: Option<bool> = None;
        let mut augmented: Option<bool> = None;
        loop {
            let terminator = expect_pred(r, cur, |t| match t {
                Token::Semicolon | Token::RightBracket => true,
                _ => false
            });
            if terminator.is_ok() {
                let terminator = terminator.unwrap();
                if size.is_some() && size.unwrap() != current_row.len() {
                    return Err(format!("matrix has inconsistent row size: expected {}, got {}", size.unwrap(), current_row.len()));
                }
                size = Some(current_row.len());
 
                // check that bar <=> augmented
                if bar_is_recent == None && augmented == Some(true) {
                    return Err(format!("Bar is missing in augmented matrix"));
                } else if bar_is_recent.is_some() && augmented == Some(false) {
                    return Err(format!("Bar in non-augmented matrix"));                    
                }

                // if no bar, set augmented to false
                if augmented == None {
                    if bar_is_recent == None {
                        augmented = Some(false)
                    } else {
                        augmented = Some(true)
                    }
                }

                // reset bar
                bar_is_recent = None;

                vec.append(&mut current_row);

                if terminator.eq(&Token::RightBracket) {
                    break
                }
            } else if !(current_row.len() == 0 && vec.len() == 0) {
                expect(r, cur, Token::Comma)?;
            }

            // check that the bar is in the right place
            // and set bar information
            if expect(r, cur, Token::Bar).is_ok() {
                bar_is_recent = Some(true);
                continue
            } else {
                // if recent_bar == Some(false), that means there was a bar, then an entry
                if bar_is_recent == Some(false) {
                    return Err(format!("Bar is in the wrong place"));
                }
                // map Some(true) => Some(false)
                bar_is_recent = bar_is_recent.map(|_| false);
            }
 
            let node = parse_expr(r, cur)?;
            current_row.push(node);
        }

        Ok(Matrix {
            entries: vec,
            width: size.unwrap_or(1),
            augmented: augmented.unwrap_or(false)
        })
    }

    /// Parses binary add & higher precedence operators
    fn parse_add<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        let lhs = parse_mul(r, cur)?;
        
        let op = match expect_pred(r, cur, |t| match t {
            Token::Op('+' | '-') => true,
            _ => false
        }) {
            Err(_) => return Ok(lhs),
            Ok(Token::Op(c)) => *c,
            _ => unreachable!()
        };
        
        let rhs = parse_add(r, cur)?;

        Ok( Node::BinaryOp(Box::new(lhs), op, Box::new(rhs)))
    }

    /// Parses binary mul & higher precedence operators
    fn parse_mul<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        let lhs = parse_unary(r, cur)?;
        
        let op = match expect_pred(r, cur,  |t| match t {
            Token::Op('*' | '/' | '.') => true,
            _ => false
        }) {
            Err(_) => return Ok(lhs),
            Ok(Token::Op(c)) => *c,
            _ => unreachable!()
        };
        
        let rhs = parse_mul(r, cur)?;

        Ok( Node::BinaryOp(Box::new(lhs), op, Box::new(rhs)))
    }

    fn parse_unary<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        Ok(match expect_pred(r, cur, |token| match token {
            Token::Op('+' | '-') => true,
            _ => false
        }) {
            Err(_) => parse_mulcat(r, cur)?,
            Ok(Token::Op(op)) => Node::UnaryOp(*op, Box::new(parse_mulcat(r, cur)?)),
            _ => unreachable!()
        })
    }

    fn parse_mulcat<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        let lhs = parse_group(r, cur)?;

        let rhs = or(
            r,cur,
            |r, cur| parse_group(r, cur).map(Some),
            |_, _| Ok(None)
        )?;

        Ok(match rhs {
            Some(rhs) => Node::BinaryOp(Box::new(lhs), '*', Box::new(rhs)),
            None => lhs
        })
    }

    /// Parses groups and atoms
    fn parse_group<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        if expect(r, cur, Token::LeftParen).is_err() {
            return parse_atom(r, cur);
        }

        let node = parse_expr(r, cur)?;

        expect(r, cur, Token::RightParen)?;

        Ok(node)
    }

    /// Parses a number, identifier, or matrix.
    fn parse_atom<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        or (
            r, cur,
            |r, cur| parse_matrix(r, cur).map(Node::Matrix),
            |r, cur| {
                let tok = expect_pred(r, cur, |t| match t {
                    Token::Ident(_) | Token::Num(_) => true,
                    _ => false
                })?;
        
                Ok(match tok {
                    Token::Num(n) => Node::Num(*n),
                    Token::Ident(s) => Node::Ident(s.clone()),
                    _ => unreachable!()
                })
            }
        )
    }

    fn parse_expr<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        parse_add(r, cur)
    }

    /// parses an assignment; returns error if not assignment
    fn parse_assign<'a>(r: TokenListRef<'_>, cur: &mut Cursor) -> Result<Node, Error> {
        let ident_tok = expect_pred(r, cur, |t| match t {
            Token::Ident(_) => true,
            _ => false
        })?;

        let mut params = Vec::new();
        if expect(r, cur, Token::LeftParen).is_ok() {
            loop {
                let tok = expect_pred(r, cur, |t| match t {
                    Token::Ident(_) => true,
                    _ => false
                })?;
                let param = match tok {
                    Token::Ident(s) => s,
                    _ => unreachable!()
                };

                params.push(param.clone());
                
                let terminator = expect_pred(r, cur, |t| match t {
                    Token::Comma | Token::RightParen => true,
                    _ => false
                })?;
                match terminator {
                    Token::Comma => continue,
                    Token::RightParen => break,
                    _ => unreachable!()
                }
            }
        };

        expect(r, cur, Token::Equal)?;
        
        let expr_node = parse_expr(r, cur)?;        
        let ident = match ident_tok {
            Token::Ident(s) => s,
            _ => unreachable!()
        };

        Ok(Node::Assign(ident.clone(), params, Box::new(expr_node)))
    }

    pub fn parse<'a>(r: TokenListRef<'_>, cur: &mut Cursor) -> Result<Node, Error> {
        or(
            r, cur,
            parse_assign,
            parse_expr
        )
    }


    #[cfg(test)]
    mod tests {
        use crate::{tokenizer::{Tokenizer, Token}, ast::parser::{parse_matrix, parse}};

        use super::{TokenListRef, Cursor};

        fn create_toklist(src: &str) -> (Vec<Token>, Cursor) {
            (Tokenizer::new(src).map(Result::unwrap).collect(), 0)
        }

        #[test]
        fn test() {
            let (tokens, mut cur) = create_toklist("[a, b, c; d, e, f]");
            let matrix = parse_matrix(&tokens, &mut cur).unwrap();
    
            let (tokens, mut cur) = create_toklist("5 - 3 + a * 5 + 22b/3cd");
            dbg!(parse(&tokens, &mut cur).unwrap());

            let (tokens, mut cur) = create_toklist("v*w");
            dbg!(parse(&tokens, &mut cur).unwrap());

            let (tokens, mut cur) = create_toklist("M = [1, 5 ; 2,  4 ; 3, 5]");
            dbg!(parse(&tokens, &mut cur).unwrap());
        }
    }
}