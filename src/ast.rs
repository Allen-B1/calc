
#[derive(Debug)]
pub enum Node {
    Matrix(Matrix),
    Num(f64),
    Ident(String),
    Assign(String, Box<Node>),
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

    type TokenListRef<'a> = &'a [Token];
    type Error = String;

    /// A cursor is the index of the next token.
    /// To consume, increase the cursor by one.
    type Cursor = usize;
    fn get<'a>(r: TokenListRef<'a>, cur: Cursor) -> Option<&'a Token> {
        if cur >= r.len() {
            None
        } else {
            Some(&r[cur])
        }
    }

    /// Consumes the token matching the given token if present,
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
 
            let node = parse_atom(r, cur)?;
            current_row.push(node);
        }

        Ok(Matrix {
            entries: vec,
            width: size.unwrap_or(1),
            augmented: augmented.unwrap_or(false)
        })
    }

    /// Parses a number or identifier.
    pub fn parse_atom<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
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

    fn or(r: TokenListRef<'_>, cur: &mut Cursor, path1: impl FnOnce(TokenListRef<'_>, &mut Cursor) -> Result<Node, Error>, 
        path2: impl FnOnce(TokenListRef<'_>, &mut Cursor) -> Result<Node, Error>) -> Result<Node, Error> {
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

    fn parse_expr<'a>(r: TokenListRef<'a>, cur: &mut Cursor) -> Result<Node, Error> {
        or (
            r, cur,
            |r, cur| parse_matrix(r, cur).map(Node::Matrix),
            |r, cur| parse_atom(r, cur)
        )
    }

    /// parses an assignment; returns error if not assignment
    pub fn parse_assign<'a>(r: TokenListRef<'_>, cur: &mut Cursor) -> Result<Node, Error> {
        let ident_tok = expect_pred(r, cur, |t| match t {
            Token::Ident(_) => true,
            _ => false
        })?;

        expect(r, cur, Token::Equal)?;
        
        let expr_node = parse_expr(r, cur)?;        
        let ident = match ident_tok {
            Token::Ident(s) => s,
            _ => unreachable!()
        };

        Ok(Node::Assign(ident.clone(), Box::new(expr_node)))
    }

    pub fn parse<'a>(r: TokenListRef<'_>, cur: &mut Cursor) -> Result<Node, Error> {
        or(
            r, cur,
            parse_assign,
            parse_expr
        )
    }

    #[test]
    fn test() {
        let tokens: Vec<_> = Tokenizer::new("[a b c ; d e f]").collect();
        dbg!(&tokens);
        let mut cur = 0;
        let matrix = parse_matrix(&tokens, &mut cur).unwrap();
        dbg!(&matrix);

        let tokens: Vec<_> = Tokenizer::new("A = [1 2 | d ; 3 4 | a; f f | f]").collect();
        let mut cur = 0;
        let node = parse(&tokens, &mut cur).unwrap();
        dbg!(&node);

        let tokens: Vec<_> = Tokenizer::new("A").collect();
        let mut cur = 0;
        let node = parse(&tokens, &mut cur).unwrap();
        dbg!(&node);

    }
}