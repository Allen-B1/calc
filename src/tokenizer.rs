use std::str::Chars;

use lazy_static::lazy_static;
lazy_static! {
    static ref BUILTINS: Vec<&'static str> = vec![
        "sin",
        "cos",
        "tan",
        "sec",
        "csc",
        "cot",
        "arctan",
        "arcsin",
        "arccos",
        "arcsec",
        "arccsc",
        "arccot",

        "rref"
    ];
}

pub struct Tokenizer<'a> {
    chars: Chars<'a>,
}

impl<'a> Tokenizer<'a> {
    pub fn new(input: &'a str) -> Self {
        return Tokenizer { chars: input.chars() }
    }

    fn parse_ident(&mut self, first: char) -> Token {
        let is_long = first == '#';
        let mut ident_name = first.to_string();

        let chars: Chars = self.chars.as_str().chars();
        let mut len = 0; // number of characters added, not including the first one
        for char in chars {
            if !char.is_alphabetic() {
                break
            }

            ident_name.push(char);
            len += 1;
        }

        if is_long || BUILTINS.iter().find(|&&x|  ident_name == x).is_some() {
            if len != 0 {
                self.chars.nth(len-1);
            }
            Token::Ident(ident_name)
        } else {
            Token::Ident(first.to_string())
        }
    }

    fn parse_num(&mut self, first: char) -> Token {
        let mut out = first.to_string();
        let mut has_dot = false;
        loop {
            let next = self.chars.as_str().chars().next(); // peek
            match next {
                Some(c @ '0'..='9') => out.push(c),
                Some('.') => {
                    if !has_dot {
                        has_dot=  true;
                        out.push('.');
                    } else {
                        break
                    }
                }
                _ => break
            }

            self.chars.next();
        }

        let num = out.parse::<f64>().unwrap();
        Token::Num(num)
    }
}

pub type Error = String;

impl<'a> Iterator for Tokenizer<'a>  {
    type Item = Result<Token, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        Some(Ok(match self.chars.next() {
            None => return None,
            Some(c @ ('+' |  '-' | '*' | '/' | '.')) => Token::Op(c),
            Some('[') => Token::LeftBracket,
            Some(']') => Token::RightBracket,
            Some('(') => Token::LeftParen,
            Some(')') => Token::RightParen,
            Some(';')  => Token::Semicolon,
            Some(',') => Token::Comma,
            Some('|') => Token::Bar,
            Some('=') => Token::Equal,
            Some(' ' | '\t' | '\n') => return self.next(),
            Some(c @ 'A'..='Z' | c @ 'a'..='z') => self.parse_ident(c),
            Some('#') => self.parse_ident('#'),
            Some(c @ '0'..='9') => self.parse_num(c),
            Some(c) => return Some(Err(format!("unexpected character {}", c)))
        }))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Num(f64),
    
    Op(char),
    LeftBracket,
    RightBracket,
    LeftParen,
    RightParen,
    Semicolon,
    Equal,

    Bar,
    Comma
}

#[test]
fn test_tokenize() {
    let mut tokenizer = Tokenizer::new("[1 (2x + 3)]");
    assert_eq!(tokenizer.next().unwrap().unwrap(), Token::LeftBracket);
    assert_eq!(tokenizer.next().unwrap().unwrap(), Token::Num(1.0));   
    assert_eq!(tokenizer.next().unwrap().unwrap(), Token::LeftParen);
    assert_eq!(tokenizer.next().unwrap().unwrap(), Token::Num(2.0));   
    assert_eq!(tokenizer.next().unwrap().unwrap(), Token::Ident("x".to_string()));   

    let mut tok = Tokenizer::new("#xaxaxa bc sin co");
    dbg!(tok.collect::<Vec<_>>());
}