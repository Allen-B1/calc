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

    fn parse_ident(&mut self, first: char) -> Option<Token> {
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
            Some(Token::Ident(ident_name))
        } else {
            Some(Token::Ident(first.to_string()))
        }
    }

    fn parse_num(&mut self, first: char) -> Option<Token> {
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
        Some(Token::Num(num))
    }
}

impl<'a> Iterator for Tokenizer<'a>  {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        match self.chars.next() {
            None => None,
            Some(c @ ('+' |  '-' | '*')) => Some(Token::Op(c)),
            Some('[') => Some(Token::LeftBracket),
            Some(']') => Some(Token::RightBracket),
            Some('(') => Some(Token::LeftParen),
            Some(')') => Some(Token::RightParen),
            Some(';')  => Some(Token::Semicolon),
            Some('|') => Some(Token::Bar),
            Some('=') => Some(Token::Equal),
            Some(' ' | '\t' | '\n') => self.next(),
            Some(c @ 'A'..='Z' | c @ 'a'..='z') => self.parse_ident(c),
            Some('#') => self.parse_ident('#'),
            Some(c @ '0'..='9') => self.parse_num(c),
            _ => None
        }
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

    Bar
}


#[test]
fn test_tokenize() {
    let mut tokenizer = Tokenizer::new("[1 (2x + 3)]");
    assert_eq!(tokenizer.next(), Some(Token::LeftBracket));
    assert_eq!(tokenizer.next(), Some(Token::Num(1.0)));   
    assert_eq!(tokenizer.next(), Some(Token::LeftParen));
    assert_eq!(tokenizer.next(), Some(Token::Num(2.0)));   
    assert_eq!(tokenizer.next(), Some(Token::Ident("x".to_string())));   

    let mut tok = Tokenizer::new("#xaxaxa bc sin co");
    dbg!(tok.collect::<Vec<_>>());
}