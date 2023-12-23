# calc
Web-based graphing calculator with support for vector-valued functions and other features.

Currently supported objects:
* (Real, Complex) Numbers
* Matrices
* Functions
   * Functions `R -> R` and `[0, 1] -> R^2` can be graphed.
Planned:
* Vector/scalar fields
* Arithemetic on fields with nonzero characteristic

This project is very much incomplete.

![Screenshot](screenshot.png)

## Getting Started
```bash
git clone https://github.com/allen-B1/calc.git && cd calc
cargo build
python3 -m http.server
```
Then open up `localhost:8000/app.html` (or whatever port Python outputs) in your web browser.
