module {
  func.func @main() -> f32 {
    %a = arith.constant 1.0 : f32
    %b = arith.constant 2.0 : f32
    %c = arith.addf %a, %b : f32
    return %c : f32
  }
}



////Expected XLS IR:////

// package _package
// file_number 0 "xls/contrib/mlir/testdata/samples/addf.mlir"
// top fn main() -> (bits[1], bits[8], bits[23]) {
//   literal.1: (bits[1], bits[8], bits[23]) = literal(value=(0, 127, 0), id=1, pos=[(0,3,10)])
//   literal.2: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0), id=2, pos=[(0,4,10)])
//   ret literal.3: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 4194304), id=3, pos=[(0,5,10)])
// }