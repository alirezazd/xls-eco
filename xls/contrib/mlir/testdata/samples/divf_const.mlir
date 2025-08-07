// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

func.func @arith_divf_constant() -> f32 {
  %a = arith.constant 6.0 : f32
  %b = arith.constant 3.0 : f32
  %c = arith.divf %a, %b : f32
  return %c : f32

  // CHECK: top fn arith_divf_constant() -> (bits[1], bits[8], bits[23]) {
  // CHECK: ret literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0)
}


/////Expected XLS IR:////
// package _package
// 
// file_number 0 "xls/contrib/mlir/testdata/samples/divf_const.mlir"
// 
// top fn arith_divf_constant() -> (bits[1], bits[8], bits[23]) {
//   literal.1: (bits[1], bits[8], bits[23]) = literal(value=(0, 129, 4194304), id=1, pos=[(0,4,8)])
//   literal.2: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 4194304), id=2, pos=[(0,5,8)])
//   ret literal.3: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0), id=3, pos=[(0,6,8)])
// }
