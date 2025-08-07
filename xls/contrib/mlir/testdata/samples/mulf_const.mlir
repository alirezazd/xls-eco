// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

func.func @arith_mulf_constant() -> f32 {
  %c1 = arith.constant 2.0 : f32
  %c2 = arith.constant 3.0 : f32
  %prod = arith.mulf %c1, %c2 : f32
  return %prod : f32

  // CHECK: top fn arith_mulf_constant() -> (bits[1], bits[8], bits[23]) {
  // CHECK: literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0)
  // CHECK: literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 4194304)
  // CHECK: ret literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 129, 2097152)
}


////Expected XLS IR:////
// package _package

// file_number 0 "xls/contrib/mlir/testdata/samples/mulf_const.mlir"

// top fn arith_mulf_constant() -> (bits[1], bits[8], bits[23]) {
//   literal.1: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0), id=1, pos=[(0,4,9)])
//   literal.2: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 4194304), id=2, pos=[(0,5,9)])
//   ret literal.3: (bits[1], bits[8], bits[23]) = literal(value=(0, 129, 4194304), id=3, pos=[(0,6,11)])
// }