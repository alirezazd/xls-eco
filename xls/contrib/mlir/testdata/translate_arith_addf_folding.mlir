// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

// Test that arith.addf with constants gets properly constant-folded during translation
func.func @arith_addf_constant_folding() -> f32 {
  // These constants should be folded: 2.0 + 3.0 = 5.0
  %c1 = arith.constant 2.0 : f32
  %c2 = arith.constant 3.0 : f32
  %sum = arith.addf %c1, %c2 : f32
  return %sum : f32
  
  // CHECK: top fn arith_addf_constant_folding() -> (bits[1], bits[8], bits[23]) {
  // CHECK: literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 0)
  // CHECK: literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 128, 4194304)
  // CHECK: ret literal.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal(value=(0, 129, 2097152)
}
