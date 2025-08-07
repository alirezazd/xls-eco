// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

// Test that arith.addf with function arguments translates properly
func.func @arith_addf_with_args(%arg0: f32, %arg1: f32) -> f32 {
  %sum = arith.addf %arg0, %arg1 : f32
  return %sum : f32
  
  // CHECK: top fn arith_addf_with_args({{.*}}: (bits[1], bits[8], bits[23]) {{.*}}, {{.*}}: (bits[1], bits[8], bits[23]) {{.*}}) -> (bits[1], bits[8], bits[23]) {
  // CHECK: invoke.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = invoke({{.*}}, {{.*}}, fn=__floats__add_fN[32]
  // CHECK: ret invoke.{{[0-9]+}}: (bits[1], bits[8], bits[23]) = literal
}
