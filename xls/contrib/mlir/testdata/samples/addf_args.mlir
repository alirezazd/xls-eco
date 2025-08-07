func.func @arith_addf_dyn(%a: f32, %b: f32) -> f32 {
  %sum = arith.addf %a, %b : f32
  return %sum : f32
}
