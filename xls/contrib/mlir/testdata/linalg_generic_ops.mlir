#tensor_map = affine_map<(d0, d1) -> (d0, d1)>
#scalar_map = affine_map<(d0, d1) -> ()>

module {
  // Test 1: Basic tensor + tensor element-wise addition
  func.func @test_tensor_add_tensor(%tensor1: tensor<2x2xf32>, %tensor2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = linalg.generic {indexing_maps = [#tensor_map, #tensor_map, #tensor_map], iterator_types = ["parallel", "parallel"]}
         ins(%tensor1, %tensor2 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%tensor1 : tensor<2x2xf32>) {
    ^bb0(%in1: f32, %in2: f32, %out: f32):
      %result = arith.addf %in1, %in2 : f32
      linalg.yield %result : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 2: Tensor + scalar element-wise multiplication
  func.func @test_tensor_mul_scalar(%tensor: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scalar = arith.constant 2.5 : f32
    %0 = linalg.generic {indexing_maps = [#tensor_map, #scalar_map, #tensor_map], iterator_types = ["parallel", "parallel"]}
         ins(%tensor, %scalar : tensor<2x2xf32>, f32) outs(%tensor : tensor<2x2xf32>) {
    ^bb0(%in: f32, %scalar_in: f32, %out: f32):
      %result = arith.mulf %in, %scalar_in : f32
      linalg.yield %result : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 3: Scalar + tensor element-wise subtraction
  func.func @test_scalar_sub_tensor(%tensor: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scalar = arith.constant 1.0 : f32
    %0 = linalg.generic {indexing_maps = [#scalar_map, #tensor_map, #tensor_map], iterator_types = ["parallel", "parallel"]}
         ins(%scalar, %tensor : f32, tensor<2x2xf32>) outs(%tensor : tensor<2x2xf32>) {
    ^bb0(%scalar_in: f32, %in: f32, %out: f32):
      %result = arith.subf %scalar_in, %in : f32
      linalg.yield %result : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 4: Multi-input with unused inputs (tensor + tensor, ignoring scalar)
  func.func @test_multi_input_unused(%tensor1: tensor<2x2xf32>, %tensor2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scalar = arith.constant 3.0 : f32
    %0 = linalg.generic {indexing_maps = [#tensor_map, #tensor_map, #scalar_map, #tensor_map], iterator_types = ["parallel", "parallel"]}
         ins(%tensor1, %tensor2, %scalar : tensor<2x2xf32>, tensor<2x2xf32>, f32) outs(%tensor1 : tensor<2x2xf32>) {
    ^bb0(%in1: f32, %in2: f32, %scalar_in: f32, %out: f32):
      %result = arith.addf %in1, %in2 : f32
      linalg.yield %result : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 5: Fill operation (should convert to tensor.splat)
  func.func @test_fill_operation() -> tensor<2x2xf32> {
    %output = tensor.empty() : tensor<2x2xf32>
    %0 = linalg.generic {indexing_maps = [#tensor_map], iterator_types = ["parallel", "parallel"]}
         outs(%output : tensor<2x2xf32>) {
    ^bb0(%out: f32):
      %cst = arith.constant 1.5 : f32
      linalg.yield %cst : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 6: Complex multi-operation (should go to DSLX codegen)
  func.func @test_complex_multi_op(%tensor1: tensor<2x2xf32>, %tensor2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %scalar = arith.constant 2.0 : f32
    %0 = linalg.generic {indexing_maps = [#tensor_map, #tensor_map, #scalar_map, #tensor_map], iterator_types = ["parallel", "parallel"]}
         ins(%tensor1, %tensor2, %scalar : tensor<2x2xf32>, tensor<2x2xf32>, f32) outs(%tensor1 : tensor<2x2xf32>) {
    ^bb0(%in1: f32, %in2: f32, %scalar_in: f32, %out: f32):
      %temp = arith.addf %in1, %in2 : f32
      %result = arith.mulf %temp, %scalar_in : f32
      linalg.yield %result : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test 7: Reduction operation (should go to DSLX codegen)
  func.func @test_reduction_op(%tensor: tensor<2x2xf32>) -> tensor<2xf32> {
    %output = tensor.empty() : tensor<2xf32>
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]}
         ins(%tensor : tensor<2x2xf32>) outs(%output : tensor<2xf32>) {
    ^bb0(%in: f32, %out: f32):
      %result = arith.addf %out, %in : f32
      linalg.yield %result : f32
    } -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
