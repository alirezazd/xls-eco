// RUN: xls_opt %s --arith-to-xls | FileCheck %s

// Test comprehensive tensor operations including:
// 1. Empty float tensors
// 2. Integer tensors  
// 3. Inline dense tensors
// 4. Dense tensors with external weight references

// CHECK-LABEL: @test_pure_tensor_ops
// CHECK: tensor.empty() : tensor<4x8xf32>
// CHECK: tensor.empty() : tensor<3x4xi32>
// CHECK: arith.constant dense<{{\[}}[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]{{\]}}> : tensor<2x3xf32>
// CHECK: arith.constant dense_resource<weight_matrix_4x3> : tensor<4x3xf32>
// CHECK: tensor.from_elements
// CHECK: tensor.splat
// CHECK: tensor.extract
// CHECK: return

module {
  // Test all 4 types of tensors in one function with pure tensor operations
  func.func @test_pure_tensor_ops(%input_float: tensor<4x8xf32>, %input_int: tensor<3x4xi32>) -> (tensor<4x8xf32>, tensor<3x4xi32>, tensor<2x3xf32>, tensor<4x3xf32>, tensor<2x2xf32>) {
    
    // Test 1: Empty float tensors
    %empty_float = tensor.empty() : tensor<4x8xf32>
    
    // Test 2: Empty integer tensors  
    %empty_int = tensor.empty() : tensor<3x4xi32>
    
    // Test 3: Inline dense tensors
    %dense_float = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
    %dense_int = arith.constant dense<[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]> : tensor<3x4xi32>
    
    // Test 4: Dense tensors with external weight references
    %external_weights = arith.constant dense_resource<weight_matrix_4x3> : tensor<4x3xf32>
    %external_bias = arith.constant dense_resource<bias_vector_3> : tensor<3xf32>
    
    // Test 5: Additional tensor operations
    // Create from individual elements
    %elem0 = arith.constant 1.0 : f32
    %elem1 = arith.constant 2.0 : f32
    %elem2 = arith.constant 3.0 : f32
    %elem3 = arith.constant 4.0 : f32
    %tensor_from_elements = tensor.from_elements %elem0, %elem1, %elem2, %elem3 : tensor<2x2xf32>
    
    // Create splat tensor (all elements same value)
    %splat_value = arith.constant 10.0 : f32
    %splat_tensor = tensor.splat %splat_value : tensor<2x2xf32>
    
    // Extract some elements to demonstrate extraction
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    
    %extracted_float = tensor.extract %dense_float[%c0, %c1] : tensor<2x3xf32>
    %extracted_int = tensor.extract %dense_int[%c1, %c2] : tensor<3x4xi32>
    %extracted_external = tensor.extract %external_weights[%c0, %c0] : tensor<4x3xf32>
    
    return %empty_float, %empty_int, %dense_float, %external_weights, %tensor_from_elements : tensor<4x8xf32>, tensor<3x4xi32>, tensor<2x3xf32>, tensor<4x3xf32>, tensor<2x2xf32>
  }
}

// External weight data for test 4
{-#
  dialect_resources: {
    builtin: {
      weight_matrix_4x3: "0x040000003333B33F0000803F000000409A99D93F666666400000A04000002040CDCCCC3F66666640333353409A9999413333F340",
      bias_vector_3: "0x040000009A99993F0000004000001040"
    }
  }
#-}
