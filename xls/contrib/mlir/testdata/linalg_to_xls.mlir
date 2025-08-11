// RUN: xls_opt %s --linalg-to-xls --arith-to-xls | FileCheck %s

// Test linalg.generic operations with various arith operations

#map = affine_map<(d0, d1) -> (d0, d1)>

module {
  // Test single operation - multiplication
  // CHECK-LABEL: @test_mulf
  // CHECK: arith.mulf %arg0, %arg1
  func.func @test_mulf(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} 
         ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) 
         outs(%arg0 : tensor<2x2xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %1 = arith.mulf %in, %in_1 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }

  // Test single operation - comparison
  // CHECK-LABEL: @test_cmp
  // CHECK: arith.cmpf ogt, %arg0, %arg1
  func.func @test_cmp(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xi1> {
    %output = tensor.empty() : tensor<2x2xi1>
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} 
         ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>) 
         outs(%output : tensor<2x2xi1>) {
    ^bb0(%in: f32, %in_1: f32, %out: i1):
      %1 = arith.cmpf ogt, %in, %in_1 : f32
      linalg.yield %1 : i1
    } -> tensor<2x2xi1>
    return %0 : tensor<2x2xi1>
  }

  // Test single operation - select
  // CHECK-LABEL: @test_select
  // CHECK: arith.select %arg0, %arg1, %arg2
  func.func @test_select(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} 
         ins(%arg0, %arg1, %arg2 : tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) 
         outs(%arg1 : tensor<2x2xf32>) {
    ^bb0(%in: i1, %in_1: f32, %in_2: f32, %out: f32):
      %1 = arith.select %in, %in_1, %in_2 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }



}
