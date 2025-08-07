module {
  func.func @main(%a: i32, %b: i32) -> i32 {
    %0 = arith.muli %a, %b : i32
    return %0 : i32
  }
}

////Expected XLS IR:////
// package _package

// file_number 0 "xls/contrib/mlir/testdata/samples/muli_args.mlir"

// top fn main(loc__xls_contrib_mlir_testdata_samples_muli_args_mlir__2_19_: bits[32] id=1, loc__xls_contrib_mlir_testdata_samples_muli_args_mlir__2_28_: bits[32] id=2) -> bits[32] {
//   ret umul.3: bits[32] = umul(loc__xls_contrib_mlir_testdata_samples_muli_args_mlir__2_19_, loc__xls_contrib_mlir_testdata_samples_muli_args_mlir__2_28_, id=3, pos=[(0,3,10)])
// }