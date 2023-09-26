//  RUN: triton-opt %s -convert-triton-to-tritongpu -convert-triton-gpu-to-llvm | FileCheck %s
//  CHECK: st.shared.b8
//  CHECK: ld.shared.b8
//  CHECK-NOT: st.shared.b1
//  CHECK-NOT: ld.shared.b1
module {
  tt.func public @copyitem() attributes {noinline = false} {
    %6 = arith.constant dense<1> : tensor<4x1xi1>
    %26 = "tt.reduce"(%6) <{axis = 1 : i32}> ({
    ^bb0(%arg4: i1, %arg5: i1):
      %54 = arith.ori %arg4, %arg5 : i1
      tt.reduce.return %54 : i1
    }) : (tensor<4x1xi1>) -> tensor<4xi1>
    tt.return
  }
}
