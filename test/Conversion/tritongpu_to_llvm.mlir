// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s


module attributes {"triton_gpu.num-warps" = 4 : i32} {

// CHECK: llvm.func @test_empty_kernel(%arg0: i32, %arg1: !llvm.ptr<f16, 1>)
// Here the 128 comes from the 4 in module attribute multiples 32
// CHECK: attributes {nvvm.maxntid = 128 : i32} {{.*}}
func @test_empty_kernel(%lb : index, %A : !tt.ptr<f16>) {

  // CHECK:  llvm.return
  return
}

} // end module

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_load
  func @basic_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK: llvm.inline_asm
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load
  func @vectorized_load(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.v4.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.v4.b32
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [8], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vectorized_load_f16
  func @vectorized_load_f16(%a_ptr_init : tensor<256x!tt.ptr<f16>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf16, #blocked0>) {
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.v2.b32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ld.global.v2.b32
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf16, #blocked0>
    return
  }
}

// -----

// TODO: Pending on the support of isSplat constant 
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: masked_load_const_other
  func @masked_load_const_other(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0>
    %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0>
    return
  }
}

// TODO: Add a testcase to verify the optimization when ptr of the LoadOp
//       is from a GEP with const idx

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_view_broadcast
  func @basic_view_broadcast(%arg : tensor<256xf32,#blocked0>) {
    // CHECK: llvm.mlir.undef
    // CHECK: %[[T0:.*]] = llvm.extractvalue
    // CHECK: %[[T1:.*]] = llvm.extractvalue
    %0 = tt.view %arg : (tensor<256xf32, #blocked0>) -> tensor<256x1xf32,#blocked2> 
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T0]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    // CHECK: llvm.insertvalue %[[T1]]
    %1 = tt.broadcast %0 : (tensor<256x1xf32,#blocked2>) -> tensor<256x4xf32, #blocked2> 
    return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: basic_make_range
  func @basic_make_range() {
    // CHECK: nvvm.read.ptx.sreg.tid.x
    // CHECK: llvm.mlir.undef
    // CHECK: llvm.insertvalue
    // CHECK: llvm.insertvalue
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked0>
    return
  }
}

// #blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// #blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
// #blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// module attributes {"triton_gpu.num-warps" = 4 : i32} { 
//   func @debut_kernel(%lb : index, %A : !tt.ptr<f32>, %B : !tt.ptr<f32>, %C : !tt.ptr<f32>) { 
//     %cst = arith.constant dense<true> : tensor<256xi1, #blocked0> 
//     %cst_0 = arith.constant dense<0.000000e+00> : tensor<256xf32, #blocked0> 
//     %cst_1 = arith.constant dense<true> : tensor<1024x256xi1, #blocked1> 
//     %cst_2 = arith.constant dense<true> : tensor<256x2048xi1, #blocked2> 
//     %a_ptr_init = tt.splat %A : (!tt.ptr<f32>) -> tensor<256x!tt.ptr<f32>, #blocked0> 
//     %1 = tt.load %a_ptr_init, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256xf32, #blocked0> 
//     %4 = tt.view %1 : (tensor<256xf32, #blocked0>) -> tensor<1x256xf32,#blocked1> 
//     %5 = tt.broadcast %4 : (tensor<1x256xf32,#blocked1>) -> tensor<1024x256xf32, #blocked1> 
//     %6 = tt.view %1 : (tensor<256xf32, #blocked0>) -> tensor<256x1xf32,#blocked2> 
//     %7 = tt.broadcast %6 : (tensor<256x1xf32,#blocked2>) -> tensor<256x2048xf32, #blocked2> 
//     %b_ptr_init = tt.splat %A : (!tt.ptr<f32>) -> tensor<1024x256x!tt.ptr<f32>, #blocked1> 
//     %c_ptr_init = tt.splat %A : (!tt.ptr<f32>) -> tensor<256x2048x!tt.ptr<f32>, #blocked2> 
//     tt.store %b_ptr_init, %5, %cst_1, : tensor<1024x256xf32, #blocked1> 
//     tt.store %c_ptr_init, %7, %cst_2, : tensor<256x2048xf32, #blocked2> 
//     return 
//   } 
// }



