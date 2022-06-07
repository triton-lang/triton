// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize -tritongpu-verifier
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize -tritongpu-verifier | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked_layout<{
  threadTileSize = [1, 4],
  warpTileSize = [4, 32],
  blockTileSize = [16, 32],
  order = [1, 0]
}>

#BL = #triton_gpu.blocked_layout<{
  threadTileSize = [1, 4],
  warpTileSize = [1, 128],
  blockTileSize = [4, 128],
  order = [1, 0]
}>

#A = #triton_gpu.shared_layout<{
  vec = 2,
  perPhase = 2,
  maxPhase = 4,
  order = [1, 0]
}>

#B = #triton_gpu.shared_layout<{
  vec = 2,
  perPhase = 2,
  maxPhase = 4,
  order = [1, 0]
}>

// TODO: check this
#C = #triton_gpu.mma_layout<{
  fragmentPerWarp = [1, 1],
  shapePerWarp = [16, 8],
  warpPerTile = [2, 2],
  shapePerTile = [32, 16],
  repetitions = [4, 4],
  contigPerThread = [1, 8]
}>

// CHECK: func @matmul_loop
// CHECK: %[[A0:.*]] = triton_gpu.copy_async
// CHECK: %[[B0:.*]] = triton_gpu.copy_async
// CHECK: %[[A1:.*]] = triton_gpu.copy_async
// CHECK: %[[B1:.*]] = triton_gpu.copy_async
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_a1:.*]] = %[[A1]], %[[arg_b0:.*]] = %[[B0]], %[[arg_b1:.*]] = %[[B1]], {{.*}})
// CHECK:   tt.dot %[[arg_a0]], %[[arg_b0]], {{.*}}
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.copy_async
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.copy_async
// CHECK:   scf.yield {{.*}}, {{.*}}, {{.*}}, %[[arg_a1]], %[[NEXT_A]], %[[arg_b1]], %[[NEXT_B]]
func @matmul_loop(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.getelementptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
    %next_b_ptr = tt.getelementptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}


// CHECK: func @matmul_loop_nested
// CHECK: scf.for
// CHECK:   %[[A0:.*]] = triton_gpu.copy_async
// CHECK:   %[[B0:.*]] = triton_gpu.copy_async
// CHECK:   %[[A1:.*]] = triton_gpu.copy_async
// CHECK:   %[[B1:.*]] = triton_gpu.copy_async
// CHECK:   scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_a1:.*]] = %[[A1]], %[[arg_b0:.*]] = %[[B0]], %[[arg_b1:.*]] = %[[B1]], {{.*}})
// CHECK:     tt.dot %[[arg_a0]], %[[arg_b0]], {{.*}}
// CHECK:     %[[NEXT_A:.*]] = triton_gpu.copy_async
// CHECK:     %[[NEXT_B:.*]] = triton_gpu.copy_async
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[arg_a1]], %[[NEXT_A]], %[[arg_b1]], %[[NEXT_B]]
func @matmul_loop_nested(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  scf.for %iv0 = %lb to %ub step %step {
    %a_ptr_init = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
    %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
    %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
    %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {allowTF32 = true} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.getelementptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
      %next_b_ptr = tt.getelementptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }
  }
  return
}


// CHECK: func @matmul_loop_single_pipeline
// CHECK: %[[B0:.*]] = triton_gpu.copy_async
// CHECK: %[[B1:.*]] = triton_gpu.copy_async
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_b0:.*]] = %[[B0]], %[[arg_b1:.*]] = %[[B1]], {{.*}})
// CHECK:   tt.dot {{.*}}, %[[arg_b0]], {{.*}}
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.copy_async
// CHECK:   scf.yield {{.*}}, {{.*}}, %[[arg_b1]], %[[NEXT_B]]
func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f16>, %B : !tt.ptr<f16>) {
  %a_ptr_init = tt.broadcast %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %b_ptr_init = tt.broadcast %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.getelementptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}
