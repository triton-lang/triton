// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #triton_gpu.mma<{version = 2, warpsPerCTA = [4, 1]}>

// CHECK: func @matmul_loop
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK: %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK: %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK: %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK: %[[A0:.*]] = tensor.extract_slice %[[A1BUFFER]][0, 0, 0] [1, 128, 32]
// CHECK: %[[B0:.*]] = tensor.extract_slice %[[B1BUFFER]][0, 0, 0] [1, 32, 128]
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = tensor.extract_slice %[[A0]][0, 0] [128, 16]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = triton_gpu.convert_layout %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = tensor.extract_slice %[[B0]][0, 0] [16, 128]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = triton_gpu.convert_layout %[[B0_PREFETCH_SMEM]]
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_0]], %[[a0_prefetch:.*]] = %[[A0_PREFETCH]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK:   %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch]], {{.*}}
// CHECK-DAG: %[[A_REM_SMEM:.*]] = tensor.extract_slice %[[arg_a0]][0, 16] [128, 16]
// CHECK-DAG: %[[A_REM:.*]] = triton_gpu.convert_layout %[[A_REM_SMEM]]
// CHECK-DAG: %[[B_REM_SMEM:.*]] = tensor.extract_slice %[[arg_b0]][16, 0] [16, 128]
// CHECK-DAG: %[[B_REM:.*]] = triton_gpu.convert_layout %[[B_REM_SMEM]]
// CHECK:   tt.dot %[[A_REM]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK-DAG: %[[INSERT_IDX:.*]] = arith.remsi %[[PIPELINE_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_INT:.*]] = arith.remsi %[[LOOP_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.index_cast %[[EXTRACT_INT]] : i32 to index
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[NEXT_A:.*]] = tensor.extract_slice %[[NEXT_A_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = tensor.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[NEXT_A_PREFETCH_SMEM:.*]] = tensor.extract_slice %[[NEXT_A]][0, 0] [128, 16]
// CHECK-DAG: %[[NEXT_A_PREFETCH:.*]] = triton_gpu.convert_layout %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG: %[[NEXT_B_PREFETCH_SMEM:.*]] = tensor.extract_slice %[[NEXT_B]][0, 0] [16, 128]
// CHECK-DAG: %[[NEXT_B_PREFETCH:.*]] = triton_gpu.convert_layout %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:   scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[NEXT_LOOP_IDX]], %[[NEXT_A_PREFETCH]], %[[NEXT_B_PREFETCH]]
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

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return
}

