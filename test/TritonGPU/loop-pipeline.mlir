// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C}>

// CHECK: func.func @matmul_loop
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK: %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_A]]
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_B]]
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_B]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK: %[[A0:.*]] = triton_gpu.extract_slice %[[A1BUFFER]][0, 0, 0]
// CHECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_1]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:   %[[arg_b0_dot_op_1:.*]] = arith.mulf %[[arg_b0_dot_op_0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_1]], {{.*}}
// CHECK-DAG: %[[INSERT_IDX:.*]] = arith.remsi %[[PIPELINE_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.remsi %[[LOOP_IDX]], %[[CONSTANT_3]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.extract_slice %[[NEXT_A_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]]
// CHECK:   scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[NEXT_LOOP_IDX]]
func.func @matmul_loop(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>


  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
    %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b_ = triton_gpu.convert_layout %b__ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return %loop#2: tensor<128x128xf32, #C>
}

// CHECK: func.func @matmul_loop_nested
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK: scf.for
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK:   %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK:   %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK:   %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK:   %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[A0:.*]] = triton_gpu.extract_slice %[[A1BUFFER]][0, 0, 0]
// CHECK:   %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK:   scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_1]]
// CHECK:     %[[arg_a0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_a0]]
// CHECK:     %[[arg_b0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:     tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[INSERT_IDX:.*]] = arith.remsi %[[PIPELINE_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.remsi %[[LOOP_IDX]], %[[CONSTANT_3]]
// CHECK:     %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:     %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:     triton_gpu.async_wait {num = 2 : i32}
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.extract_slice %[[NEXT_A_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[NEXT_A]], %[[NEXT_B]], {{.*}}, {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[NEXT_LOOP_IDX]]
func.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C>{

  %c_start = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %loop1:1 = scf.for %iv0 = %lb to %ub step %step iter_args(%c_init = %c_start) -> (tensor<128x128xf32, #C>) {
    // A ptrs
    %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
    %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    // B ptrs
    %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
    %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
    %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
    %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
    %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %loop2:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
      %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
      %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  return %loop1#0 : tensor<128x128xf32, #C>
}


// CHECK: func.func @matmul_loop_single_pipeline
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[CONSTANT_3:.*]] = arith.constant 3 : i32
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]]
// CHECK: triton_gpu.async_wait {num = 1 : i32}
// CHECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B1BUFFER]][0, 0, 0]
// CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[arg_b0:.*]] = %[[B0]], {{.*}}, {{.*}}, %[[PIPELINE_IDX:.*]] = %[[CONSTANT_2]], %[[LOOP_IDX:.*]] = %[[CONSTANT_1]]
// CHECK:   %[[arg_b0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:   tt.dot {{.*}}, %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[INSERT_IDX:.*]] = arith.remsi %[[PIPELINE_IDX]], %[[CONSTANT_3]]
// CHECK-DAG: %[[EXTRACT_IDX:.*]] = arith.remsi %[[LOOP_IDX]], %[[CONSTANT_3]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INSERT_IDX]]
// CHECK:   triton_gpu.async_wait {num = 1 : i32}
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %[[NEXT_B_BUFFER]][%[[EXTRACT_IDX]], 0, 0]
// CHECK-DAG: %[[NEXT_PIPELINE_IDX:.*]] = arith.addi %[[PIPELINE_IDX]], %[[CONSTANT_1]]
// CHECK-DAG: %[[NEXT_LOOP_IDX:.*]] = arith.addi %[[LOOP_IDX]], %[[CONSTANT_1]]
// CHECK:   scf.yield {{.*}}, {{.*}}, %[[NEXT_B_BUFFER]], %[[NEXT_B]], {{.*}}, {{.*}}, %[[NEXT_PIPELINE_IDX]], %[[NEXT_LOOP_IDX]]
func.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
                                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : (!tt.ptr<f16>) -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : (tensor<32xi32, #ALs0>) -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : (tensor<1x32xi32, #AL>) -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : (!tt.ptr<f16>) -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : (tensor<128xi32, #BLs0>) -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : (tensor<1x128xi32, #BL>) -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #AL>
  %a = triton_gpu.convert_layout %a_ : (tensor<128x32xf16, #AL>) -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b = triton_gpu.convert_layout %b_ : (tensor<32x128xf16, #BL>) -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK: func.func @lut_bmm
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.insert_slice_async
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[LUT_PTR:.*]] = tt.addptr
// CHECK: %arg27 = %[[LUT_PTR]]
// CHECK: %[[LUT_BUFFER_0:.*]] = tt.load %arg27, {{.*}}
// CHECK: %[[LUT_BUFFER_1:.*]] = arith.muli {{.*}}, %[[LUT_BUFFER_0]]
// CHECK: %[[LUT_BUFFER_2:.*]] = tt.splat %[[LUT_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[LUT_BUFFER_2]]
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %arg26, {{.*}}
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_1]]
// CHECK: triton_gpu.insert_slice_async %[[NEXT_BUFFER_0]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
func.func @lut_bmm(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) {
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %0 = tt.get_program_id {axis = 2 : i32} : i32
  %1 = tt.get_program_id {axis = 0 : i32} : i32
  %2 = tt.get_program_id {axis = 1 : i32} : i32
  %3 = tt.get_num_programs {axis = 0 : i32} : i32
  %4 = tt.get_num_programs {axis = 1 : i32} : i32
  %5 = arith.muli %1, %4 : i32
  %6 = arith.addi %5, %2 : i32
  %7 = arith.muli %4, %c4_i32 : i32
  %8 = arith.divsi %6, %7 : i32
  %9 = arith.muli %8, %c4_i32 : i32
  %10 = arith.subi %3, %9 : i32
  %11 = arith.cmpi slt, %10, %c4_i32 : i32
  %12 = arith.select %11, %10, %c4_i32 : i32
  %13 = arith.remsi %6, %12 : i32
  %14 = arith.addi %9, %13 : i32
  %15 = arith.remsi %6, %7 : i32
  %16 = arith.divsi %15, %12 : i32
  %17 = arith.muli %arg5, %0 : i32
  %18 = tt.addptr %arg4, %17 : !tt.ptr<i64>, i32
  %19 = tt.addptr %18, %14 : !tt.ptr<i64>, i32
  %20 = tt.load %19 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
  %21 = tt.addptr %19, %c1_i32 : !tt.ptr<i64>, i32
  %22 = tt.load %21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
  %23 = arith.subi %22, %20 : i64
  %24 = arith.cmpi eq, %23, %c0_i64 : i64
  cf.cond_br %24, ^bb1, ^bb2
^bb1:  // pred: ^bb0
  return
^bb2:  // pred: ^bb0
  %25 = arith.muli %arg1, %0 : i32
  %26 = tt.addptr %arg0, %25 : !tt.ptr<f16>, i32
  %27 = arith.extsi %arg2 : i32 to i64
  %28 = arith.muli %27, %20 : i64
  %29 = tt.addptr %26, %28 : !tt.ptr<f16>, i64
  %30 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  %31 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
  %32 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  %33 = tt.expand_dims %30 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16x1xi32, #blocked>
  %34 = tt.expand_dims %31 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<16x1xi32, #blocked1>
  %35 = tt.expand_dims %32 {axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16x1xi32, #blocked>
  %36 = tt.splat %arg3 : (i32) -> tensor<16x1xi32, #blocked>
  %37 = arith.muli %36, %33 : tensor<16x1xi32, #blocked>
  %38 = tt.splat %29 : (!tt.ptr<f16>) -> tensor<16x1x!tt.ptr<f16>, #blocked>
  %39 = tt.addptr %38, %37 : tensor<16x1x!tt.ptr<f16>, #blocked>, tensor<16x1xi32, #blocked>
  %40 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
  %41 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
  %42 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
  %43 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked>
  %44 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x16xi32, #blocked1>
  %45 = tt.expand_dims %42 {axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x16xi32, #blocked>
  %46 = tt.broadcast %39 : (tensor<16x1x!tt.ptr<f16>, #blocked>) -> tensor<16x16x!tt.ptr<f16>, #blocked>
  %47 = tt.broadcast %43 : (tensor<1x16xi32, #blocked>) -> tensor<16x16xi32, #blocked>
  %48 = tt.broadcast %45 : (tensor<1x16xi32, #blocked>) -> tensor<16x16xi32, #blocked>
  %49 = tt.addptr %46, %47 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
  %50 = arith.muli %arg9, %0 : i32
  %51 = tt.addptr %arg8, %50 : !tt.ptr<f16>, i32
  %52 = arith.muli %arg11, %16 : i32
  %53 = tt.addptr %51, %52 : !tt.ptr<f16>, i32
  %54 = tt.splat %53 : (!tt.ptr<f16>) -> tensor<16x1x!tt.ptr<f16>, #blocked1>
  %55 = tt.addptr %54, %34 : tensor<16x1x!tt.ptr<f16>, #blocked1>, tensor<16x1xi32, #blocked1>
  %56 = tt.splat %arg12 : (i32) -> tensor<1x16xi32, #blocked1>
  %57 = arith.muli %56, %44 : tensor<1x16xi32, #blocked1>
  %58 = tt.broadcast %55 : (tensor<16x1x!tt.ptr<f16>, #blocked1>) -> tensor<16x16x!tt.ptr<f16>, #blocked1>
  %59 = tt.broadcast %57 : (tensor<1x16xi32, #blocked1>) -> tensor<16x16xi32, #blocked1>
  %60 = tt.addptr %58, %59 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi32, #blocked1>
  %61 = arith.muli %arg14, %0 : i32
  %62 = tt.addptr %arg13, %61 : !tt.ptr<f16>, i32
  %63 = arith.muli %arg15, %14 : i32
  %64 = tt.addptr %62, %63 : !tt.ptr<f16>, i32
  %65 = arith.muli %arg16, %16 : i32
  %66 = tt.addptr %64, %65 : !tt.ptr<f16>, i32
  %67 = tt.splat %arg17 : (i32) -> tensor<16x1xi32, #blocked>
  %68 = arith.muli %67, %35 : tensor<16x1xi32, #blocked>
  %69 = tt.splat %66 : (!tt.ptr<f16>) -> tensor<16x1x!tt.ptr<f16>, #blocked>
  %70 = tt.addptr %69, %68 : tensor<16x1x!tt.ptr<f16>, #blocked>, tensor<16x1xi32, #blocked>
  %71 = tt.broadcast %70 : (tensor<16x1x!tt.ptr<f16>, #blocked>) -> tensor<16x16x!tt.ptr<f16>, #blocked>
  %72 = tt.addptr %71, %48 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
  %73 = arith.muli %arg7, %0 : i32
  %74 = tt.addptr %arg6, %73 : !tt.ptr<i64>, i32
  %75 = tt.addptr %74, %20 : !tt.ptr<i64>, i64
  %76 = arith.index_cast %23 : i64 to index
  %77 = arith.extsi %arg10 : i32 to i64
  %78 = tt.splat %arg2 : (i32) -> tensor<16x16xi32, #blocked>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>) {
    %82 = tt.load %arg20 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blocked>
    %83 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : (i64) -> tensor<16x16xi64, #blocked1>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #blocked1>, tensor<16x16xi64, #blocked1>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #blocked1>
    %88 = triton_gpu.convert_layout %82 : (tensor<16x16xf16, #blocked>) -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #blocked1>) -> tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true} : tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<16x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma}>> -> tensor<16x16xf32, #mma>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #blocked>, tensor<16x16xi32, #blocked>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #mma>, tensor<16x16x!tt.ptr<f16>, #blocked>, !tt.ptr<i64>
  }
  %80 = arith.truncf %79#0 : tensor<16x16xf32, #mma> to tensor<16x16xf16, #mma>
  %81 = triton_gpu.convert_layout %80 : (tensor<16x16xf16, #mma>) -> tensor<16x16xf16, #blocked>
  tt.store %72, %81 {cache = 1 : i32, evict = 1 : i32} : tensor<16x16xf16, #blocked>
  return
}
