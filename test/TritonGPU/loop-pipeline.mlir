// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

// CHECK-LABEL: tt.func @matmul_loop
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK: %[[ABUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK: %[[BBUFFER:.*]] = triton_gpu.alloc_tensor
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[A0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK: %[[B0BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_0]], %[[LOOP_COND_0_SPLAT_B]]
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[A1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK: %[[B1BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[CONSTANT_1]], %[[LOOP_COND_1_SPLAT_B]]
// CHECK: %[[A0:.*]] = triton_gpu.extract_slice %[[A0BUFFER]][%[[CONSTANT_0]], 0, 0]
// CHECK: %[[B0:.*]] = triton_gpu.extract_slice %[[B0BUFFER]][%[[CONSTANT_0]], 0, 0]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.convert_layout %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.convert_layout %[[arg_b0]]
// CHECK:   %[[arg_b0_dot_op_1:.*]] = arith.mulf %[[arg_b0_dot_op_0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_1]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INS_IDX_3]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.insert_slice_async {{.*}}, {{.*}}, %[[INS_IDX_3]]
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK:   %[[NEXT_A:.*]] = triton_gpu.extract_slice %{{.+}}[%[[EXT_IDX_3]], 0, 0]
// CHECK:   %[[NEXT_B:.*]] = triton_gpu.extract_slice %{{.+}}[%[[EXT_IDX_3]], 0, 0]
// CHECK:   triton_gpu.async_wait {num = 2 : i32}
// CHECK:   scf.yield {{.*}}, %[[NEXT_A_BUFFER]], %[[NEXT_B_BUFFER]], %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_A]], %[[NEXT_B]]

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.compute-capability" = 80} {
tt.func @matmul_loop(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
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
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #BL>
    %b_ = triton_gpu.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}
}
