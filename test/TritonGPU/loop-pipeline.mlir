// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s --check-prefixes=COMMON,CHECK
// RUN: triton-opt %s -split-input-file -tritonamdgpu-stream-pipeline-v2=num_stages=2 -canonicalize | FileCheck %s --check-prefixes=COMMON,AMD

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
// CHECK: %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK: %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK-DAG: %[[LOOP_COND_0:.*]] = arith.cmpi slt, %[[LB:.*]], %[[UB:.*]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[ASUB:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_A0:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[ASUB]] mask %[[LOOP_COND_0_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_0_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_0]]
// CHECK-DAG: %[[BSUB:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B0:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[BSUB]] mask %[[LOOP_COND_0_SPLAT_B]] other %{{.*}}
// CHECK-DAG: %[[IV_1:.*]] = arith.addi %[[LB]], %[[STEP:.*]]
// CHECK-DAG: %[[LOOP_COND_1:.*]] = arith.cmpi slt, %[[IV_1]], %[[UB]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_A:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[ASUB1:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_A1:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[ASUB1]] mask %[[LOOP_COND_1_SPLAT_A]]
// CHECK-DAG: %[[LOOP_COND_1_SPLAT_B:.*]] = tt.splat %[[LOOP_COND_1]]
// CHECK-DAG: %[[BSUB1:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: %[[T_B1:.*]] = triton_gpu.async_copy_global_to_local %{{.*}}, %[[BSUB1]] mask %[[LOOP_COND_1_SPLAT_B]]
// CHECK-DAG: %[[A0:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK-DAG: %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK-DAG:   triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK: scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.local_load %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   %[[arg_b0_dot_op_1:.*]] = arith.mulf %[[arg_b0_dot_op_0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_1]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   %[[ASUB3:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_A_BUFFER:.*]] = triton_gpu.async_copy_global_to_local {{.*}}, %[[ASUB3]]
// CHECK:   %[[BSUB3:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   %[[NEXT_B_BUFFER:.*]] = triton_gpu.async_copy_global_to_local {{.*}}, %[[BSUB3]]
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_A:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]],
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]],
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_A]], %[[NEXT_B]]

// AMD-LABEL:  tt.func @matmul_loop
//       AMD:   %{{.*}}:6 = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}})
//       AMD:       %[[LOCAL_LOAD_32:.*]] = triton_gpu.local_load %[[ARG10]]
//       AMD:       %[[LOCAL_LOAD_33:.*]] = triton_gpu.local_load %[[ARG11]]
//       AMD:       %[[MULF_34:.*]] = arith.mulf %[[LOCAL_LOAD_33]], %{{.*}}
//       AMD:       %[[DOT_35:.*]] = tt.dot %[[LOCAL_LOAD_32]], %[[MULF_34]], %[[ARG8]]
//       AMD:       %[[ADDPTR_36:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//       AMD:       %[[ADDPTR_37:.*]] = tt.addptr %[[ARG7]], %{{.*}}
//       AMD:       %[[LOAD_38:.*]] = tt.load %[[ADDPTR_36]]
//       AMD:       %[[LOAD_39:.*]] = tt.load %[[ADDPTR_37]]
//       AMD:       %[[ADDI_40:.*]] = arith.addi %[[ARG9]], %{{.*}}
//       AMD:       %[[CMPI_41:.*]] = arith.cmpi slt, %[[ADDI_40]], %{{.*}}
//       AMD:       %[[SELECT_42:.*]] = arith.select %[[CMPI_41]], %[[ADDI_40]], %{{.*}}
//       AMD:       %[[MEMDESC_SUBVIEW_43:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_42]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[LOAD_38]], %[[MEMDESC_SUBVIEW_43]]
//       AMD:       %[[MEMDESC_SUBVIEW_44:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_42]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[LOAD_39]], %[[MEMDESC_SUBVIEW_44]]
//       AMD:       scf.yield %[[ADDPTR_36]], %[[ADDPTR_37]], %[[DOT_35]], %[[SELECT_42]], %[[MEMDESC_SUBVIEW_43]], %[[MEMDESC_SUBVIEW_44]]
//       AMD:   }
//       AMD:   %[[SUBI_21:.*]] = arith.subi %{{.*}}, %{{.*}}
//       AMD:   %[[ADDI_22:.*]] = arith.addi %[[SUBI_21]], %{{.*}}
//       AMD:   %[[ADDI_23:.*]] = arith.addi %[[ADDI_22]], %{{.*}}-1
//       AMD:   %[[DIVUI_24:.*]] = arith.divui %[[ADDI_23]], %{{.*}}
//       AMD:   %[[ADDI_25:.*]] = arith.addi %[[DIVUI_24]], %{{.*}}-1
//       AMD:   %[[CMPI_26:.*]] = arith.cmpi sge, %[[ADDI_25]], %{{.*}}
//       AMD:   %[[LOCAL_LOAD_27:.*]] = triton_gpu.local_load %{{.*}}#4
//       AMD:   %[[LOCAL_LOAD_28:.*]] = triton_gpu.local_load %{{.*}}#5
//       AMD:   %[[MULF_29:.*]] = arith.mulf %[[LOCAL_LOAD_28]], %{{.*}}
//       AMD:   %[[IF_30:.*]] = scf.if %[[CMPI_26]]
//       AMD:       %[[DOT_32:.*]] = tt.dot %[[LOCAL_LOAD_27]], %[[MULF_29]], %{{.*}}#2
//       AMD:       scf.yield %[[DOT_32]]
//       AMD:   } else {
//       AMD:       scf.yield %{{.*}}#2
//       AMD:   }
//       AMD:   %[[SELECT_31:.*]] = arith.select %[[CMPI_26]], %[[IF_30]], %{{.*}}#2
//       AMD:   triton_gpu.local_dealloc %{{.*}}
//       AMD:   triton_gpu.local_dealloc %{{.*}}

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
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
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b__ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_ = triton_gpu.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: tt.func @matmul_loop_nested
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK: scf.for
// CHECK-DAG:   %[[A0:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]],
// CHECK-DAG:   %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]],
// CHECK-DAG:   triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_a0_dot_op:.*]] = triton_gpu.local_load %[[arg_a0]]
// CHECK:   %[[arg_b0_dot_op_0:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   tt.dot %[[arg_a0_dot_op]], %[[arg_b0_dot_op_0]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_A:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[EXT_IDX_3]]
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[EXT_IDX_3]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_A]], %[[NEXT_B]]
// CHECK:   triton_gpu.async_wait {num = 0 : i32}
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK    scf.yield

//   AMD-LABEL:  tt.func @matmul_loop_nested
//         AMD:  scf.for
// AMD-COUNT-2:  triton_gpu.local_alloc
// AMD-COUNT-2:  tt.load
//         AMD:  %[[SUBVIEW0:.*]] = triton_gpu.memdesc_subview
//         AMD:  triton_gpu.local_store %{{.+}}, %[[SUBVIEW0]]
//         AMD:  %[[SUBVIEW1:.*]] = triton_gpu.memdesc_subview
//         AMD:  triton_gpu.local_store %{{.+}}, %[[SUBVIEW1]]
//         AMD:  %[[FOR:.*]]:6 = scf.for
// AMD-COUNT-2:    triton_gpu.local_load
//         AMD:    tt.dot
// AMD-COUNT-2:    tt.addptr
// AMD-COUNT-2:    tt.load
//         AMD:    %[[SUBVIEW0:.*]] = triton_gpu.memdesc_subview
//         AMD:    triton_gpu.local_store %{{.+}}, %[[SUBVIEW0]]
//         AMD:    %[[SUBVIEW1:.*]] = triton_gpu.memdesc_subview
//         AMD:    triton_gpu.local_store %{{.+}}, %[[SUBVIEW1]]
//         AMD:    scf.yield
// AMD-COUNT-2:  triton_gpu.local_load
//         AMD:  %[[IF1:.*]] = scf.if
//         AMD:  %[[DOT1:.*]] = tt.dot
//         AMD:  scf.yield %[[DOT1]]
//         AMD:  %[[SEL1:.*]] = arith.select %{{.*}}, %[[IF1]], %[[FOR]]#2
// AMD-COUNT-2:  triton_gpu.local_dealloc
//         AMD:  scf.yield %[[SEL1]]

tt.func @matmul_loop_nested(%lb : index, %ub : index, %step : index,
                         %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                         %B : !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C>{

  %c_start = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %loop1:1 = scf.for %iv0 = %lb to %ub step %step iter_args(%c_init = %c_start) -> (tensor<128x128xf32, #C>) {
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

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

    %loop2:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
      %a_ = tt.load %a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
      %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
      %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
      %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

      %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
    }

    scf.yield %loop2#2 : tensor<128x128xf32, #C>
  }
  tt.return %loop1#0 : tensor<128x128xf32, #C>
}

// CHECK-LABEL: tt.func @matmul_loop_single_pipeline
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[CONSTANT_2:.*]] = arith.constant 2 : i32
// CHECK: %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK: triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[B0:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK:   scf.for {{.*}} iter_args({{.*}}, %[[INS_IDX:.*]] = %[[CONSTANT_1]], %[[EXT_IDX:.*]] = %[[CONSTANT_0]]{{.*}}, %[[arg_b0:.*]] = %[[B0]]
// CHECK:   %[[arg_b0_dot_op:.*]] = triton_gpu.local_load %[[arg_b0]]
// CHECK:   tt.dot {{.*}}, %[[arg_b0_dot_op]], {{.*}}
// CHECK-DAG: %[[INS_IDX_2:.*]] = arith.addi %[[INS_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_INS:.*]] = arith.cmpi slt, %[[INS_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[INS_IDX_3:.*]] = arith.select %[[CMP_INS]], %[[INS_IDX_2]], %[[CONSTANT_0]]
// CHECK:     triton_gpu.memdesc_subview %[[BBUFFER]][%[[INS_IDX_3]], %[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:     triton_gpu.async_copy_global_to_local
// CHECK-DAG: %[[EXT_IDX_2:.*]] = arith.addi %[[EXT_IDX]], %[[CONSTANT_1]] : i32
// CHECK-DAG: %[[CMP_EXT:.*]] = arith.cmpi slt, %[[EXT_IDX_2]], %[[CONSTANT_2]]
// CHECK-DAG: %[[EXT_IDX_3:.*]] = arith.select %[[CMP_EXT]], %[[EXT_IDX_2]], %[[CONSTANT_0]]
// CHECK-DAG: %[[NEXT_B:.*]] = triton_gpu.memdesc_subview %{{.+}}[%[[EXT_IDX_3]]
// CHECK-DAG: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK:   scf.yield {{.*}}, %[[INS_IDX_3]], %[[EXT_IDX_3]], %[[NEXT_B]]

// AMD-LABEL:  tt.func @matmul_loop_single_pipeline
//       AMD:   %[[LOAD_10:.*]] = tt.load %{{.*}}
//       AMD:   %[[CONVERT_LAYOUT_11:.*]] = triton_gpu.convert_layout %[[LOAD_10]]
//       AMD:   %[[LOCAL_ALLOC_12:.*]] = triton_gpu.local_alloc
//       AMD:   %[[CMPI_13:.*]] = arith.cmpi slt, %{{.*}}, %{{.*}}
//       AMD:   %[[SPLAT_14:.*]] = tt.splat %[[CMPI_13]]
//       AMD:   %[[LOAD_15:.*]] = tt.load %{{.*}}, %[[SPLAT_14]], %{{.*}}
//       AMD:   %[[MEMDESC_SUBVIEW_16:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%{{.*}}, %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_15]], %[[MEMDESC_SUBVIEW_16]]
//       AMD:   %[[SUBI_17:.*]] = arith.subi %{{.*}}, %{{.*}}
//       AMD:   %{{.*}}:4 = scf.for %[[ARG5:.*]] = %{{.*}} to %[[SUBI_17]] step %{{.*}} iter_args(%[[ARG6:.*]] = %{{.*}}, %[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %[[MEMDESC_SUBVIEW_16]])
//       AMD:       %[[LOCAL_LOAD_30:.*]] = triton_gpu.local_load %[[ARG9]]
//       AMD:       %[[DOT_31:.*]] = tt.dot %[[CONVERT_LAYOUT_11]], %[[LOCAL_LOAD_30]], %[[ARG7]]
//       AMD:       %[[ADDPTR_32:.*]] = tt.addptr %[[ARG6]], %{{.*}}
//       AMD:       %[[LOAD_33:.*]] = tt.load %[[ADDPTR_32]]
//       AMD:       %[[ADDI_34:.*]] = arith.addi %[[ARG8]], %{{.*}}
//       AMD:       %[[CMPI_35:.*]] = arith.cmpi slt, %[[ADDI_34]], %{{.*}}
//       AMD:       %[[SELECT_36:.*]] = arith.select %[[CMPI_35]], %[[ADDI_34]], %{{.*}}
//       AMD:       %[[MEMDESC_SUBVIEW_37:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_12]][%[[SELECT_36]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[LOAD_33]], %[[MEMDESC_SUBVIEW_37]]
//       AMD:       scf.yield %[[ADDPTR_32]], %[[DOT_31]], %[[SELECT_36]], %[[MEMDESC_SUBVIEW_37]]
//       AMD:  triton_gpu.local_dealloc %[[LOCAL_ALLOC_12]]

tt.func @matmul_loop_single_pipeline(%lb : index, %ub : index, %step : index,
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

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
  %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_b_ptr, %c : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_scalar
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = tt.load %{{.*}}, {{.*}}
// CHECK: %[[IND_BUFFER_1:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_0]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.splat %[[IND_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_2]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 2 : i32}

// AMD-LABEL:  tt.func @indirect_bmm_scalar
//       AMD:   %[[LOCAL_ALLOC_0:.*]] = triton_gpu.local_alloc
//       AMD:   %[[LOCAL_ALLOC_1:.*]] = triton_gpu.local_alloc
//       AMD:   %[[CMPI_2:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
//       AMD:   %[[SPLAT_3:.*]] = tt.splat %[[CMPI_2]]
//       AMD:   %[[LOAD_4:.*]] = tt.load %{{.*}}, %[[SPLAT_3]]
//       AMD:   %[[LOAD_5:.*]] = tt.load %{{.*}}, %[[CMPI_2]]
//       AMD:   %[[MULI_6:.*]] = arith.muli %{{.*}}, %[[LOAD_5]]
//       AMD:   %[[SPLAT_7:.*]] = tt.splat %[[MULI_6]]
//       AMD:   %[[ADDPTR_8:.*]] = tt.addptr %{{.*}}, %[[SPLAT_7]]
//       AMD:   %[[SPLAT_9:.*]] = tt.splat %[[CMPI_2]]
//       AMD:   %[[LOAD_10:.*]] = tt.load %[[ADDPTR_8]], %[[SPLAT_9]]
//       AMD:   %[[CMPI_11:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
//       AMD:   %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %{{.*}}
//       AMD:   %[[ADDPTR_13:.*]] = tt.addptr %{{.*}}, %{{.*}}
//       AMD:   %[[SPLAT_14:.*]] = tt.splat %[[CMPI_11]]
//       AMD:   %[[LOAD_15:.*]] = tt.load %[[ADDPTR_12]], %[[SPLAT_14]]
//       AMD:   %[[LOAD_16:.*]] = tt.load %[[ADDPTR_13]], %[[CMPI_11]]
//       AMD:   %[[MULI_17:.*]] = arith.muli %{{.*}}, %[[LOAD_16]]
//       AMD:   %[[SPLAT_18:.*]] = tt.splat %[[MULI_17]]
//       AMD:   %[[ADDPTR_19:.*]] = tt.addptr %{{.*}}, %[[SPLAT_18]]
//       AMD:   %[[SPLAT_20:.*]] = tt.splat %[[CMPI_11]]
//       AMD:   %[[LOAD_21:.*]] = tt.load %[[ADDPTR_19]], %[[SPLAT_20]]
//       AMD:   %[[MEMDESC_SUBVIEW_22:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%{{.*}}, %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_4]], %[[MEMDESC_SUBVIEW_22]]
//       AMD:   %[[MEMDESC_SUBVIEW_23:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%{{.*}}, %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_10]], %[[MEMDESC_SUBVIEW_23]]
//       AMD:   %[[SUBI_24:.*]] = arith.subi %{{.*}}, %{{.*}}
//       AMD:   %{{.*}}:8 = scf.for %[[ARG6:.*]] = %{{.*}} to %[[SUBI_24]] step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %[[ADDPTR_12]], %[[ARG9:.*]] = %[[ADDPTR_13]], %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %[[MEMDESC_SUBVIEW_22]], %[[ARG12:.*]] = %[[MEMDESC_SUBVIEW_23]], %[[ARG13:.*]] = %[[LOAD_15]], %[[ARG14:.*]] = %[[LOAD_21]])
//       AMD:       %[[LOCAL_LOAD_43:.*]] = triton_gpu.local_load %[[ARG11]]
//       AMD:       %[[LOCAL_LOAD_44:.*]] = triton_gpu.local_load %[[ARG12]]
//       AMD:       %[[DOT_45:.*]] = tt.dot %[[LOCAL_LOAD_43]], %[[LOCAL_LOAD_44]], %[[ARG7]]
//       AMD:       %[[ADDPTR_46:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//       AMD:       %[[ADDPTR_47:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//       AMD:       %[[LOAD_48:.*]] = tt.load %[[ADDPTR_46]]
//       AMD:       %[[LOAD_49:.*]] = tt.load %[[ADDPTR_47]]
//       AMD:       %[[MULI_50:.*]] = arith.muli %{{.*}}, %[[LOAD_49]]
//       AMD:       %[[SPLAT_51:.*]] = tt.splat %[[MULI_50]]
//       AMD:       %[[ADDPTR_52:.*]] = tt.addptr %{{.*}}, %[[SPLAT_51]]
//       AMD:       %[[LOAD_53:.*]] = tt.load %[[ADDPTR_52]]
//       AMD:       %[[ADDI_54:.*]] = arith.addi %[[ARG10]], %{{.*}}
//       AMD:       %[[CMPI_55:.*]] = arith.cmpi slt, %[[ADDI_54]], %{{.*}}
//       AMD:       %[[SELECT_56:.*]] = arith.select %[[CMPI_55]], %[[ADDI_54]], %{{.*}}
//       AMD:       %[[MEMDESC_SUBVIEW_57:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%[[SELECT_56]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[ARG13]], %[[MEMDESC_SUBVIEW_57]]
//       AMD:       %[[MEMDESC_SUBVIEW_58:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%[[SELECT_56]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[ARG14]], %[[MEMDESC_SUBVIEW_58]]
//       AMD:       scf.yield %[[DOT_45]], %[[ADDPTR_46]], %[[ADDPTR_47]], %[[SELECT_56]], %[[MEMDESC_SUBVIEW_57]], %[[MEMDESC_SUBVIEW_58]], %[[LOAD_48]], %[[LOAD_53]]
//       AMD:   }
//       AMD:   %[[ADDI_26:.*]] = arith.addi %{{.*}}, %{{.*}}-1
//       AMD:   %[[CMPI_27:.*]] = arith.cmpi sge, %[[ADDI_26]], %{{.*}}
//       AMD:   %[[ADDI_28:.*]] = arith.addi %{{.*}}, %{{.*}}-2
//       AMD:   %[[CMPI_29:.*]] = arith.cmpi sge, %[[ADDI_28]], %{{.*}}
//       AMD:   %[[LOCAL_LOAD_30:.*]] = triton_gpu.local_load %{{.*}}#4
//       AMD:   %[[LOCAL_LOAD_31:.*]] = triton_gpu.local_load %{{.*}}#5
//       AMD:   %[[IF_32:.*]] = scf.if %[[CMPI_27]]
//       AMD:       %[[DOT_43:.*]] = tt.dot %[[LOCAL_LOAD_30]], %[[LOCAL_LOAD_31]], %{{.*}}#0
//       AMD:       scf.yield %[[DOT_43]]
//       AMD:   } else {
//       AMD:       scf.yield %{{.*}}#0
//       AMD:   }
//       AMD:   %[[ADDI_33:.*]] = arith.addi %{{.*}}#3, %{{.*}}
//       AMD:   %[[CMPI_34:.*]] = arith.cmpi slt, %[[ADDI_33]], %{{.*}}
//       AMD:   %[[SELECT_35:.*]] = arith.select %[[CMPI_34]], %[[ADDI_33]], %{{.*}}
//       AMD:   %[[MEMDESC_SUBVIEW_36:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_35]], %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %{{.*}}#6, %[[MEMDESC_SUBVIEW_36]]
//       AMD:   %[[MEMDESC_SUBVIEW_37:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_35]], %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %{{.*}}#7, %[[MEMDESC_SUBVIEW_37]]
//       AMD:   %[[SELECT_38:.*]] = arith.select %[[CMPI_27]], %[[IF_32]], %{{.*}}#0
//       AMD:   %[[LOCAL_LOAD_39:.*]] = triton_gpu.local_load %[[MEMDESC_SUBVIEW_36]]
//       AMD:   %[[LOCAL_LOAD_40:.*]] = triton_gpu.local_load %[[MEMDESC_SUBVIEW_37]]
//       AMD:   %[[IF_41:.*]] = scf.if %[[CMPI_29]]
//       AMD:       %[[DOT_43:.*]] = tt.dot %[[LOCAL_LOAD_39]], %[[LOCAL_LOAD_40]], %[[SELECT_38]]
//       AMD:       scf.yield %[[DOT_43]]
//       AMD:   } else {
//       AMD:       scf.yield %[[SELECT_38]]
//       AMD:   }
//       AMD:   %[[SELECT_42:.*]] = arith.select %[[CMPI_29]], %[[IF_41]], %[[SELECT_38]]
//       AMD:   triton_gpu.local_dealloc %[[LOCAL_ALLOC_0]]
//       AMD:   triton_gpu.local_dealloc %[[LOCAL_ALLOC_1]]

tt.func @indirect_bmm_scalar(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : !tt.ptr<i64>
    %84 = arith.muli %77, %83 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_scalar_dist_one
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: scf.for %{{.*}} iter_args(%{{[^,]*}}, %{{[^,]*}}, %{{[^,]*}}, %[[IND_BUFFER_PREV:[^,]*]] = {{[^,]*}}
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = tt.load %{{.*}}, {{.*}}
// CHECK: %[[IND_BUFFER_1:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_PREV]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.splat %[[IND_BUFFER_1]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_2]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 2 : i32}
// CHECK: scf.yield {{.*}}, {{.*}}, {{.*}}, %[[IND_BUFFER_0]]

// AMD-LABEL:  tt.func @indirect_bmm_scalar_dist_one
// AMD-COUNT-4:  tt.load
//       AMD:  scf.for
//       AMD:    tt.dot
//       AMD:    tt.load
//       AMD:    triton_gpu.local_store
//       AMD:    scf.yield

tt.func @indirect_bmm_scalar_dist_one(%77: i64 {tt.divisibility=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: !tt.ptr<i64>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %50 = tt.load %75 : !tt.ptr<i64>
  %51 = tt.addptr %75, %c1_i32 : !tt.ptr<i64>, i32
  %79:4 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %51, %arg22 = %50) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : !tt.ptr<i64>
    %84 = arith.muli %77, %arg22 : i64
    %85 = tt.splat %84 : i64 -> tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32 : !tt.ptr<i64>, i32
    scf.yield %90, %91, %92, %83 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, !tt.ptr<i64>, i64
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// CHECK-LABEL: tt.func @indirect_bmm_vector
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_commit_group
// CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK: scf.for
// CHECK: tt.dot
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK-DAG: %[[IND_BUFFER_WAIT_TOKEN:.*]] = triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK-DAG: %[[IND_BUFFER_0:.*]] = triton_gpu.memdesc_subview
// CHECK: %[[IND_BUFFER_1:.*]] = triton_gpu.local_load %[[IND_BUFFER_0]] token %[[IND_BUFFER_WAIT_TOKEN]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.expand_dims %[[IND_BUFFER_1]] {axis = 1 : i32}
// CHECK: %[[IND_BUFFER_3:.*]] = tt.broadcast %[[IND_BUFFER_2]]
// CHECK: %[[IND_BUFFER_4:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_3]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_4]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}
// CHECK: scf.yield

// AMD-LABEL:  tt.func @indirect_bmm_vector
//       AMD:   %[[LOCAL_ALLOC_0:.*]] = triton_gpu.local_alloc
//       AMD:   %[[LOCAL_ALLOC_1:.*]] = triton_gpu.local_alloc
//       AMD:   %[[CMPI_2:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
//       AMD:   %[[SPLAT_3:.*]] = tt.splat %[[CMPI_2]]
//       AMD:   %[[LOAD_4:.*]] = tt.load %{{.*}}, %[[SPLAT_3]]
//       AMD:   %[[CMPI_5:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}}
//       AMD:   %[[ADDPTR_6:.*]] = tt.addptr %{{.*}}, %{{.*}}
//       AMD:   %[[SPLAT_7:.*]] = tt.splat %[[CMPI_2]]
//       AMD:   %[[LOAD_8:.*]] = tt.load %{{.*}}, %[[SPLAT_7]]
//       AMD:   %[[EXPAND_DIMS_9:.*]] = tt.expand_dims %[[LOAD_4]] {axis = 1 : i32}
//       AMD:   %[[BROADCAST_10:.*]] = tt.broadcast %[[EXPAND_DIMS_9]]
//       AMD:   %[[MULI_11:.*]] = arith.muli %{{.*}}, %[[BROADCAST_10]]
//       AMD:   %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %[[MULI_11]]
//       AMD:   %[[SPLAT_13:.*]] = tt.splat %[[CMPI_2]]
//       AMD:   %[[LOAD_14:.*]] = tt.load %[[ADDPTR_12]], %[[SPLAT_13]]
//       AMD:   %[[SPLAT_15:.*]] = tt.splat %[[CMPI_5]]
//       AMD:   %[[LOAD_16:.*]] = tt.load %[[ADDPTR_6]], %[[SPLAT_15]]
//       AMD:   %[[MEMDESC_SUBVIEW_17:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%{{.*}}, %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_8]], %[[MEMDESC_SUBVIEW_17]]
//       AMD:   %[[MEMDESC_SUBVIEW_18:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%{{.*}}, %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_14]], %[[MEMDESC_SUBVIEW_18]]
//       AMD:   %[[SUBI_19:.*]] = arith.subi %{{.*}}, %{{.*}}
//       AMD:   %{{.*}}:7 = scf.for %[[ARG6:.*]] = %{{.*}} to %[[SUBI_19]] step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %[[ADDPTR_6]], %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %[[MEMDESC_SUBVIEW_17]], %[[ARG12:.*]] = %[[MEMDESC_SUBVIEW_18]], %[[ARG13:.*]] = %[[LOAD_16]])
//       AMD:       %[[LOCAL_LOAD_47:.*]] = triton_gpu.local_load %[[ARG11]]
//       AMD:       %[[LOCAL_LOAD_48:.*]] = triton_gpu.local_load %[[ARG12]]
//       AMD:       %[[DOT_49:.*]] = tt.dot %[[LOCAL_LOAD_47]], %[[LOCAL_LOAD_48]], %[[ARG7]]
//       AMD:       %[[ADDPTR_50:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//       AMD:       %[[ADDPTR_51:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//       AMD:       %[[LOAD_52:.*]] = tt.load %[[ADDPTR_50]]
//       AMD:       %[[EXPAND_DIMS_53:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
//       AMD:       %[[BROADCAST_54:.*]] = tt.broadcast %[[EXPAND_DIMS_53]]
//       AMD:       %[[MULI_55:.*]] = arith.muli %{{.*}}, %[[BROADCAST_54]]
//       AMD:       %[[ADDPTR_56:.*]] = tt.addptr %{{.*}}, %[[MULI_55]]
//       AMD:       %[[LOAD_57:.*]] = tt.load %[[ADDPTR_56]]
//       AMD:       %[[LOAD_58:.*]] = tt.load %[[ADDPTR_51]]
//       AMD:       %[[ADDI_59:.*]] = arith.addi %[[ARG10]], %{{.*}}
//       AMD:       %[[CMPI_60:.*]] = arith.cmpi slt, %[[ADDI_59]], %{{.*}}
//       AMD:       %[[SELECT_61:.*]] = arith.select %[[CMPI_60]], %[[ADDI_59]], %{{.*}}
//       AMD:       %[[MEMDESC_SUBVIEW_62:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_0]][%[[SELECT_61]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[LOAD_52]], %[[MEMDESC_SUBVIEW_62]]
//       AMD:       %[[MEMDESC_SUBVIEW_63:.*]] = triton_gpu.memdesc_subview %[[LOCAL_ALLOC_1]][%[[SELECT_61]], %{{.*}}, %{{.*}}]
//       AMD:       triton_gpu.local_store %[[LOAD_57]], %[[MEMDESC_SUBVIEW_63]]
//       AMD:       scf.yield %[[DOT_49]], %[[ADDPTR_50]], %[[ADDPTR_51]], %[[SELECT_61]], %[[MEMDESC_SUBVIEW_62]], %[[MEMDESC_SUBVIEW_63]], %[[LOAD_58]]

tt.func @indirect_bmm_vector(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}

// COMMON-LABEL: tt.func @post_load_inv
// COMMON: scf.for
// COMMON-DAG: %[[IV:.*]] = arith.index_cast
// COMMON: %[[NEXT_IV:.*]] = arith.addi %[[IV]], %c1_i32 : i32
// COMMON: arith.index_cast
// COMMON-NOT: arith.addi %[[NEXT_IV]]
tt.func @post_load_inv(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                       %arg3: i32 {tt.divisibility = 16 : i32},
                       %arg4: i32 {tt.divisibility = 16 : i32},
                       %arg5: i32 {tt.divisibility = 16 : i32},
                       %arg6: i32 {tt.divisibility = 16 : i32},
                       %arg7: i32 {tt.divisibility = 16 : i32},
                       %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_index = arith.constant 0 : index
  %c1_index = arith.constant 1 : index
  %c1_i32 = arith.constant 1 : i32
  %c32_i32 = arith.constant 32 : i32
  %84 = arith.constant 900 : index
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %50 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #AL>
  %59 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %81 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %66 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #AL>
  %60 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %82 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %85:3 = scf.for %arg9 = %c0_index to %84 step %c1_index iter_args(%arg10 = %cst, %arg11 = %59, %arg12 = %81) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %130 = arith.index_cast %arg9 : index to i32
    %107 = arith.muli %130, %c32_i32 : i32
    %108 = arith.subi %arg5, %107 : i32
    %109 = tt.splat %108 : i32 -> tensor<1x32xi32, #AL>
    %110 = arith.cmpi "slt", %50, %109 : tensor<1x32xi32, #AL>
    %111 = tt.broadcast %110 : tensor<1x32xi1, #AL> -> tensor<32x32xi1, #AL>
    %112 = tt.load %arg11, %111, %cst_0 : tensor<32x32x!tt.ptr<f32>, #AL>
    %113 = tt.splat %108 : i32 -> tensor<32x1xi32, #AL>
    %114 = arith.cmpi "slt", %66, %113 : tensor<32x1xi32, #AL>
    %115 = tt.broadcast %114 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %116 = tt.load %arg12, %115, %cst_0 : tensor<32x32x!tt.ptr<f32>, #AL>
    %117 = triton_gpu.convert_layout %112 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %118 = triton_gpu.convert_layout %116 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %119 = tt.dot %117, %118, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %131 = arith.index_cast %arg9 : index to i32
    %120 = arith.addi %131, %c1_i32 : i32
    %121 = arith.muli %120, %c32_i32 : i32
    %122 = tt.splat %121 : i32 -> tensor<32x32xi32, #AL>
    %123 = tt.addptr %60, %122 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %124 = arith.muli %121, %arg7 : i32
    %125 = tt.splat %124 : i32 -> tensor<32x32xi32, #AL>
    %126 = tt.addptr %82, %125 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %119, %123, %126 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %85#0 : tensor<32x32xf32, #C>
}

// COMMON-LABEL: tt.func @cross_iter_dep
// TODO: enable pipelining with distance of 2
// COMMON-NOT: triton_gpu.async_commit_group
// COMMON: scf.for
// COMMON: scf.yield

tt.func @cross_iter_dep(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %arg3: i32 {tt.divisibility = 16 : i32},
                        %arg4: i32 {tt.divisibility = 16 : i32},
                        %arg5: i32 {tt.divisibility = 16 : i32},
                        %arg6: i32 {tt.divisibility = 16 : i32},
                        %arg7: i32 {tt.divisibility = 16 : i32},
                        %arg8: i32 {tt.divisibility = 16 : i32}) -> tensor<32x32xf32, #C> {
  %c0_i32 = arith.constant 0 : index
  %118 = arith.constant 32 : index
  %c1_i32 = arith.constant 1 : index
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #C>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #AL>
  %78 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %110 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %112 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %113 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %116 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %65 = tt.splat %arg3 : i32 -> tensor<1x32xi32, #AL>
  %88 = tt.splat %arg4 : i32 -> tensor<32x1xi32, #AL>
  %80 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #AL>
  %119:5 = scf.for %arg9 = %c0_i32 to %118 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %78, %arg12 = %110, %arg13 = %113, %arg14 = %116) -> (tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>)  {
    %161 = arith.index_cast %arg9 : index to i32
    %141 = arith.muli %161, %c32_i32 : i32
    %142 = arith.subi %arg5, %141 : i32
    %143 = tt.splat %142 : i32 -> tensor<1x32xi32, #AL>
    %144 = arith.cmpi "slt", %65, %143 : tensor<1x32xi32, #AL>
    %145 = tt.broadcast %144 : tensor<1x32xi1, #AL> -> tensor<32x32xi1, #AL>
    %146 = tt.load %arg11, %145, %cst_1 : tensor<32x32x!tt.ptr<f32>, #AL>
    %147 = tt.splat %142 : i32 -> tensor<32x1xi32, #AL>
    %148 = arith.cmpi "slt", %88, %147 : tensor<32x1xi32, #AL>
    %149 = tt.broadcast %148 : tensor<32x1xi1, #AL> -> tensor<32x32xi1, #AL>
    %150 = tt.load %arg12, %149, %cst_1 : tensor<32x32x!tt.ptr<f32>, #AL>
    %151 = triton_gpu.convert_layout %146 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>>
    %152 = triton_gpu.convert_layout %150 : tensor<32x32xf32, #AL> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>>
    %153 = tt.dot %151, %152, %arg10 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 1}>> -> tensor<32x32xf32, #C>
    %162 = arith.index_cast %arg9 : index to i32
    %154 = arith.addi %162, %c2_i32 : i32
    %155 = arith.muli %154, %c32_i32 : i32
    %156 = tt.splat %155 : i32 -> tensor<32x32xi32, #AL>
    %157 = tt.addptr %80, %156 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    %158 = arith.muli %155, %arg7 : i32
    %159 = tt.splat %158 : i32 -> tensor<32x32xi32, #AL>
    %160 = tt.addptr %112, %159 : tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32xi32, #AL>
    scf.yield %153, %arg13, %arg14, %157, %160 : tensor<32x32xf32, #C>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>, tensor<32x32x!tt.ptr<f32>, #AL>
  }
  tt.return %119#0 : tensor<32x32xf32, #C>
}

// COMMON-LABEL: tt.func @dep_arg_two_uses
// COMMON: tt.expand_dims
// COMMON: tt.expand_dims
// COMMON: tt.expand_dims %arg5
// COMMON-NEXT: tt.expand_dims %arg5
// COMMON: %[[PTR0:.*]] = tt.splat %arg6
// COMMON: %[[PTR1:.*]] = tt.addptr %[[PTR0]]
// COMMON-NEXT: tt.load %[[PTR1]]
tt.func @dep_arg_two_uses(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                          %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
                          %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
  %23 = arith.constant 100 : index
  %c64 = arith.constant 64 : i64
  %56 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %57 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %58 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
  %83 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
  %85 = tt.splat %c64 : i64 -> tensor<1x32xi64, #AL>
  %86 = tt.splat %c64 : i64 -> tensor<1x32xi64, #AL>
  %68 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %c32_index = arith.constant 32 : index
  %c32_i32 = arith.index_cast %c32_index : index to i32
  %80 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %cst_6 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #BL>
  %88 = arith.truncf %cst_6 : tensor<32x128xf32, #BL> to tensor<32x128xf16, #BL>
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #C>
  %90 = tt.splat %c64 : i64 -> tensor<32x128xi64, #BL>
  %92 = tt.addptr %arg1, %c32_i32 : !tt.ptr<i32>, i32
  %c0_index = arith.constant 0 : index
  %91:5 = scf.for %arg19 = %c0_index to %23 step %c32_index iter_args(%arg20 = %68, %arg21 = %83, %arg22 = %92, %arg23 = %cst, %arg24 = %80) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>)   {
    %1750 = arith.subi %23, %arg19 : index
    %175 = arith.index_cast %1750 : index to i32
    %176 = tt.splat %175 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %177 = tt.splat %175 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %178 = arith.cmpi "slt", %57, %176 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %179 = arith.cmpi "slt", %58, %177 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #BL}>>
    %180 = tt.expand_dims %178 {axis = 0 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi1, #AL>
    %181 = tt.expand_dims %179 {axis = 1 : i32} : tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #BL}>> -> tensor<32x1xi1, #BL>
    %182 = tt.expand_dims %arg21 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi32, #AL>
    %183 = tt.expand_dims %arg21 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>> -> tensor<1x32xi32, #AL>
    %184 = arith.extsi %182 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %185 = arith.extsi %183 : tensor<1x32xi32, #AL> to tensor<1x32xi64, #AL>
    %186 = arith.muli %184, %85 : tensor<1x32xi64, #AL>
    %187 = arith.muli %185, %86 : tensor<1x32xi64, #AL>
    %188 = tt.broadcast %186 : tensor<1x32xi64, #AL> -> tensor<128x32xi64, #AL>
    %189 = tt.broadcast %187 : tensor<1x32xi64, #AL> -> tensor<128x32xi64, #AL>
    %190 = tt.addptr %arg20, %188 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %191 = tt.addptr %arg20, %189 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi64, #AL>
    %192 = tt.broadcast %180 : tensor<1x32xi1, #AL> -> tensor<128x32xi1, #AL>
    %193 = tt.load %191, %192 : tensor<128x32x!tt.ptr<f16>, #AL>
    %194 = tt.splat %arg22 : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %195 = tt.addptr %194, %56 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %196 = tt.load %195 : tensor<32x!tt.ptr<i32>, #triton_gpu.slice<{dim = 0, parent = #AL}>>
    %197 = tt.addptr %arg22, %c32_i32 : !tt.ptr<i32>, i32
    %198 = tt.broadcast %181 : tensor<32x1xi1, #BL> -> tensor<32x128xi1, #BL>
    %199 = tt.load %arg24, %198, %88 : tensor<32x128x!tt.ptr<f16>, #BL>
    %200 = triton_gpu.convert_layout %193 : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>>
    %201 = triton_gpu.convert_layout %199 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>>
    %202 = tt.dot %200, %201, %arg23 : tensor<128x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>> -> tensor<128x128xf32, #C>
    %203 = tt.addptr %arg24, %90 : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi64, #BL>
    scf.yield %190, %196, %197, %202, %203 : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #AL}>>, !tt.ptr<i32>, tensor<128x128xf32, #C>, tensor<32x128x!tt.ptr<f16>, #BL>
  }
  tt.return %91#3 : tensor<128x128xf32, #C>
}
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
// COMMON-LABEL: tt.func @load_two_users_incompatible_layouts
  tt.func @load_two_users_incompatible_layouts(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>) {
    %cst = arith.constant dense<0> : tensor<1x16xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<128x1xi32, #blocked1>
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.addptr %arg0, %c0_i64 : !tt.ptr<f16>, i64
    %1 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f16>, i64
    %2 = tt.splat %1 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %3 = tt.addptr %2, %cst_0 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %6 = tt.broadcast %3 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %7 = tt.broadcast %5 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %8 = tt.addptr %6, %7 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %9 = tt.load %8 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %10 = tt.splat %0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %cst : tensor<1x16x!tt.ptr<f16>, #blocked>, tensor<1x16xi32, #blocked>
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %14 = tt.broadcast %11 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %15 = tt.broadcast %13 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %16 = tt.addptr %14, %15 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    // check that the load didn't get pipelined.
    // COMMON-NOT: alloc
    // COMMON: scf.for
    %17:2 = scf.for %arg2 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg3 = %cst_1, %arg4 = %cst_2) -> (tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>)  : i32 {
      %18 = tt.load %16 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %19 = triton_gpu.convert_layout %9 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %20 = triton_gpu.convert_layout %18 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %21 = tt.dot %19, %20, %cst_1 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %22 = arith.truncf %21 : tensor<128x16xf32, #mma> to tensor<128x16xf16, #mma>
      %23 = triton_gpu.convert_layout %22 : tensor<128x16xf16, #mma> -> tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %24 = triton_gpu.local_alloc %18 : (tensor<64x16xf16, #blocked>) -> !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory>
      %25 = tt.trans %24 {order=array<i32: 1,0>} : !tt.memdesc<64x16xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory>
      %26 = triton_gpu.local_load %25 : !tt.memdesc<16x64xf16, #shared1, #triton_gpu.shared_memory> -> tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %27 = tt.dot %23, %26, %arg4 : tensor<128x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
      scf.yield %21, %27 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
    }
    tt.return %17#0, %17#1 : tensor<128x16xf32, #mma>, tensor<128x64xf32, #mma>
  }
}

// -----

// CHECK-LABEL: nested_loops
// CHECK: triton_gpu.local_alloc
// CHECK: scf.for
// CHECK-NOT: triton_gpu.local_alloc
// CHECK:   scf.for
// CHECK:     scf.yield
// CHECK:   triton_gpu.async_wait {num = 0 : i32}
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.async_commit_group
// CHECK:   triton_gpu.async_copy_global_to_local
// CHECK:   triton_gpu.async_commit_group
// CHECK:   scf.yield

// AMD-LABEL: tt.func public @nested_loops
//       AMD: scf.for
//       AMD:   triton_gpu.local_alloc
//   AMD-NOT:   triton_gpu.local_alloc
//       AMD:   scf.for
//       AMD:     scf.yield
//   AMD-DIS:   scf.yield

//
// The following code has the structure:
//
// ```
// for {
//   %a = load()
//   for {
//     %b = load()
//     dot(%a, %b)
//   }
// }
// ```
//
// For CUDA, we pipeline the inner loop first then pipeline the outer
// loop to prefetch the async copy after the inner loop.
// For HIP, we only pipeline the inner loop for now.
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<320> : tensor<32x1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c10_i32 = arith.constant 10 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = arith.muli %2, %cst_0 : tensor<32x1xi32, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %5 = tt.addptr %4, %3 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    %8 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %9 = arith.muli %arg4, %c32_i32 : i32
      %10 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %11 = tt.splat %9 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.addi %10, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %13 = arith.addi %11, %1 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %12 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %15 = tt.broadcast %14 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
      %16 = tt.addptr %6, %15 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %17 = tt.load %16 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %18 = tt.expand_dims %13 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
      %19 = arith.muli %18, %cst_0 : tensor<32x1xi32, #blocked>
      %20 = tt.addptr %7, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %21 = tt.broadcast %20 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      %22 = tt.addptr %8, %19 : tensor<32x1x!tt.ptr<f32>, #blocked>, tensor<32x1xi32, #blocked>
      %23 = tt.broadcast %22 : tensor<32x1x!tt.ptr<f32>, #blocked> -> tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.for %arg5 = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
        %24 = arith.muli %arg5, %c32_i32 : i32
        %25 = tt.splat %24 : i32 -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %26 = arith.addi %25, %0 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %27 = tt.expand_dims %26 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
        %28 = tt.broadcast %27 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
        %29 = tt.addptr %21, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %30 = tt.load %29 : tensor<32x32x!tt.ptr<f32>, #blocked>
        %31 = triton_gpu.convert_layout %30 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %32 = triton_gpu.convert_layout %17 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
        %33 = tt.dot %31, %32, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
        %34 = tt.addptr %23, %28 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
        %35 = triton_gpu.convert_layout %33 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
        tt.store %34, %35 : tensor<32x32x!tt.ptr<f32>, #blocked>
      }
    }
    tt.return
  }
}  // end module


// -----
// CHECK: #[[$SHARED_LAYOUT:shared.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// CHECK-LABEL: tt.func @indirect_load_shared_layout
// CHECK: scf.for
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]]
// CHECK: %[[IND_BUFFER_0:.*]] = triton_gpu.memdesc_subview {{.*}} : !tt.memdesc<1x16xi64, #[[$SHARED_LAYOUT]], #triton_gpu.shared_memory, mutable> -> !tt.memdesc<16xi64, #[[$SHARED_LAYOUT]], #triton_gpu.shared_memory, mutable>
// CHECK: %[[IND_BUFFER_1:.*]] = triton_gpu.local_load %[[IND_BUFFER_0]]
// CHECK: %[[IND_BUFFER_2:.*]] = tt.expand_dims %[[IND_BUFFER_1]] {axis = 1 : i32}
// CHECK: %[[IND_BUFFER_3:.*]] = tt.broadcast %[[IND_BUFFER_2]]
// CHECK: %[[IND_BUFFER_4:.*]] = arith.muli {{.*}}, %[[IND_BUFFER_3]]
// CHECK: %[[NEXT_BUFFER_0:.*]] = tt.addptr {{.*}}, %[[IND_BUFFER_4]]
// CHECK: triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_0]]
// CHECK: triton_gpu.async_wait {{.*}} {num = 1 : i32}

//   AMD-DIS: #[[$SHARED_LAYOUT:shared.*]] = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
// AMD-LABEL: tt.func @indirect_load_shared_layout
//       AMD:   %{{.*}}:7 = scf.for %[[ARG6:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG7:.*]] = %{{.*}}, %[[ARG8:.*]] = %{{.*}}, %[[ARG9:.*]] = %{{.*}}, %[[ARG10:.*]] = %{{.*}}, %[[ARG11:.*]] = %{{.*}}, %[[ARG12:.*]] = %{{.*}}, %[[ARG13:.*]] = %{{.*}})
//       AMD:     %[[LOCAL_LOAD_47:.*]] = triton_gpu.local_load %[[ARG11]]
//       AMD:     %[[LOCAL_LOAD_48:.*]] = triton_gpu.local_load %[[ARG12]]
//       AMD:     %[[DOT_49:.*]] = tt.dot %[[LOCAL_LOAD_47]], %[[LOCAL_LOAD_48]], %[[ARG7]]
//       AMD:     %[[ADDPTR_50:.*]] = tt.addptr %[[ARG8]], %{{.*}}
//       AMD:     %[[ADDPTR_51:.*]] = tt.addptr %[[ARG9]], %{{.*}}
//       AMD:     %[[LOAD_52:.*]] = tt.load %[[ADDPTR_50]]
//       AMD:     %[[EXPAND_DIMS_53:.*]] = tt.expand_dims %[[ARG13]] {axis = 1 : i32}
//       AMD:     %[[BROADCAST_54:.*]] = tt.broadcast %[[EXPAND_DIMS_53]]
//       AMD:     %[[MULI_55:.*]] = arith.muli %{{.*}}, %[[BROADCAST_54]]
//       AMD:     %[[ADDPTR_56:.*]] = tt.addptr %{{.*}}, %[[MULI_55]]
//       AMD:     %[[LOAD_57:.*]] = tt.load %[[ADDPTR_56]]
//       AMD:     %[[LOAD_58:.*]] = tt.load %[[ADDPTR_51]]
//       AMD:     %[[ADDI_59:.*]] = arith.addi %[[ARG10]], %{{.*}}
//       AMD:     %[[CMPI_60:.*]] = arith.cmpi slt, %[[ADDI_59]], %{{.*}}
//       AMD:     %[[SELECT_61:.*]] = arith.select %[[CMPI_60]], %[[ADDI_59]], %{{.*}}
//       AMD:     %[[MEMDESC_SUBVIEW_62:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_61]], %{{.*}}, %{{.*}}]
//       AMD:     triton_gpu.local_store %[[LOAD_52]], %[[MEMDESC_SUBVIEW_62]]
//       AMD:     %[[MEMDESC_SUBVIEW_63:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_61]], %{{.*}}, %{{.*}}]
//       AMD:     triton_gpu.local_store %[[LOAD_57]], %[[MEMDESC_SUBVIEW_63]]
//       AMD:     scf.yield %[[DOT_49]], %[[ADDPTR_50]], %[[ADDPTR_51]], %[[SELECT_61]], %[[MEMDESC_SUBVIEW_62]], %[[MEMDESC_SUBVIEW_63]], %[[LOAD_58]]
//       AMD:   }
//       AMD:   %[[ADDI_21:.*]] = arith.addi %{{.*}}, %{{.*}}-1
//       AMD:   %[[CMPI_22:.*]] = arith.cmpi sge, %[[ADDI_21]], %{{.*}}
//       AMD:   %[[ADDI_23:.*]] = arith.addi %{{.*}}, %{{.*}}-2
//       AMD:   %[[CMPI_24:.*]] = arith.cmpi sge, %[[ADDI_23]], %{{.*}}
//       AMD:   %[[LOCAL_LOAD_25:.*]] = triton_gpu.local_load %{{.*}}#4
//       AMD:   %[[LOCAL_LOAD_26:.*]] = triton_gpu.local_load %{{.*}}#5
//       AMD:   %[[IF_27:.*]] = scf.if %[[CMPI_22]]
//       AMD:     %[[DOT_47:.*]] = tt.dot %[[LOCAL_LOAD_25]], %[[LOCAL_LOAD_26]], %{{.*}}#0
//       AMD:     scf.yield %[[DOT_47]]
//       AMD:   } else {
//       AMD:     scf.yield %{{.*}}#0
//       AMD:   }
//       AMD:   %[[ADDPTR_28:.*]] = tt.addptr %{{.*}}#1, %{{.*}}
//       AMD:   %[[SPLAT_29:.*]] = tt.splat %[[CMPI_24]]
//       AMD:   %[[LOAD_30:.*]] = tt.load %[[ADDPTR_28]], %[[SPLAT_29]]
//       AMD:   %[[EXPAND_DIMS_31:.*]] = tt.expand_dims %{{.*}}#6 {axis = 1 : i32}
//       AMD:   %[[BROADCAST_32:.*]] = tt.broadcast %[[EXPAND_DIMS_31]]
//       AMD:   %[[MULI_33:.*]] = arith.muli %{{.*}}, %[[BROADCAST_32]]
//       AMD:   %[[ADDPTR_34:.*]] = tt.addptr %{{.*}}, %[[MULI_33]]
//       AMD:   %[[SPLAT_35:.*]] = tt.splat %[[CMPI_24]]
//       AMD:   %[[LOAD_36:.*]] = tt.load %[[ADDPTR_34]], %[[SPLAT_35]]
//       AMD:   %[[ADDI_37:.*]] = arith.addi %{{.*}}#3, %{{.*}}
//       AMD:   %[[CMPI_38:.*]] = arith.cmpi slt, %[[ADDI_37]], %{{.*}}
//       AMD:   %[[SELECT_39:.*]] = arith.select %[[CMPI_38]], %[[ADDI_37]], %{{.*}}
//       AMD:   %[[MEMDESC_SUBVIEW_40:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_30]], %[[MEMDESC_SUBVIEW_40]]
//       AMD:   %[[MEMDESC_SUBVIEW_41:.*]] = triton_gpu.memdesc_subview %{{.*}}[%[[SELECT_39]], %{{.*}}, %{{.*}}]
//       AMD:   triton_gpu.local_store %[[LOAD_36]], %[[MEMDESC_SUBVIEW_41]]
//       AMD:   %[[SELECT_42:.*]] = arith.select %[[CMPI_22]], %[[IF_27]], %{{.*}}#0
//       AMD:   %[[LOCAL_LOAD_43:.*]] = triton_gpu.local_load %[[MEMDESC_SUBVIEW_40]]
//       AMD:   %[[LOCAL_LOAD_44:.*]] = triton_gpu.local_load %[[MEMDESC_SUBVIEW_41]]
//       AMD:   %[[IF_45:.*]] = scf.if %[[CMPI_24]]
//       AMD:     %[[DOT_47:.*]] = tt.dot %[[LOCAL_LOAD_43]], %[[LOCAL_LOAD_44]], %[[SELECT_42]]
//       AMD:     scf.yield %[[DOT_47]]
//       AMD:   } else {
//       AMD:     scf.yield %[[SELECT_42]]
//       AMD:   }
//       AMD:   %[[SELECT_46:.*]] = arith.select %[[CMPI_24]], %[[IF_45]], %[[SELECT_42]]
//       AMD:   triton_gpu.local_dealloc %{{.*}}
//       AMD:   triton_gpu.local_dealloc %{{.*}}

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @indirect_load_shared_layout(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}
}


// -----

// CHECK-LABEL: @kernel_yield_constant
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview
// CHECK: scf.for
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.memdesc_subview
// CHECK: tt.return

// AMD-LABEL: @kernel_yield_constant
// AMD: tt.load
// AMD: triton_gpu.memdesc_subview
// AMD: triton_gpu.local_store
// AMD: scf.for
// AMD: tt.load
// AMD: triton_gpu.memdesc_subview
// AMD: triton_gpu.local_store
// AMD: tt.return
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @kernel_yield_constant(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst1 = arith.constant dense<1.000000e+00> : tensor<32x32xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %cst_1 = arith.constant dense<2.000000e+00> : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %0 = tt.get_program_id x : i32
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %arg4, %c31_i32 : i32
    %13 = arith.divsi %12, %c32_i32 : i32
    %14 = tt.expand_dims %7 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %22 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %34 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %42 = scf.for %arg7 = %c0_i32 to %13 step %c1_i32 iter_args(%arg8 = %cst) -> (tensor<32x32xf32, #mma>)  : i32 {
      %43 = arith.muli %arg7, %c32_i32 : i32
      %44 = arith.muli %43, %arg5 : i32
      %45 = tt.splat %44 : i32 -> tensor<32x32xi32, #blocked>
      %46 = tt.addptr %22, %45 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
      %47 = arith.subi %arg4, %43 : i32
      %48 = tt.splat %47 : i32 -> tensor<32x1xi32, #blocked>
      %49 = arith.cmpi slt, %14, %48 : tensor<32x1xi32, #blocked>
      %50 = tt.broadcast %49 : tensor<32x1xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %51 = tt.load %46, %50, %cst_0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %52 = triton_gpu.convert_layout %51 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %53 = tt.dot %cst_1, %52, %arg8 : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %54 = triton_gpu.convert_layout %53 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %34, %54 : tensor<32x32x!tt.ptr<f32>, #blocked>
      scf.yield %cst1 : tensor<32x32xf32, #mma>
    }
    tt.return
  }
}


// -----

// CHECK-LABEL: @add_kernel
// CHECK-DAG: %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:   %[[ABUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[BBUFFER:.*]] = triton_gpu.local_alloc
// CHECK:   %[[A0BUFFER:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[A0BUFFER]]
// CHECK:   %[[B0BUFFER:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_0]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[B0BUFFER]]
// CHECK:   %[[A1BUFFER:.*]] = triton_gpu.memdesc_subview %[[ABUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[A1BUFFER]]
// CHECK:   %[[B1BUFFER:.*]] = triton_gpu.memdesc_subview %[[BBUFFER]][%[[CONSTANT_1]], %[[CONSTANT_0]]]
// CHECK:   triton_gpu.async_copy_global_to_local {{.*}}, %[[B1BUFFER]]
// CHECK:   scf.for

// AMD-LABEL:  tt.func public @add_kernel
// AMD:  %[[LOAD_11:.*]] = tt.load %{{.*}}, %{{.*}}
// AMD:  %[[ADDPTR_12:.*]] = tt.addptr %{{.*}}, %{{.*}}
// AMD:  %[[LOAD_13:.*]] = tt.load %[[ADDPTR_12]], %{{.*}}
// AMD:  %[[ADDI_14:.*]] = arith.addi %{{.*}}, %{{.*}}
// AMD:  %[[SPLAT_15:.*]] = tt.splat %[[ADDI_14]]
// AMD:  %[[ADDI_16:.*]] = arith.addi %[[SPLAT_15]], %{{.*}}
// AMD:  %[[CMPI_17:.*]] = arith.cmpi slt, %[[ADDI_16]], %{{.*}}
// AMD:  %[[ADDPTR_18:.*]] = tt.addptr %{{.*}}, %[[ADDI_16]]
// AMD:  %[[LOAD_19:.*]] = tt.load %[[ADDPTR_18]], %[[CMPI_17]]
// AMD:  %[[ADDPTR_20:.*]] = tt.addptr %{{.*}}, %[[ADDI_16]]
// AMD:  %[[LOAD_21:.*]] = tt.load %[[ADDPTR_20]], %[[CMPI_17]]
// AMD:  scf.for
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}


// -----

// CHECK-LABEL: @nested_loops
// CHECK: tt.addptr %{{.*}}, {{.*}}
// CHECK: %[[NEXT_BUFFER_1:.*]] = tt.addptr %{{.*}}, {{.*}}
// CHECK: %[[BUFFER_1:.*]] = triton_gpu.local_alloc
// CHECK: %[[SUBVIEW_1:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK: %[[ASYNC_COPY_1:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_1]]
// CHECK: triton_gpu.async_commit_group %[[ASYNC_COPY_1]]
// CHECK: %[[SUBVIEW_2:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK: %[[ASYNC_COPY_2:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_2]]
// CHECK: triton_gpu.async_commit_group %[[ASYNC_COPY_2]]
// CHECK: scf.for
// CHECK:   %[[LOAD_1:.*]] = tt.load %[[NEXT_BUFFER_1]]
// CHECK:   %[[BUFFER_2:.*]] = triton_gpu.local_alloc %[[LOAD_1]]
// CHECK:   %[[TRANS:.*]] = tt.trans %[[BUFFER_2]]
// CHECK:   %[[LOCAL_LOAD_1:.*]] = triton_gpu.local_load %[[TRANS]]
// CHECK:   triton_gpu.async_wait
// CHECK:   triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   scf.for
// CHECK:     %[[LOCAL_LOAD_2:.*]] = triton_gpu.local_load
// CHECK:     %[[DOT:.*]] = tt.dot %[[LOCAL_LOAD_2]], %[[LOCAL_LOAD_1]]
// CHECK:     %[[CONVERT_LAYOUT_3:.*]] = triton_gpu.convert_layout %[[DOT]]
// CHECK:     %[[SUBVIEW_4:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:     %[[ASYNC_COPY_3:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_4]]
// CHECK:     triton_gpu.async_commit_group %[[ASYNC_COPY_3]]
// CHECK:     triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[SUBVIEW_6:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[ASYNC_COPY_4:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_6]] mask
// CHECK:   %[[COMMIT_1:.*]] = triton_gpu.async_commit_group %[[ASYNC_COPY_4]]
// CHECK:   %[[SUBVIEW_7:.*]] = triton_gpu.memdesc_subview %[[BUFFER_1]]
// CHECK:   %[[ASYNC_COPY_5:.*]] = triton_gpu.async_copy_global_to_local %[[NEXT_BUFFER_1]], %[[SUBVIEW_7]] mask
// CHECK:   %[[COMMIT_2:.*]] = triton_gpu.async_commit_group %[[ASYNC_COPY_5]]
// CHECK:   scf.yield %[[COMMIT_1]], %[[COMMIT_2]]
// CHECK: triton_gpu.local_dealloc %[[BUFFER_1]]

// AMD-LABEL:  tt.func public @nested_loops
// AMD-NOT:  triton_gpu.local_alloc
// AMD:      scf.for
// AMD:        triton_gpu.local_alloc
// AMD:        scf.for
// AMD:          triton_gpu.local_load
// AMD:          tt.dot
// AMD:          triton_gpu.local_store
// AMD:          scf.yield
// AMD:        triton_gpu.local_dealloc
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 2], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  tt.func public @nested_loops(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<16> : tensor<16x1xi32, #blocked>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %2 = arith.muli %1, %cst_0 : tensor<16x1xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %4 = tt.addptr %3, %2 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %7 = tt.broadcast %4 : tensor<16x1x!tt.ptr<f32>, #blocked> -> tensor<16x16x!tt.ptr<f32>, #blocked>
    %8 = tt.broadcast %6 : tensor<1x16xi32, #blocked> -> tensor<16x16xi32, #blocked>
    %9 = tt.addptr %7, %8 : tensor<16x16x!tt.ptr<f32>, #blocked>, tensor<16x16xi32, #blocked>
    scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
      %11 = triton_gpu.local_alloc %10 : (tensor<16x16xf32, #blocked>) -> !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory>
      %12 = tt.trans %11 {order = array<i32: 1, 0>} : !tt.memdesc<16x16xf32, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory>
      %13 = triton_gpu.local_load %12 : !tt.memdesc<16x16xf32, #shared1, #triton_gpu.shared_memory> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
        %14 = tt.load %9 : tensor<16x16x!tt.ptr<f32>, #blocked>
        %15 = triton_gpu.convert_layout %14 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
        %16 = tt.dot %15, %13, %cst : tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
        %17 = triton_gpu.convert_layout %16 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
        tt.store %9, %17 : tensor<16x16x!tt.ptr<f32>, #blocked>
      }
    }
    tt.return
  }
}

// -----

  // CHECK-LABEL: @int4_matmul_ampere
#blocked = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [16, 1, 2], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 8, 1], order = [2, 0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [16, 2, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 8], order = [1, 0, 2]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [32, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  tt.func public @int4_matmul_ampere(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}
  ) -> tensor<16x256xf32, #mma> attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<64x256xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<16x128xi32, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c16_i32 = arith.constant 16 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x128xf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_2 = arith.constant dense<4> : tensor<64x256xi8, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #mma>

    %35 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %36 = tt.expand_dims %35 {axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %38 = tt.broadcast %36 : tensor<1x128xi32, #blocked1> -> tensor<16x128xi32, #blocked1>
    %40 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #blocked1>
    %41 = tt.addptr %40, %38 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>

    %42 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %43 = tt.expand_dims %42 {axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %47 = tt.broadcast %43 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %50 = tt.splat %arg1 : !tt.ptr<i8> -> tensor<64x256x!tt.ptr<i8>, #blocked>
    %51 = tt.addptr %50, %47 : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>

    // Check that both loads in the loop are pipelined.
    // CHECK: scf.for
    // CHECK-NOT: tt.load
    // CHECK: triton_gpu.async_copy_global_to_local
    // CHECK-NOT: tt.load
    // CHECK: triton_gpu.async_copy_global_to_local
    // CHECK-NOT: tt.load
    // CHECK: scf.yield
    %54:3 = scf.for %arg9 = %c0_i32 to %c16_i32 step %c1_i32 iter_args(%arg10 = %cst_3, %arg11 = %41, %arg12 = %51) -> (tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>)  : i32 {
      %78 = tt.load %arg11 : tensor<16x128x!tt.ptr<f16>, #blocked1>
      %79 = tt.load %arg12 : tensor<64x256x!tt.ptr<i8>, #blocked>
      %80 = arith.shli %79, %cst_2 : tensor<64x256xi8, #blocked>
      %81 = arith.shrsi %80, %cst_2 : tensor<64x256xi8, #blocked>
      %82 = arith.shrsi %79, %cst_2 : tensor<64x256xi8, #blocked>
      %83 = arith.sitofp %81 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %84 = arith.sitofp %82 : tensor<64x256xi8, #blocked> to tensor<64x256xf16, #blocked>
      %85 = tt.join %83, %84 : tensor<64x256xf16, #blocked> -> tensor<64x256x2xf16, #blocked3>
      %86 = tt.trans %85 {order = array<i32: 0, 2, 1>} : tensor<64x256x2xf16, #blocked3> -> tensor<64x2x256xf16, #blocked4>
      %87 = tt.reshape %86 {allow_reorder = false} : tensor<64x2x256xf16, #blocked4> -> tensor<128x256xf16, #blocked5>
      %88 = triton_gpu.convert_layout %78 : tensor<16x128xf16, #blocked1> -> tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %89 = triton_gpu.convert_layout %87 : tensor<128x256xf16, #blocked5> -> tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %90 = tt.dot %88, %89, %arg10 : tensor<16x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x256xf32, #mma>
      %91 = tt.addptr %arg11, %cst_0 : tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<16x128xi32, #blocked1>
      %92 = tt.addptr %arg12, %cst : tensor<64x256x!tt.ptr<i8>, #blocked>, tensor<64x256xi32, #blocked>
      scf.yield %90, %91, %92 : tensor<16x256xf32, #mma>, tensor<16x128x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<i8>, #blocked>
    }
    tt.return %54#0 : tensor<16x256xf32, #mma>
  }
}


// -----

// This test triggered some failure in the verifier, so we only
// included a simple check for the kernel name.
// COMMON-LABEL: @load_convert_layout
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
tt.func @load_convert_layout(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %49: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #BLs1>
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %cst_0 = arith.constant dense<2> : tensor<16xi32, #BLs1>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : i32 -> tensor<16xi32, #BLs1>
  %15 = arith.cmpi slt, %1, %cst_0 : tensor<16xi32, #BLs1>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %82 = tt.load %arg20 : tensor<16x16x!tt.ptr<f16>, #AL>
    %83 = tt.load %arg21, %15 : tensor<16x!tt.ptr<i64>, #BLs1>
    %84 = tt.expand_dims %83 {axis=1: i32}: tensor<16xi64, #BLs1> -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : tensor<16x1xi64, #BL> -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %60, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 : tensor<16x16x!tt.ptr<f16>, #BL>
    %88 = triton_gpu.convert_layout %82 : tensor<16x16xf16, #AL> -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : tensor<16x16xf16, #BL> -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    %91 = tt.addptr %arg20, %78 : tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x16xi32, #AL>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    scf.yield %90, %91, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #AL>, tensor<16x!tt.ptr<i64>, #BLs1>
  } {tt.num_stages = 3 : i32}
  tt.return %79#0 : tensor<16x16xf32, #C>
}
}


// -----

// This test captured some ICE in MatmulLoopPipeline pass, so we only
// included a simple check for the kernel name.
// COMMON-LABEL: @matmul_indirect_pipeline
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 2 : i32} {
  tt.func public @matmul_indirect_pipeline(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %4 = tt.broadcast %2 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %5 = tt.broadcast %3 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %6 = arith.addi %4, %5 : tensor<32x32xi32, #blocked>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %8 = tt.addptr %7, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %9 = tt.load %8 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %10 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %11 = tt.addptr %10, %6 : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %12 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.addptr %12, %0 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %15 = tt.load %13 : tensor<32x!tt.ptr<i64>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %16 = tt.addptr %14, %15 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>, tensor<32xi64, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %17 = tt.load %16 : tensor<32x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %18 = tt.expand_dims %17 {axis = 0 : i32} : tensor<32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xf32, #blocked>
      %19 = tt.broadcast %18 : tensor<1x32xf32, #blocked> -> tensor<32x32xf32, #blocked>
      %20 = arith.addf %9, %19 : tensor<32x32xf32, #blocked>
      %21 = triton_gpu.convert_layout %9 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %22 = triton_gpu.convert_layout %20 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %23 = tt.dot %21, %22, %cst : tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      %24 = triton_gpu.convert_layout %23 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
      tt.store %11, %24 : tensor<32x32x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

// -----

// COMMON-LABEL: @dont_pipeline_128x1
// COMMON-NOT: local_load{{.*}}128x1
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @dont_pipeline_128x1(%arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_4 = arith.constant dense<-1.000000e+30> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

    %99:1 = scf.for %arg25 = %c0_i32 to %c128_i32 step %c64_i32 iter_args(%arg31 = %cst_4) -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %94 = tt.splat %arg6 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>, #blocked>
      %151 = tt.load %94 : tensor<128x1x!tt.ptr<i32>, #blocked>
      %161 = triton_gpu.convert_layout %151 : tensor<128x1xi32, #blocked> -> tensor<128x1xi32, #mma>
      %162 = tt.broadcast %161 : tensor<128x1xi32, #mma> -> tensor<128x64xi32, #mma>
      %170 = arith.sitofp %162 : tensor<128x64xi32, #mma> to tensor<128x64xf32, #mma>

      %173 = "tt.reduce"(%170) <{axis = 1 : i32}> ({
      ^bb0(%arg33: f32, %arg34: f32):
        %207 = arith.maxnumf %arg33, %arg34 : f32
        tt.reduce.return %207 : f32
      }) : (tensor<128x64xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %175 = arith.maxnumf %arg31, %173 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>

      %201 = arith.truncf %170 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %202 = triton_gpu.convert_layout %201 : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

      %192 = arith.constant dense<0.> : tensor<128x64xf32, #mma>
      %203 = arith.constant dense<0.> : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %204 = tt.dot %202, %203, %192 : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>

      scf.yield %175 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    }
    tt.return
  }
}

// -----

// Check that the dependencies across ops of different nesting does not cause crash or
// incorrect schedule that fails to pipeline.
// COMMON-LABEL: @matmul_nested_ops
// COMMON: triton_gpu.local_load

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
tt.func @matmul_nested_ops(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %ext : index) -> tensor<128x128xf32, #C> {
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
  %b_ptr = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %b_ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x128xf32, #C>) {
    %cnd = arith.cmpi slt, %iv, %ext : index
    %inc_a_ptr = scf.if %cnd -> (tensor<128x32x!tt.ptr<f16>, #AL>) {
      %a_ptr_ = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
      scf.yield %a_ptr_ : tensor<128x32x!tt.ptr<f16>, #AL>
    } else {
      scf.yield %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    }
    %a_ = tt.load %inc_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %inc_a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#1: tensor<128x128xf32, #C>
}
}

// -----

// CHECK-LABEL: @masked_add_kernel
// CHECK: %[[CONSTANT:.*]] = arith.constant dense<0xFF800000>
// CHECK:   scf.for
// CHECK: %[[A:.*]] = triton_gpu.local_load
// CHECK: arith.select {{.*}}, %[[A]], %[[CONSTANT]]
// CHECK: %[[B:.*]] = triton_gpu.local_load
// CHECK: arith.select {{.*}}, %[[B]], %[[CONSTANT]]

// AMD-LABEL: @masked_add_kernel
// AMD: %[[CONSTANT:.*]] = arith.constant dense<0xFF800000>
// AMD: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// AMD: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// AMD: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// AMD: tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// AMD: scf.for
// AMD:   arith.select
// AMD:   arith.addf
// AMD:   %[[A:.*]] = tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]
// AMD:   %[[B:.*]] = tt.load {{.*}}, %{{.*}}, %[[CONSTANT]]

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @masked_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1016800_i32 = arith.constant 1016800 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<1024xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1016800_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    scf.for %arg4 = %c0_i32 to %c1016800_i32 step %c1024_i32  : i32 {
      %7 = arith.addi %1, %arg4 : i32
      %8 = tt.splat %7 : i32 -> tensor<1024xi32, #blocked>
      %9 = arith.addi %8, %2 : tensor<1024xi32, #blocked>
      %10 = arith.cmpi slt, %9, %3 : tensor<1024xi32, #blocked>
      %11 = tt.addptr %4, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %12 = tt.load %11, %10, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %13 = tt.addptr %5, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      %14 = tt.load %13, %10, %cst : tensor<1024x!tt.ptr<f32>, #blocked>
      %15 = arith.addf %12, %14 : tensor<1024xf32, #blocked>
      %16 = tt.addptr %6, %9 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
      tt.store %16, %15, %10 : tensor<1024x!tt.ptr<f32>, #blocked>
    }{tt.num_stages = 3 : i32}
    tt.return
  }
}
