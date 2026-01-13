// RUN: triton-opt %s -split-input-file -tritongpu-prefetch -canonicalize | FileCheck %s --dump-input-context=50

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @matmul_loop_mixed
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[A0:.*]][0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[B0:.*]][0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_a0]][0, 16]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_b0]][16, 0]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x32xf8E5M2, #A, #smem> -> tensor<128x32xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x32xf8E5M2, #A_OP> -> tensor<128x32xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<32x128xf16, #B, #smem> -> tensor<32x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

// 4 warps
// matmul: 128x16 @ 16x128 -> 128x128
// CHECK: tt.func @matmul_loop_mixed_4warps
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[A0:.*]][0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[B0:.*]][0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[a0_prefetch]], %[[b0_prefetch]], {{.*}}
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_loop_mixed_4warps(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x16x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<16x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x16xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x16xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<16x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<16x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x16xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<16x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x16x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x16xf8E5M2, #AL>) -> !ttg.memdesc<128x16xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<16x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<16x128xf16, #BL>) -> !ttg.memdesc<16x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<16x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x16xf8E5M2, #A, #smem>, !ttg.memdesc<16x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x16xf8E5M2, #A, #smem> -> tensor<128x16xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x16xf8E5M2, #A_OP> -> tensor<128x16xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<16x128xf16, #B, #smem> -> tensor<16x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x16xf16, #A_OP> * tensor<16x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<128x16xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<16x128x!tt.ptr<f16>, #BL>, tensor<16x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x16x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x16xf8E5M2, #AL>) -> !ttg.memdesc<128x16xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<16x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<16x128xf16, #BL>) -> !ttg.memdesc<16x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x16x!tt.ptr<f8E5M2>, #AL>, tensor<16x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x16xf8E5M2, #A, #smem>, !ttg.memdesc<16x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

#AL_3D = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>
#BL_3D = #ttg.blocked<{sizePerThread = [1, 1, 4], threadsPerWarp = [2, 4, 4], warpsPerCTA = [1, 4, 1], order = [2, 0, 1]}>
#A_3D = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [2, 0, 1]}>
#B_3D = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [2, 0, 1]}>
#C_3D = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 4, 1], instrShape = [1, 16, 8]}>
#A_OP_3D = #ttg.dot_op<{opIdx = 0, parent = #C_3D, kWidth = 2}>
#B_OP_3D = #ttg.dot_op<{opIdx = 1, parent = #C_3D, kWidth = 2}>

// matmul: 8x128x16 @ 8x16x128 -> 8x128x128
// CHECK: tt.func @matmul_3D_loop_mixed
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[A0:.*]][0, 0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[B0:.*]][0, 0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0, 0]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0, 0]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[a0_prefetch]], %[[b0_prefetch]], {{.*}}
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_3D_loop_mixed(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<8x128x128xf32, #C_3D>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<8x16x128x!tt.ptr<f16>, #BL_3D>

  %a_mask = arith.constant dense<true> : tensor<8x128x16xi1, #AL_3D>
  %a_other = arith.constant dense<0.00e+00> : tensor<8x128x16xf8E5M2, #AL_3D>
  %b_mask = arith.constant dense<true> : tensor<8x16x128xi1, #BL_3D>
  %b_other = arith.constant dense<0.00e+00> : tensor<8x16x128xf16, #BL_3D>
  %c_init = arith.constant dense<0.00e+00> : tensor<8x128x128xf32, #C_3D>

  %a_off = arith.constant dense<4> : tensor<8x128x16xi32, #AL_3D>
  %b_off = arith.constant dense<4> : tensor<8x16x128xi32, #BL_3D>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
  %a_init = ttg.local_alloc %a_ : (tensor<8x128x16xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>
  %b_init = ttg.local_alloc %b_ : (tensor<8x16x128xf16, #BL_3D>) -> !ttg.memdesc<8x16x128xf16, #B_3D, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x16x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem> -> tensor<8x128x16xf8E5M2, #A_OP_3D>
    %a_op = tt.fp_to_fp %a_op_ : tensor<8x128x16xf8E5M2, #A_OP_3D> -> tensor<8x128x16xf16, #A_OP_3D>
    %b_op = ttg.local_load %b : !ttg.memdesc<8x16x128xf16, #B_3D, #smem> -> tensor<8x16x128xf16, #B_OP_3D>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<8x128x16xf16, #A_OP_3D> * tensor<8x16x128xf16, #B_OP_3D> -> tensor<8x128x128xf32, #C_3D>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x128x16xi32, #AL_3D>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, tensor<8x16x128xi32, #BL_3D>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>
    %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x16xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x16x128x!tt.ptr<f16>, #BL_3D>
    %next_b = ttg.local_alloc %b_ : (tensor<8x16x128xf16, #BL_3D>) -> !ttg.memdesc<8x16x128xf16, #B_3D, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<8x128x16x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x16x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x16xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x16x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>
  }
  tt.return %loop#4 : tensor<8x128x128xf32, #C_3D>
}
}  // end module

// matmul: 8x128x32 @ 8x32x128 -> 8x128x128
// CHECK: tt.func @matmul_3D_loop_mixed2
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[A0:.*]][0, 0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[B0:.*]][0, 0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_a0]][0, 0, 16]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_b0]][0, 16, 0]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0, 0]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0, 0]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @matmul_3D_loop_mixed2(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<8x128x128xf32, #C_3D>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<8x32x128x!tt.ptr<f16>, #BL_3D>

  %a_mask = arith.constant dense<true> : tensor<8x128x32xi1, #AL_3D>
  %a_other = arith.constant dense<0.00e+00> : tensor<8x128x32xf8E5M2, #AL_3D>
  %b_mask = arith.constant dense<true> : tensor<8x32x128xi1, #BL_3D>
  %b_other = arith.constant dense<0.00e+00> : tensor<8x32x128xf16, #BL_3D>
  %c_init = arith.constant dense<0.00e+00> : tensor<8x128x128xf32, #C_3D>

  %a_off = arith.constant dense<4> : tensor<8x128x32xi32, #AL_3D>
  %b_off = arith.constant dense<4> : tensor<8x32x128xi32, #BL_3D>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
  %a_init = ttg.local_alloc %a_ : (tensor<8x128x32xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>
  %b_init = ttg.local_alloc %b_ : (tensor<8x32x128xf16, #BL_3D>) -> !ttg.memdesc<8x32x128xf16, #B_3D, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x32x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem> -> tensor<8x128x32xf8E5M2, #A_OP_3D>
    %a_op = tt.fp_to_fp %a_op_ : tensor<8x128x32xf8E5M2, #A_OP_3D> -> tensor<8x128x32xf16, #A_OP_3D>
    %b_op = ttg.local_load %b : !ttg.memdesc<8x32x128xf16, #B_3D, #smem> -> tensor<8x32x128xf16, #B_OP_3D>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<8x128x32xf16, #A_OP_3D> * tensor<8x32x128xf16, #B_OP_3D> -> tensor<8x128x128xf32, #C_3D>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x128x32xi32, #AL_3D>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, tensor<8x32x128xi32, #BL_3D>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>
    %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x32xf8E5M2, #AL_3D>) -> !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x32x128x!tt.ptr<f16>, #BL_3D>
    %next_b = ttg.local_alloc %b_ : (tensor<8x32x128xf16, #BL_3D>) -> !ttg.memdesc<8x32x128xf16, #B_3D, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<8x128x32x!tt.ptr<f8E5M2>, #AL_3D>, tensor<8x32x128x!tt.ptr<f16>, #BL_3D>, !ttg.memdesc<8x128x32xf8E5M2, #A_3D, #smem>, !ttg.memdesc<8x32x128xf16, #B_3D, #smem>, tensor<8x128x128xf32, #C_3D>
  }
  tt.return %loop#4 : tensor<8x128x128xf32, #C_3D>
}
}  // end module

// CHECK: tt.func @matmul_loop_yield_no_operand
// CHECK: scf.for
// CHECK: scf.if
// CHECK: tt.store
// CHECK-NOT: scf.yield
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:86", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_loop_yield_no_operand(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %c32_i32 = arith.constant 32 : i32
    %c31_i32 = arith.constant 31 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.muli %arg9, %arg10 : i32
    %1 = arith.addi %arg8, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.addi %0, %c31_i32 : i32
    %4 = arith.divsi %3, %c32_i32 : i32
    %5 = arith.muli %1, %4 : i32
    %6 = tt.get_program_id x : i32
    %7 = tt.get_num_programs x : i32
    %8 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    scf.for %arg11 = %6 to %5 step %7  : i32 {
      %9 = arith.divsi %arg11, %4 : i32
      %10 = arith.remsi %9, %2 : i32
      %11 = tt.load %8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %12 = tt.load %8 : tensor<32x32x!tt.ptr<f16>, #blocked>
      %13 = ttg.convert_layout %12 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %14 = ttg.convert_layout %11 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = tt.dot %13, %14, %cst, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
      %16 = arith.cmpi sgt, %10, %c0_i32 : i32
      %17 = scf.if %16 -> (tensor<32x32xf32, #mma>) {
        %21 = tt.dot %13, %14, %15, inputPrecision = tf32 : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
        scf.yield %21 : tensor<32x32xf32, #mma>
      } else {
        scf.yield %15 : tensor<32x32xf32, #mma>
      }
      %18 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked1>
      %19 = arith.truncf %17 : tensor<32x32xf32, #mma> to tensor<32x32xf16, #mma>
      %20 = ttg.convert_layout %19 : tensor<32x32xf16, #mma> -> tensor<32x32xf16, #blocked1>
      tt.store %18, %20 : tensor<32x32x!tt.ptr<f16>, #blocked1>
    }
    tt.return
  }
}

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = false}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @matmul_loop_mixed_amd
// CHECK-DAG: %[[A0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[A0:.*]][0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.*]] = ttg.local_load %[[A0_PREFETCH_SMEM]]
// CHECK-DAG: %[[A0_CVT:.*]] = tt.fp_to_fp %[[A0_PREFETCH]]
// CHECK-DAG: %[[B0_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice %[[B0:.*]][0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.*]] = ttg.local_load %[[B0_PREFETCH_SMEM]]
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = %[[A0]], %[[arg_b0:.*]] = %[[B0]], {{.*}}, %[[a0_prefetch:.*]] = %[[A0_CVT]], %[[b0_prefetch:.*]] = %[[B0_PREFETCH]]
// CHECK-DAG:   %[[A_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_a0]][0, 16]
// CHECK-DAG:   %[[A_REM:.*]] = ttg.local_load %[[A_REM_SMEM]]
// CHECK-DAG:   %[[A_REM_CVT:.*]] = tt.fp_to_fp %[[A_REM]]
// CHECK-DAG:   %[[B_REM_SMEM:.*]] = ttg.memdesc_subslice %[[arg_b0]][16, 0]
// CHECK-DAG:   %[[B_REM:.*]] = ttg.local_load %[[B_REM_SMEM]]
// CHECK:       %[[D_FIRST:.*]] = tt.dot %[[a0_prefetch]], %[[b0_prefetch:.*]], {{.*}}
// CHECK-DAG:   %[[NEXT_A_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_A_PREFETCH:.*]] = ttg.local_load %[[NEXT_A_PREFETCH_SMEM]]
// CHECK-DAG:   %[[NEXT_A_PREFETCH_CVT:.*]] = tt.fp_to_fp %[[NEXT_A_PREFETCH]]
// CHECK-DAG:   %[[NEXT_B_PREFETCH_SMEM:.*]] = ttg.memdesc_subslice {{.*}}[0, 0]
// CHECK-DAG:   %[[NEXT_B_PREFETCH:.*]] = ttg.local_load %[[NEXT_B_PREFETCH_SMEM]]
// CHECK:       tt.dot %[[A_REM_CVT]], %[[B_REM]], %[[D_FIRST:.*]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[NEXT_A_PREFETCH_CVT]], %[[NEXT_B_PREFETCH]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
tt.func @matmul_loop_mixed_amd(%lb : index, %ub : index, %step : index, %A : !tt.ptr<f8E5M2>, %B : !tt.ptr<f16>) -> tensor<128x128xf32, #C>{
  %a_ptr_init = tt.splat %A : !tt.ptr<f8E5M2> -> tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %b_ptr_init = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E5M2, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
  %a_init = ttg.local_alloc %a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
  %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
  %b_init = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>) {
    %a_op_ = ttg.local_load %a : !ttg.memdesc<128x32xf8E5M2, #A, #smem> -> tensor<128x32xf8E5M2, #A_OP>
    %a_op = tt.fp_to_fp %a_op_ : tensor<128x32xf8E5M2, #A_OP> -> tensor<128x32xf16, #A_OP>
    %b_op = ttg.local_load %b : !ttg.memdesc<32x128xf16, #B, #smem> -> tensor<32x128xf16, #B_OP>
    %c = tt.dot %a_op, %b_op, %prev_c : tensor<128x32xf16, #A_OP> * tensor<32x128xf16, #B_OP> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f8E5M2>, #AL>
    %next_a = ttg.local_alloc %next_a_ : (tensor<128x32xf8E5M2, #AL>) -> !ttg.memdesc<128x32xf8E5M2, #A, #smem>
    %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %next_b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #B, #smem>

    scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c : tensor<128x32x!tt.ptr<f8E5M2>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, !ttg.memdesc<128x32xf8E5M2, #A, #smem>, !ttg.memdesc<32x128xf16, #B, #smem>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4 : tensor<128x128xf32, #C>
}
}  // end module

// -----

// matmul: local_loads with async_wait tokens
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 2], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: lds_prefetch_matmul_async_copy
  // CHECK-DAG: %[[A_ALLOC:.*]] = ttg.local_alloc : {{.*}} -> !ttg.memdesc<2x128x256xf16
  // CHECK-DAG: %[[B_ALLOC:.*]] = ttg.local_alloc : {{.*}} -> !ttg.memdesc<2x256x128xf16
  // CHECK: %[[TOKEN_INIT:.*]] = ttg.async_wait {num = 0 : i32}
  // CHECK: %[[A_IDX_INIT:.*]] = ttg.memdesc_index %[[A_ALLOC]][{{.*}}] : !ttg.memdesc<2x128x256xf16{{.*}}> -> !ttg.memdesc<128x256xf16
  // CHECK: %[[B_IDX_INIT:.*]] = ttg.memdesc_index %[[B_ALLOC]][{{.*}}] : !ttg.memdesc<2x256x128xf16{{.*}}> -> !ttg.memdesc<256x128xf16
  // CHECK: %[[A_SUBSLICE_INIT:.*]] = ttg.memdesc_subslice %[[A_IDX_INIT]][0, 0] : !ttg.memdesc<128x256xf16{{.*}}> -> !ttg.memdesc<128x64xf16{{.*}}, 128x256>
  // CHECK: %[[A_PREFETCH_INIT:.*]] = ttg.local_load %[[A_SUBSLICE_INIT]] token %[[TOKEN_INIT]]
  // CHECK: %[[B_SUBSLICE_INIT:.*]] = ttg.memdesc_subslice %[[B_IDX_INIT]][0, 0] : !ttg.memdesc<256x128xf16{{.*}}> -> !ttg.memdesc<64x128xf16{{.*}}, 256x128>
  // CHECK: %[[B_PREFETCH_INIT:.*]] = ttg.local_load %[[B_SUBSLICE_INIT]] token %[[TOKEN_INIT]]
  // CHECK: scf.for {{.*}} iter_args({{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[TOKEN:.*]] = %[[TOKEN_INIT]], %[[A_IDX:.*]] = %[[A_IDX_INIT]], %[[B_IDX:.*]] = %[[B_IDX_INIT]], %[[A_PREFETCH:.*]] = %[[A_PREFETCH_INIT]], %[[B_PREFETCH:.*]] = %[[B_PREFETCH_INIT]]
  // CHECK: %[[A_IDX_NEXT:.*]] = ttg.memdesc_index %[[A_ALLOC]]
  // CHECK: %[[A_COPY:.*]] = ttg.async_copy_global_to_local {{.*}}, %[[A_IDX_NEXT]]
  // CHECK: %[[A_COMMIT:.*]] = ttg.async_commit_group tokens %[[A_COPY]]
  // CHECK: %[[B_IDX_NEXT:.*]] = ttg.memdesc_index %[[B_ALLOC]]
  // CHECK: %[[B_COPY:.*]] = ttg.async_copy_global_to_local {{.*}}, %[[B_IDX_NEXT]]
  // CHECK: %[[B_COMMIT:.*]] = ttg.async_commit_group tokens %[[B_COPY]]
  // CHECK: %[[A_SUBSLICE_1:.*]] = ttg.memdesc_subslice %[[A_IDX]][0, 64]
  // CHECK: %[[A_LOAD_1:.*]] = ttg.local_load %[[A_SUBSLICE_1]] token %[[TOKEN]]
  // CHECK: %[[B_SUBSLICE_1:.*]] = ttg.memdesc_subslice %[[B_IDX]][64, 0]
  // CHECK: %[[B_LOAD_1:.*]] = ttg.local_load %[[B_SUBSLICE_1]] token %[[TOKEN]]
  // CHECK: %[[DOT_0:.*]] = tt.dot %[[A_PREFETCH]], %[[B_PREFETCH]]
  // CHECK: %[[A_SUBSLICE_2:.*]] = ttg.memdesc_subslice %[[A_IDX]][0, 128]
  // CHECK: %[[A_LOAD_2:.*]] = ttg.local_load %[[A_SUBSLICE_2]] token %[[TOKEN]]
  // CHECK: %[[B_SUBSLICE_2:.*]] = ttg.memdesc_subslice %[[B_IDX]][128, 0]
  // CHECK: %[[B_LOAD_2:.*]] = ttg.local_load %[[B_SUBSLICE_2]] token %[[TOKEN]]
  // CHECK: %[[DOT_1:.*]] = tt.dot %[[A_LOAD_1]], %[[B_LOAD_1]], %[[DOT_0]]
  // CHECK: %[[A_SUBSLICE_3:.*]] = ttg.memdesc_subslice %[[A_IDX]][0, 192]
  // CHECK: %[[A_LOAD_3:.*]] = ttg.local_load %[[A_SUBSLICE_3]] token %[[TOKEN]]
  // CHECK: %[[B_SUBSLICE_3:.*]] = ttg.memdesc_subslice %[[B_IDX]][192, 0]
  // CHECK: %[[B_LOAD_3:.*]] = ttg.local_load %[[B_SUBSLICE_3]] token %[[TOKEN]]
  // CHECK: %[[DOT_2:.*]] = tt.dot %[[A_LOAD_2]], %[[B_LOAD_2]], %[[DOT_1]]
  // CHECK: %[[TOKEN_NEXT:.*]] = ttg.async_wait %[[A_COMMIT]], %[[B_COMMIT]] {num = 0 : i32}
  // CHECK: %[[A_SUBSLICE_NEXT:.*]] = ttg.memdesc_subslice %[[A_IDX_NEXT]][0, 0]
  // CHECK: %[[A_PREFETCH_NEXT:.*]] = ttg.local_load %[[A_SUBSLICE_NEXT]] token %[[TOKEN_NEXT]]
  // CHECK: %[[B_SUBSLICE_NEXT:.*]] = ttg.memdesc_subslice %[[B_IDX_NEXT]][0, 0]
  // CHECK: %[[B_PREFETCH_NEXT:.*]] = ttg.local_load %[[B_SUBSLICE_NEXT]] token %[[TOKEN_NEXT]]
  // CHECK: %[[DOT_3:.*]] = tt.dot %[[A_LOAD_3]], %[[B_LOAD_3]], %[[DOT_2]]
  // CHECK: scf.yield %[[DOT_3]], {{.*}}, {{.*}}, {{.*}}, %[[TOKEN_NEXT]], %[[A_IDX_NEXT]], %[[B_IDX_NEXT]], %[[A_PREFETCH_NEXT]], %[[B_PREFETCH_NEXT]]

  tt.func public @lds_prefetch_matmul_async_copy(%a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %c_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<4> : tensor<128x256xi32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<256x128xi32, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %a_ptr_init = tt.splat %a_ptr : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    %b_ptr_init = tt.splat %b_ptr : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %a = ttg.local_alloc : () -> !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable>
    %a_token_init = ttg.async_wait {num = 0 : i32}
    %a_idx_init = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    %b_idx_init = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared1, #smem, mutable>
    %loop:10 = scf.for %iv = %c0_i32 to %K step %c1_i32 iter_args(%accumulator = %cst_1, %a_ptrs = %a_ptr_init, %b_ptrs = %b_ptr_init, %buf_idx = %c0_i32, %a_memdesc = %a, %a_token = %a_token_init, %b_memdesc = %b, %b_token = %a_token_init, %a_idx_arg = %a_idx_init, %b_idx_arg = %b_idx_init) -> (tensor<128x128xf32, #mma>, tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<256x128x!tt.ptr<f16>, #blocked1>, i32, !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.memdesc<128x256xf16, #shared, #smem, mutable>, !ttg.memdesc<256x128xf16, #shared1, #smem, mutable>)  : i32 {
      %a_ptrs_next = tt.addptr %a_ptrs, %cst : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
      %b_ptrs_next = tt.addptr %b_ptrs, %cst_0 : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
      %buf_idx_next = arith.addi %buf_idx, %c1_i32 : i32
      %should_reset = arith.cmpi slt, %buf_idx_next, %c1_i32 : i32
      %buf_idx_wrapped = arith.select %should_reset, %buf_idx_next, %c0_i32 : i32
      %iv_next = arith.addi %iv, %c1_i32 : i32

      %a_loaded = ttg.local_load %a_idx_arg token %a_token : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> tensor<128x256xf16, #blocked>
      %b_loaded = ttg.local_load %b_idx_arg token %b_token : !ttg.memdesc<256x128xf16, #shared1, #smem, mutable> -> tensor<256x128xf16, #blocked1>

      %a_idx_next = ttg.memdesc_index %a_memdesc[%buf_idx_wrapped] : !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
      %a_copy_token = ttg.async_copy_global_to_local %a_ptrs_next, %a_idx_next : tensor<128x256x!tt.ptr<f16>, #blocked> -> <128x256xf16, #shared, #smem, mutable>
      %a_commit_token = ttg.async_commit_group tokens %a_copy_token

      %b_idx_next = ttg.memdesc_index %b_memdesc[%buf_idx_wrapped] : !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared1, #smem, mutable>
      %b_copy_token = ttg.async_copy_global_to_local %b_ptrs_next, %b_idx_next : tensor<256x128x!tt.ptr<f16>, #blocked1> -> <256x128xf16, #shared1, #smem, mutable>
      %b_commit_token = ttg.async_commit_group tokens %b_copy_token

      %a_converted = ttg.convert_layout %a_loaded : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %b_converted = ttg.convert_layout %b_loaded : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %accumulator_next = tt.dot %a_converted, %b_converted, %accumulator : tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
      %wait_token = ttg.async_wait %a_commit_token, %b_commit_token {num = 0 : i32}
      scf.yield %accumulator_next, %a_ptrs_next, %b_ptrs_next, %buf_idx_wrapped, %a_memdesc, %wait_token, %b_memdesc, %wait_token, %a_idx_next, %b_idx_next : tensor<128x128xf32, #mma>, tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<256x128x!tt.ptr<f16>, #blocked1>, i32, !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.memdesc<128x256xf16, #shared, #smem, mutable>, !ttg.memdesc<256x128xf16, #shared1, #smem, mutable>
    }
    ttg.local_dealloc %b : !ttg.memdesc<2x256x128xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %a : !ttg.memdesc<2x128x256xf16, #shared, #smem, mutable>
    tt.return
  }
}  // end module
