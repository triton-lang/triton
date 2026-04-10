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

#A_RING = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_RING = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C_RING = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A_RING_OP = #ttg.dot_op<{opIdx = 0, parent = #C_RING, kWidth = 2}>
#B_RING_OP = #ttg.dot_op<{opIdx = 1, parent = #C_RING, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @split_pipelined_mmav2_loads
// CHECK-DAG: %[[A_VIEW:.+]] = ttg.memdesc_index %[[A_BUF:.+]][
// CHECK-DAG: %[[WAIT:.+]] = ttg.async_wait
// CHECK-DAG: %[[B_VIEW:.+]] = ttg.memdesc_index %[[B_BUF:.+]][
// CHECK-DAG: %[[A0_SMEM:.+]] = ttg.memdesc_subslice %[[A_VIEW]][0, 0]
// CHECK-DAG: %[[A0:.+]] = ttg.local_load %[[A0_SMEM]] token %[[WAIT]]
// CHECK-DAG: %[[B0_SMEM:.+]] = ttg.memdesc_subslice %[[B_VIEW]][0, 0]
// CHECK-DAG: %[[B0:.+]] = ttg.local_load %[[B0_SMEM]] token %[[WAIT]]
// CHECK: %[[LOOP:.+]]:8 = scf.for {{.+}} iter_args(%[[IDX_ARG:.+]] = %[[C0:.+]], %[[ACC_ARG:.+]] = %{{.+}}, %[[A_VIEW_ARG:.+]] = %[[A_VIEW]], %[[B_VIEW_ARG:.+]] = %[[B_VIEW]], %[[A_WAIT_ARG:.+]] = %[[WAIT]], %[[B_WAIT_ARG:.+]] = %[[WAIT]], %[[A0_ARG:.+]] = %[[A0]], %[[B0_ARG:.+]] = %[[B0]])
// CHECK-DAG: %[[A1_SMEM:.+]] = ttg.memdesc_subslice %[[A_VIEW_ARG]][0, 16]
// CHECK-DAG: %[[A1:.+]] = ttg.local_load %[[A1_SMEM]] token %[[A_WAIT_ARG]]
// CHECK-DAG: %[[B1_SMEM:.+]] = ttg.memdesc_subslice %[[B_VIEW_ARG]][16, 0]
// CHECK-DAG: %[[B1:.+]] = ttg.local_load %[[B1_SMEM]] token %[[B_WAIT_ARG]]
// CHECK: %[[DOT0:.+]] = tt.dot %[[A0_ARG]], %[[B0_ARG]], %[[ACC_ARG]]
// CHECK: ttg.memdesc_index %[[A_BUF]][
// CHECK: ttg.memdesc_index %[[B_BUF]][
// CHECK: %[[NEXT_A_HEAD_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0] : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable, 128x32>
// CHECK: ttg.local_load %[[NEXT_A_HEAD_SMEM]] token %{{.+}} : !ttg.memdesc<128x16xf16, #shared, #smem, mutable, 128x32> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
// CHECK: %[[NEXT_B_HEAD_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0] : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable, 32x128>
// CHECK: ttg.local_load %[[NEXT_B_HEAD_SMEM]] token %{{.+}} : !ttg.memdesc<16x128xf16, #shared, #smem, mutable, 32x128> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
// CHECK: tt.dot %[[A1]], %[[B1]], %[[DOT0]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @split_pipelined_mmav2_loads(%lb : index, %ub : index, %step : index, %tok0 : !ttg.async.token, %tok1 : !ttg.async.token) -> tensor<128x128xf32, #C_RING> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C_RING>
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%idx = %c0_i32, %acc = %cst) -> (i32, tensor<128x128xf32, #C_RING>) {
    %idx_p1 = arith.addi %idx, %c1_i32 : i32
    %idx_cmp = arith.cmpi sge, %idx_p1, %c3_i32 : i32
    %idx_next = arith.select %idx_cmp, %c0_i32, %idx_p1 : i32
    %wait = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
    %a_view = ttg.memdesc_index %a[%idx_next] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %a_val = ttg.local_load %a_view token %wait : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_RING_OP>
    %b_view = ttg.memdesc_index %b[%idx_next] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %b_val = ttg.local_load %b_view token %wait : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_RING_OP>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_RING_OP> * tensor<32x128xf16, #B_RING_OP> -> tensor<128x128xf32, #C_RING>
    scf.yield %idx_next, %acc_next : i32, tensor<128x128xf32, #C_RING>
  }
  tt.return %loop#1 : tensor<128x128xf32, #C_RING>
}
}  // end module

// CHECK-LABEL: tt.func @prefetch_pipelined_loop_args
// CHECK-DAG: %[[INIT_WAIT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[A0_SMEM:.+]] = ttg.memdesc_subslice %[[A0:.+]][0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.+]] = ttg.local_load %[[A0_SMEM]] token %[[INIT_WAIT]]
// CHECK-DAG: %[[B0_SMEM:.+]] = ttg.memdesc_subslice %[[B0:.+]][0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.+]] = ttg.local_load %[[B0_SMEM]] token %[[INIT_WAIT]]
// CHECK: %[[LOOP:.+]]:7 = scf.for {{.+}} iter_args(%[[IDX_ARG:.+]] = %[[C0:.+]], %[[A_ARG:.+]] = %[[A0]], %[[B_ARG:.+]] = %[[B0]], %[[WAIT_ARG:.+]] = %[[INIT_WAIT]], %[[ACC_ARG:.+]] = %{{.+}}, %[[A_PREFETCH_ARG:.+]] = %[[A0_PREFETCH]], %[[B_PREFETCH_ARG:.+]] = %[[B0_PREFETCH]])
// CHECK: %[[WAIT_NEXT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[A1_SMEM:.+]] = ttg.memdesc_subslice %[[A_ARG]][0, 16]
// CHECK-DAG: %[[A1:.+]] = ttg.local_load %[[A1_SMEM]] token %[[WAIT_ARG]]
// CHECK-DAG: %[[B1_SMEM:.+]] = ttg.memdesc_subslice %[[B_ARG]][16, 0]
// CHECK-DAG: %[[B1:.+]] = ttg.local_load %[[B1_SMEM]] token %[[WAIT_ARG]]
// CHECK: %{{.+}} = tt.dot %[[A_PREFETCH_ARG]], %[[B_PREFETCH_ARG]], %[[ACC_ARG]]
// CHECK-DAG: %[[NEXT_A_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[NEXT_A_PREFETCH:.+]] = ttg.local_load %[[NEXT_A_SMEM]] token %[[WAIT_NEXT]]
// CHECK-DAG: %[[NEXT_B_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[NEXT_B_PREFETCH:.+]] = ttg.local_load %[[NEXT_B_SMEM]] token %[[WAIT_NEXT]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @prefetch_pipelined_loop_args(%lb : index, %ub : index, %step : index, %tok0 : !ttg.async.token, %tok1 : !ttg.async.token) -> tensor<128x128xf32, #C_RING> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C_RING>
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable>
  %wait0 = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
  %a0 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
  %b0 = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
  %loop:7 = scf.for %iv = %lb to %ub step %step iter_args(%idx = %c0_i32, %a_view = %a0, %b_view = %b0, %wait = %wait0, %tok_a = %tok0, %tok_b = %tok1, %acc = %cst) -> (i32, !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>, !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #C_RING>) {
    %a_val = ttg.local_load %a_view token %wait : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_RING_OP>
    %b_val = ttg.local_load %b_view token %wait : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_RING_OP>
    %idx_p1 = arith.addi %idx, %c1_i32 : i32
    %idx_cmp = arith.cmpi sge, %idx_p1, %c3_i32 : i32
    %idx_next = arith.select %idx_cmp, %c0_i32, %idx_p1 : i32
    %wait_next = ttg.async_wait %tok_a, %tok_b {num = 4 : i32}
    %a_next = ttg.memdesc_index %a[%idx_next] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %b_next = ttg.memdesc_index %b[%idx_next] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_RING_OP> * tensor<32x128xf16, #B_RING_OP> -> tensor<128x128xf32, #C_RING>
    scf.yield %idx_next, %a_next, %b_next, %wait_next, %tok_a, %tok_b, %acc_next : i32, !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>, !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #C_RING>
  }
  tt.return %loop#6 : tensor<128x128xf32, #C_RING>
}
}  // end module

// CHECK-LABEL: tt.func @prefetch_induction_var_source
// CHECK-DAG: %[[LB_I32:.+]] = arith.index_cast %arg0 : index to i32
// CHECK-DAG: %[[INIT_WAIT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[A0_VIEW:.+]] = ttg.memdesc_index %{{.+}}[%[[LB_I32]]]
// CHECK-DAG: %[[A0_HEAD_SMEM:.+]] = ttg.memdesc_subslice %[[A0_VIEW]][0, 0]
// CHECK-DAG: %[[A0_HEAD:.+]] = ttg.local_load %[[A0_HEAD_SMEM]] token %[[INIT_WAIT]]
// CHECK-DAG: %[[B0_VIEW:.+]] = ttg.memdesc_index %{{.+}}[%[[LB_I32]]]
// CHECK-DAG: %[[B0_HEAD_SMEM:.+]] = ttg.memdesc_subslice %[[B0_VIEW]][0, 0]
// CHECK-DAG: %[[B0_HEAD:.+]] = ttg.local_load %[[B0_HEAD_SMEM]] token %[[INIT_WAIT]]
// CHECK: %[[LOOP:.+]]:{{[0-9]+}} = scf.for %[[IV:.+]] = %arg0 to %arg1 step %arg2 iter_args({{.*}}%[[A_PREFETCH_ARG:.+]] = %[[A0_HEAD]], %[[B_PREFETCH_ARG:.+]] = %[[B0_HEAD]])
// CHECK: %[[WAIT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[IV_NEXT:.+]] = arith.addi %[[IV]], %arg2 : index
// CHECK-DAG: %[[IV_NEXT_I32:.+]] = arith.index_cast %[[IV_NEXT]] : index to i32
// CHECK: %[[A1_HEAD_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0] : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x16xf16, #shared, #smem, mutable, 128x32>
// CHECK: %[[A1_HEAD:.+]] = ttg.local_load %[[A1_HEAD_SMEM]] token %{{.+}} : !ttg.memdesc<128x16xf16, #shared, #smem, mutable, 128x32> -> tensor<128x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
// CHECK: %[[B1_HEAD_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0] : !ttg.memdesc<32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<16x128xf16, #shared, #smem, mutable, 32x128>
// CHECK: %[[B1_HEAD:.+]] = ttg.local_load %[[B1_HEAD_SMEM]] token %{{.+}} : !ttg.memdesc<16x128xf16, #shared, #smem, mutable, 32x128> -> tensor<16x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
// CHECK: %{{.+}} = tt.dot %{{.+}}, %{{.+}}, %{{.+}}
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @prefetch_induction_var_source(%lb : index, %ub : index, %step : index, %tok0 : !ttg.async.token, %tok1 : !ttg.async.token) -> tensor<128x128xf32, #C_RING> {
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C_RING>
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable>
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst) -> (tensor<128x128xf32, #C_RING>) {
    %iv_i32 = arith.index_cast %iv : index to i32
    %wait = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
    %a_view = ttg.memdesc_index %a[%iv_i32] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %a_val = ttg.local_load %a_view token %wait : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_RING_OP>
    %b_view = ttg.memdesc_index %b[%iv_i32] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %b_val = ttg.local_load %b_view token %wait : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_RING_OP>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_RING_OP> * tensor<32x128xf16, #B_RING_OP> -> tensor<128x128xf32, #C_RING>
    scf.yield %acc_next : tensor<128x128xf32, #C_RING>
  }
  tt.return %loop : tensor<128x128xf32, #C_RING>
}
}  // end module

// -----

#shared_f64 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 4, order = [1, 0]}>
#shared1_f64 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#mma_f64 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a_f64_op = #ttg.dot_op<{opIdx = 0, parent = #mma_f64, kWidth = 1}>
#b_f64_op = #ttg.dot_op<{opIdx = 1, parent = #mma_f64, kWidth = 1}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @split_pipelined_mmav2_loads_f64
// CHECK-DAG: %[[A_VIEW:.+]] = ttg.memdesc_index %[[A_BUF:.+]][
// CHECK-DAG: %[[WAIT:.+]] = ttg.async_wait
// CHECK-DAG: %[[B_VIEW:.+]] = ttg.memdesc_index %[[B_BUF:.+]][
// CHECK-DAG: %[[A0_SMEM:.+]] = ttg.memdesc_subslice %[[A_VIEW]][0, 0]
// CHECK-DAG: %[[A0:.+]] = ttg.local_load %[[A0_SMEM]] token %[[WAIT]]
// CHECK-DAG: %[[B0_SMEM:.+]] = ttg.memdesc_subslice %[[B_VIEW]][0, 0]
// CHECK-DAG: %[[B0:.+]] = ttg.local_load %[[B0_SMEM]] token %[[WAIT]]
// CHECK: %[[LOOP:.+]]:8 = scf.for {{.+}} iter_args(%[[IDX_ARG:.+]] = %[[C0:.+]], %[[ACC_ARG:.+]] = %{{.+}}, %[[A_VIEW_ARG:.+]] = %[[A_VIEW]], %[[B_VIEW_ARG:.+]] = %[[B_VIEW]], %[[A_WAIT_ARG:.+]] = %[[WAIT]], %[[B_WAIT_ARG:.+]] = %[[WAIT]], %[[A0_ARG:.+]] = %[[A0]], %[[B0_ARG:.+]] = %[[B0]])
// CHECK-DAG: %[[A1_SMEM:.+]] = ttg.memdesc_subslice %[[A_VIEW_ARG]][0, 8]
// CHECK-DAG: %[[A1:.+]] = ttg.local_load %[[A1_SMEM]] token %[[A_WAIT_ARG]]
// CHECK-DAG: %[[B1_SMEM:.+]] = ttg.memdesc_subslice %[[B_VIEW_ARG]][8, 0]
// CHECK-DAG: %[[B1:.+]] = ttg.local_load %[[B1_SMEM]] token %[[B_WAIT_ARG]]
// CHECK: %[[DOT0:.+]] = tt.dot %[[A0_ARG]], %[[B0_ARG]], %[[ACC_ARG]]
// CHECK: tt.dot %[[A1]], %[[B1]], %[[DOT0]]
module attributes {ttg.target = "cuda:90", "ttg.num-warps" = 1 : i32} {
tt.func @split_pipelined_mmav2_loads_f64(%lb : index, %ub : index, %step : index, %tok0 : !ttg.async.token, %tok1 : !ttg.async.token) -> tensor<16x16xf64, #mma_f64> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<16x16xf64, #mma_f64>
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x16x16xf64, #shared_f64, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<3x16x16xf64, #shared1_f64, #smem, mutable>
  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%idx = %c0_i32, %acc = %cst) -> (i32, tensor<16x16xf64, #mma_f64>) {
    %idx_p1 = arith.addi %idx, %c1_i32 : i32
    %idx_cmp = arith.cmpi sge, %idx_p1, %c3_i32 : i32
    %idx_next = arith.select %idx_cmp, %c0_i32, %idx_p1 : i32
    %wait = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
    %a_view = ttg.memdesc_index %a[%idx_next] : !ttg.memdesc<3x16x16xf64, #shared_f64, #smem, mutable> -> !ttg.memdesc<16x16xf64, #shared_f64, #smem, mutable>
    %a_val = ttg.local_load %a_view token %wait : !ttg.memdesc<16x16xf64, #shared_f64, #smem, mutable> -> tensor<16x16xf64, #a_f64_op>
    %b_view = ttg.memdesc_index %b[%idx_next] : !ttg.memdesc<3x16x16xf64, #shared1_f64, #smem, mutable> -> !ttg.memdesc<16x16xf64, #shared1_f64, #smem, mutable>
    %b_val = ttg.local_load %b_view token %wait : !ttg.memdesc<16x16xf64, #shared1_f64, #smem, mutable> -> tensor<16x16xf64, #b_f64_op>
    %acc_next = tt.dot %a_val, %b_val, %acc, inputPrecision = tf32 : tensor<16x16xf64, #a_f64_op> * tensor<16x16xf64, #b_f64_op> -> tensor<16x16xf64, #mma_f64>
    scf.yield %idx_next, %acc_next : i32, tensor<16x16xf64, #mma_f64>
  }
  tt.return %loop#1 : tensor<16x16xf64, #mma_f64>
}
}  // end module

// -----

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#A = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = false}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#B_OP = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @matmul_loop_mixed_amd
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
