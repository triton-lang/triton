// RUN: triton-opt %s -split-input-file -tritongpu-wgmma-prefetch -canonicalize | FileCheck %s

// matmul: 128x64 @ 64x256
#AL = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[C48:.+]] = arith.constant 48 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}}, %[[arg_a_token:.*]] = {{.*}}, %[[arg_b_token:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[WAIT_TOKEN:.*]] = ttg.async_wait %[[arg_a_token]], %[[arg_b_token]] {num = 0 : i32}
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]] token %[[WAIT_TOKEN]]
// CHECK:       %[[A_SUBTILE_2_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C16]]]
// CHECK:       %[[A_SUBITLE_2_REG:.*]] = ttg.local_load %[[A_SUBTILE_2_SMEM]]
// CHECK:       %[[A_SUBTILE_3_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C32]]]
// CHECK:       %[[A_SUBITLE_3_REG:.*]] = ttg.local_load %[[A_SUBTILE_3_SMEM]]
// CHECK:       %[[A_SUBTILE_4_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C48]]]
// CHECK:       %[[A_SUBITLE_4_REG:.*]] = ttg.local_load %[[A_SUBTILE_4_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM:.*]], %[[D_arg]]
// CHECK:       %[[A_SUBITLE_2_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_2_REG]]
// CHECK:       %[[D_2:.*]] = ttng.warp_group_dot %[[A_SUBITLE_2_REG_CVT]], %[[B_SUBTILE_2_SMEM:.*]], %[[D_1]], %[[TRUE]]
// CHECK:       %[[A_SUBITLE_3_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_3_REG]]
// CHECK:       %[[D_3:.*]] = ttng.warp_group_dot %[[A_SUBITLE_3_REG_CVT]], %[[B_SUBTILE_3_SMEM:.*]], %[[D_2]], %[[TRUE]]
// CHECK:       %[[A_SUBITLE_4_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_4_REG]]
// CHECK:       %[[D_4:.*]] = ttng.warp_group_dot %[[A_SUBITLE_4_REG_CVT]], %[[B_SUBTILE_4_SMEM:.*]], %[[D_3]], %[[TRUE]]
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_4]]
// CHECK-NEXT:  %[[D_ADD_CONSTANT:.*]] = arith.addf %[[D]]
// CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D_ADD_CONSTANT]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<bf16>) -> tensor<128x256xf32, #C> {
     %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<128x64x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x64xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<128x64xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<64x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<64x256xbf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x256xf32, #C>
    %constant_init = arith.constant dense<1.00e+00> : tensor<128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x64xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<64x256xi32, #BL>

    %a_init = ttg.local_alloc : () -> !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>
    %a_ = ttg.async_copy_global_to_local %a_ptr_init, %a_init mask %a_mask other %a_other : tensor<128x64x!tt.ptr<i8>, #AL> -> !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>
    %a_load_commit = ttg.async_commit_group %a_

    %b_init = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>
    %b_ = ttg.async_copy_global_to_local %b_ptr_init, %b_init mask %b_mask other %b_other : tensor<64x256x!tt.ptr<bf16>, #BL> -> !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>
    %b_load_commit = ttg.async_commit_group %b_

    %loop:7 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init, %a_token = %a_load_commit, %b_token = %b_load_commit) -> (tensor<128x64x!tt.ptr<i8>, #AL>, tensor<64x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>, !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>, tensor<128x256xf32, #C>, !ttg.async.token, !ttg.async.token) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32] : !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable> -> !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32] : !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>
        %wait_token = ttg.async_wait %a_token, %b_token {num = 0 : i32}

        %a_op_ = ttg.local_load %a_smem_tile token  %wait_token : !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable> -> tensor<128x64xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<128x64xi8, #A_OP> to tensor<128x64xbf16, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, isAsync = true} : tensor<128x64xbf16, #A_OP> * !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable> -> tensor<128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<128x64x!tt.ptr<i8>, #AL>, tensor<128x64xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<64x256x!tt.ptr<bf16>, #BL>, tensor<64x256xi32, #BL>

        %c_add_constant = arith.addf %c, %constant_init: tensor<128x256xf32, #C>

        %next_a = ttg.local_alloc : () -> !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>
        %next_a_ = ttg.async_copy_global_to_local %next_a_ptr, %next_a mask %a_mask other %a_other : tensor<128x64x!tt.ptr<i8>, #AL> ->  !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>
        %next_a_load_commit = ttg.async_commit_group %next_a_
        %next_b = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>
        %next_b_ = ttg.async_copy_global_to_local %next_b_ptr, %next_b mask %b_mask other %b_other : tensor<64x256x!tt.ptr<bf16>, #BL> -> !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>
        %next_b_load_commit = ttg.async_commit_group %next_b_

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c_add_constant, %next_a_load_commit, %next_b_load_commit: tensor<128x64x!tt.ptr<i8>, #AL>, tensor<64x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<128x64xi8, #A_SMEM, #smem, mutable>, !ttg.memdesc<64x256xbf16, #B_SMEM, #smem, mutable>, tensor<128x256xf32, #C>, !ttg.async.token, !ttg.async.token

    }
    tt.return %loop#4 : tensor<128x256xf32, #C>
   }
}

// -----

// matmul: 128x16 @ 16x256
#AL = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[B_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[B_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM]], %[[D_arg]]
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_1]]
// CHECK-NEXT: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<bf16>) -> tensor<128x256xf32, #C> {
     %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<128x16x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<bf16> -> tensor<16x256x!tt.ptr<bf16>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x16xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<128x16xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<16x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<16x256xbf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x16xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<16x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x16x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<128x16xi8, #AL>) -> !ttg.memdesc<128x16xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<16x256x!tt.ptr<bf16>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<16x256xbf16, #BL>) -> !ttg.memdesc<16x256xbf16, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x16x!tt.ptr<i8>, #AL>, tensor<16x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<128x16xi8, #A_SMEM, #smem>, !ttg.memdesc<16x256xbf16, #B_SMEM, #smem>, tensor<128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32] : !ttg.memdesc<128x16xi8, #A_SMEM, #smem> -> !ttg.memdesc<128x16xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32] : !ttg.memdesc<16x256xbf16, #B_SMEM, #smem> -> !ttg.memdesc<16x256xbf16, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<128x16xi8, #A_SMEM, #smem> -> tensor<128x16xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<128x16xi8, #A_OP> to tensor<128x16xbf16, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, isAsync = true} : tensor<128x16xbf16, #A_OP> * !ttg.memdesc<16x256xbf16, #B_SMEM, #smem> -> tensor<128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<128x16x!tt.ptr<i8>, #AL>, tensor<128x16xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<16x256x!tt.ptr<bf16>, #BL>, tensor<16x256xi32, #BL>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x16x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<128x16xi8, #AL>) -> !ttg.memdesc<128x16xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<16x256x!tt.ptr<bf16>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<16x256xbf16, #BL>) -> !ttg.memdesc<16x256xbf16, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c: tensor<128x16x!tt.ptr<i8>, #AL>, tensor<16x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<128x16xi8, #A_SMEM, #smem>, !ttg.memdesc<16x256xbf16, #B_SMEM, #smem>, tensor<128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<128x256xf32, #C>
   }
}

// -----

// matmul: 8x128x16 @ 8x16x256
#AL = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 8], order = [2, 0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [2, 1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[B_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[B_TILE_SMEM]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM]], %[[D_arg]]
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_1]]
// CHECK:     scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<bf16>) -> tensor<8x128x256xf32, #C> {
     %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<8x128x16x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<bf16> -> tensor<8x16x256x!tt.ptr<bf16>, #BL>

    %a_mask = arith.constant dense<true> : tensor<8x128x16xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<8x128x16xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<8x16x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<8x16x256xbf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<8x128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<8x128x16xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<8x16x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<8x128x16xi8, #AL>) -> !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x16x256x!tt.ptr<bf16>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<8x16x256xbf16, #BL>) -> !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x16x!tt.ptr<i8>, #AL>, tensor<8x16x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem>, !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem>, tensor<8x128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem> -> !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem> -> !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem> -> tensor<8x128x16xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<8x128x16xi8, #A_OP> to tensor<8x128x16xbf16, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, isAsync = true} : tensor<8x128x16xbf16, #A_OP> * !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem> -> tensor<8x128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<8x128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<8x128x16x!tt.ptr<i8>, #AL>, tensor<8x128x16xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<8x16x256x!tt.ptr<bf16>, #BL>, tensor<8x16x256xi32, #BL>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x16x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x16xi8, #AL>) -> !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x16x256x!tt.ptr<bf16>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<8x16x256xbf16, #BL>) -> !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c: tensor<8x128x16x!tt.ptr<i8>, #AL>, tensor<8x16x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<8x128x16xi8, #A_SMEM, #smem>, !ttg.memdesc<8x16x256xbf16, #B_SMEM, #smem>, tensor<8x128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<8x128x256xf32, #C>
   }
}

// -----

// matmul: 8x128x64 @ 8x64x256
#AL = #ttg.blocked<{sizePerThread = [1, 1, 16], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 8, 1], order = [2, 1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 8, 4], warpsPerCTA = [1, 1, 8], order = [2, 0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [2, 1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 8, 1], instrShape = [16, 256, 16]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C16:.+]] = arith.constant 16 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[C48:.+]] = arith.constant 48 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBTILE_2_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]], %[[C16]]]
// CHECK:       %[[A_SUBITLE_2_REG:.*]] = ttg.local_load %[[A_SUBTILE_2_SMEM]]
// CHECK:       %[[A_SUBTILE_3_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]], %[[C32]]]
// CHECK:       %[[A_SUBITLE_3_REG:.*]] = ttg.local_load %[[A_SUBTILE_3_SMEM]]
// CHECK:       %[[A_SUBTILE_4_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]], %[[C48]]]
// CHECK:       %[[A_SUBITLE_4_REG:.*]] = ttg.local_load %[[A_SUBTILE_4_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM:.*]], %[[D_arg]]
// CHECK:       %[[A_SUBITLE_2_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_2_REG]]
// CHECK:       %[[D_2:.*]] = ttng.warp_group_dot %[[A_SUBITLE_2_REG_CVT]], %[[B_SUBTILE_2_SMEM:.*]], %[[D_1]], %[[TRUE]]
// CHECK:       %[[A_SUBITLE_3_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_3_REG]]
// CHECK:       %[[D_3:.*]] = ttng.warp_group_dot %[[A_SUBITLE_3_REG_CVT]], %[[B_SUBTILE_3_SMEM:.*]], %[[D_2]], %[[TRUE]]
// CHECK:       %[[A_SUBITLE_4_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_4_REG]]
// CHECK:       %[[D_4:.*]] = ttng.warp_group_dot %[[A_SUBITLE_4_REG_CVT]], %[[B_SUBTILE_4_SMEM:.*]], %[[D_3]], %[[TRUE]]
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_4]]
// CHECK-NEXT: scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<bf16>) -> tensor<8x128x256xf32, #C> {
     %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<8x128x64x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<bf16> -> tensor<8x64x256x!tt.ptr<bf16>, #BL>

    %a_mask = arith.constant dense<true> : tensor<8x128x64xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<8x128x64xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<8x64x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<8x64x256xbf16, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<8x128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<8x128x64xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<8x64x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<8x128x64x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<8x128x64xi8, #AL>) -> !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<8x64x256x!tt.ptr<bf16>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<8x64x256xbf16, #BL>) -> !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<8x128x64x!tt.ptr<i8>, #AL>, tensor<8x64x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem>, !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem>, tensor<8x128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem> -> !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem> -> !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem> -> tensor<8x128x64xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<8x128x64xi8, #A_OP> to tensor<8x128x64xbf16, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, isAsync = true} : tensor<8x128x64xbf16, #A_OP> * !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem> -> tensor<8x128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<8x128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<8x128x64x!tt.ptr<i8>, #AL>, tensor<8x128x64xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<8x64x256x!tt.ptr<bf16>, #BL>, tensor<8x64x256xi32, #BL>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<8x128x64x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<8x128x64xi8, #AL>) -> !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<8x64x256x!tt.ptr<bf16>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<8x64x256xbf16, #BL>) -> !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c: tensor<8x128x64x!tt.ptr<i8>, #AL>, tensor<8x64x256x!tt.ptr<bf16>, #BL>, !ttg.memdesc<8x128x64xi8, #A_SMEM, #smem>, !ttg.memdesc<8x64x256xbf16, #B_SMEM, #smem>, tensor<8x128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<8x128x256xf32, #C>
   }
}

// -----

// matmul: 128x128 @ 128x256, test for maxNumImpreciseAcc
#AL = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 4}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : i32
// CHECK-DAG: %[[C96:.+]] = arith.constant 96 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBTILE_2_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C32]]]
// CHECK:       %[[A_SUBITLE_2_REG:.*]] = ttg.local_load %[[A_SUBTILE_2_SMEM]]
// CHECK:       %[[A_SUBTILE_3_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C64]]]
// CHECK:       %[[A_SUBITLE_3_REG:.*]] = ttg.local_load %[[A_SUBTILE_3_SMEM]]
// CHECK:       %[[A_SUBTILE_4_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C96]]]
// CHECK:       %[[A_SUBITLE_4_REG:.*]] = ttg.local_load %[[A_SUBTILE_4_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM:.*]], %[[D_arg]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_2_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_2_REG]]
// CHECK:       %[[D_2:.*]] = ttng.warp_group_dot %[[A_SUBITLE_2_REG_CVT]], %[[B_SUBTILE_2_SMEM:.*]], %[[D_1]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_3_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_3_REG]]
// CHECK:       %[[D_3:.*]] = ttng.warp_group_dot %[[A_SUBITLE_3_REG_CVT]], %[[B_SUBTILE_3_SMEM:.*]], %[[D_2]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_4_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_4_REG]]
// CHECK:       %[[D_4:.*]] = ttng.warp_group_dot %[[A_SUBITLE_4_REG_CVT]], %[[B_SUBTILE_4_SMEM:.*]], %[[D_3]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_4]]
// CHECK-NEXT:  %[[D_ADD_CONSTANT:.*]] = arith.addf %[[D]]
// CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D_ADD_CONSTANT]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<f8E5M2>) -> tensor<128x256xf32, #C> {
    %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x128xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<128x128xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<128x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<128x256xf8E5M2, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x256xf32, #C>
    %constant_init = arith.constant dense<1.00e+00> : tensor<128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x128xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<128x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32] : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32] : !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> tensor<128x128xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<128x128xi8, #A_OP> to tensor<128x128xf8E5M2, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, maxNumImpreciseAcc = 129 : i32, isAsync = true} : tensor<128x128xf8E5M2, #A_OP> * !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> tensor<128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x128xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<128x256x!tt.ptr<f8E5M2>, #BL>, tensor<128x256xi32, #BL>

        %c_add_constant = arith.addf %c, %constant_init: tensor<128x256xf32, #C>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c_add_constant: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<128x256xf32, #C>
   }
}
// -----

// matmul: 128x128 @ 128x256, test for maxNumImpreciseAcc
#AL = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 4}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : i32
// CHECK-DAG: %[[C96:.+]] = arith.constant 96 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBTILE_2_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C32]]]
// CHECK:       %[[A_SUBITLE_2_REG:.*]] = ttg.local_load %[[A_SUBTILE_2_SMEM]]
// CHECK:       %[[A_SUBTILE_3_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C64]]]
// CHECK:       %[[A_SUBITLE_3_REG:.*]] = ttg.local_load %[[A_SUBTILE_3_SMEM]]
// CHECK:       %[[A_SUBTILE_4_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C96]]]
// CHECK:       %[[A_SUBITLE_4_REG:.*]] = ttg.local_load %[[A_SUBTILE_4_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM:.*]], %[[D_arg]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_2_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_2_REG]]
// CHECK:       %[[D_2:.*]] = ttng.warp_group_dot %[[A_SUBITLE_2_REG_CVT]], %[[B_SUBTILE_2_SMEM:.*]], %[[D_1]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_3_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_3_REG]]
// CHECK:       %[[D_3:.*]] = ttng.warp_group_dot %[[A_SUBITLE_3_REG_CVT]], %[[B_SUBTILE_3_SMEM:.*]], %[[D_2]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_4_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_4_REG]]
// CHECK:       %[[D_4:.*]] = ttng.warp_group_dot %[[A_SUBITLE_4_REG_CVT]], %[[B_SUBTILE_4_SMEM:.*]], %[[D_3]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 32 : i32}
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_4]]
// CHECK-NEXT:  %[[D_ADD_CONSTANT:.*]] = arith.addf %[[D]]
// CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D_ADD_CONSTANT]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<f8E5M2>) -> tensor<128x256xf32, #C> {
    %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x128xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<128x128xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<128x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<128x256xf8E5M2, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x256xf32, #C>
    %constant_init = arith.constant dense<1.00e+00> : tensor<128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x128xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<128x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32] : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32] : !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> tensor<128x128xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<128x128xi8, #A_OP> to tensor<128x128xf8E5M2, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, maxNumImpreciseAcc = 128 : i32, isAsync = true} : tensor<128x128xf8E5M2, #A_OP> * !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> tensor<128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x128xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<128x256x!tt.ptr<f8E5M2>, #BL>, tensor<128x256xi32, #BL>

        %c_add_constant = arith.addf %c, %constant_init: tensor<128x256xf32, #C>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c_add_constant: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<128x256xf32, #C>
   }
}

// -----

// matmul: 128x128 @ 128x256, test for maxNumImpreciseAcc
#AL = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#A_SMEM = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 32]}>
#B_SMEM = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#A_OP = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth = 4}>
#smem = #ttg.shared_memory

// CHECK: tt.func @wgmma_mixed_precision
// CHECK-DAG: %[[TRUE:.+]] = arith.constant true
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[C32:.+]] = arith.constant 32 : i32
// CHECK-DAG: %[[C64:.+]] = arith.constant 64 : i32
// CHECK-DAG: %[[C96:.+]] = arith.constant 96 : i32
// CHECK:     scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[arg_a0:.*]] = {{.*}}, %[[arg_b0:.*]] = {{.*}}, %[[D_arg:.*]] = {{.*}})
// CHECK-DAG:   %[[A_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_a0]][%[[C0]], %[[C0]]]
// CHECK-DAG:   %[[B_TILE_SMEM:.*]] = ttg.memdesc_subview %[[arg_b0]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBTILE_1_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C0]]]
// CHECK:       %[[A_SUBITLE_1_REG:.*]] = ttg.local_load %[[A_SUBTILE_1_SMEM]]
// CHECK:       %[[A_SUBTILE_2_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C32]]]
// CHECK:       %[[A_SUBITLE_2_REG:.*]] = ttg.local_load %[[A_SUBTILE_2_SMEM]]
// CHECK:       %[[A_SUBTILE_3_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C64]]]
// CHECK:       %[[A_SUBITLE_3_REG:.*]] = ttg.local_load %[[A_SUBTILE_3_SMEM]]
// CHECK:       %[[A_SUBTILE_4_SMEM:.*]] = ttg.memdesc_subview %[[A_TILE_SMEM]][%[[C0]], %[[C96]]]
// CHECK:       %[[A_SUBITLE_4_REG:.*]] = ttg.local_load %[[A_SUBTILE_4_SMEM]]
// CHECK:       %[[A_SUBITLE_1_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_1_REG]]
// CHECK:       %[[D_1:.*]] = ttng.warp_group_dot %[[A_SUBITLE_1_REG_CVT]], %[[B_SUBTILE_1_SMEM:.*]], %[[D_arg]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_2_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_2_REG]]
// CHECK:       %[[D_2:.*]] = ttng.warp_group_dot %[[A_SUBITLE_2_REG_CVT]], %[[B_SUBTILE_2_SMEM:.*]], %[[D_1]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 1073741824 : i32}
// CHECK:       %[[A_SUBITLE_3_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_3_REG]]
// CHECK:       %[[D_3:.*]] = ttng.warp_group_dot %[[A_SUBITLE_3_REG_CVT]], %[[B_SUBTILE_3_SMEM:.*]], %[[D_2]], %[[TRUE]] {{{.*}}, {{.*}}, maxNumImpreciseAcc = 32 : i32}
// CHECK:       %[[A_SUBITLE_4_REG_CVT:.*]] = arith.sitofp %[[A_SUBITLE_4_REG]]
// CHECK:       %[[D_4:.*]] = ttng.warp_group_dot %[[A_SUBITLE_4_REG_CVT]], %[[B_SUBTILE_4_SMEM:.*]], %[[D_3]], %[[TRUE]] {{{.*}}, {{.*}}}
// CHECK-DAG:   %[[D:.*]] = ttng.warp_group_dot_wait %[[D_4]]
// CHECK-NEXT:  %[[D_ADD_CONSTANT:.*]] = arith.addf %[[D]]
// CHECK:       scf.yield {{.*}}, {{.*}}, {{.*}}, {{.*}}, %[[D_ADD_CONSTANT]]

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
    tt.func @wgmma_mixed_precision(%lb : index, %ub : index, %step : index, %A : !tt.ptr<i8>, %B : !tt.ptr<f8E5M2>) -> tensor<128x256xf32, #C> {
    %c0_i32 = arith.constant 0 : i32

    %a_ptr_init = tt.splat %A : !tt.ptr<i8> -> tensor<128x128x!tt.ptr<i8>, #AL>
    %b_ptr_init = tt.splat %B : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #BL>

    %a_mask = arith.constant dense<true> : tensor<128x128xi1, #AL>
    %a_other = arith.constant dense<0> : tensor<128x128xi8, #AL>
    %b_mask = arith.constant dense<true> : tensor<128x256xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<128x256xf8E5M2, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x256xf32, #C>
    %constant_init = arith.constant dense<1.00e+00> : tensor<128x256xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x128xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<128x256xi32, #BL>

    %a_ = tt.load %a_ptr_init, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
    %a_init = ttg.local_alloc %a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
    %b_ = tt.load %b_ptr_init, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
    %b_init = ttg.local_alloc %b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

    %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %a = %a_init, %b = %b_init, %prev_c = %c_init) -> (tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>) {
        %a_smem_tile = ttg.memdesc_subview %a[%c0_i32, %c0_i32] : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %b_smem_tile = ttg.memdesc_subview %b[%c0_i32, %c0_i32] : !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>
        %a_op_ = ttg.local_load %a_smem_tile : !ttg.memdesc<128x128xi8, #A_SMEM, #smem> -> tensor<128x128xi8, #A_OP>
        %a_op = arith.sitofp %a_op_ :  tensor<128x128xi8, #A_OP> to tensor<128x128xf8E5M2, #A_OP>
        %c_ = ttng.warp_group_dot %a_op, %b_smem_tile, %prev_c  {inputPrecision = 0 : i32, maxNumImpreciseAcc = 96 : i32, isAsync = true} : tensor<128x128xf8E5M2, #A_OP> * !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem> -> tensor<128x256xf32, #C>

        %c =  ttng.warp_group_dot_wait %c_ {pendings = 0 : i32}:  tensor<128x256xf32, #C>

        %next_a_ptr = tt.addptr %a_ptr, %a_off: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x128xi32, #AL>
        %next_b_ptr = tt.addptr %b_ptr, %b_off: tensor<128x256x!tt.ptr<f8E5M2>, #BL>, tensor<128x256xi32, #BL>

        %c_add_constant = arith.addf %c, %constant_init: tensor<128x256xf32, #C>

        %next_a_ = tt.load %next_a_ptr, %a_mask, %a_other : tensor<128x128x!tt.ptr<i8>, #AL>
        %next_a = ttg.local_alloc %next_a_ : (tensor<128x128xi8, #AL>) -> !ttg.memdesc<128x128xi8, #A_SMEM, #smem>
        %next_b_ = tt.load %next_b_ptr, %b_mask, %b_other : tensor<128x256x!tt.ptr<f8E5M2>, #BL>
        %next_b = ttg.local_alloc %next_b_ : (tensor<128x256xf8E5M2, #BL>) -> !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>

        scf.yield %next_a_ptr, %next_b_ptr, %next_a, %next_b, %c_add_constant: tensor<128x128x!tt.ptr<i8>, #AL>, tensor<128x256x!tt.ptr<f8E5M2>, #BL>, !ttg.memdesc<128x128xi8, #A_SMEM, #smem>, !ttg.memdesc<128x256xf8E5M2, #B_SMEM, #smem>, tensor<128x256xf32, #C>

    }
    tt.return %loop#4 : tensor<128x256xf32, #C>
   }
}
