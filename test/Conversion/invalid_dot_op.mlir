// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -verify-diagnostics

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#mma0 = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #mma0, kWidth = 1}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #mma0, kWidth = 1}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func public @test_mmav2_dot_unsupported_type(%A: tensor<16x4xf64, #blocked0>, %B: tensor<4x16xf64, #blocked0>) -> tensor<16x16xi32, #mma0> {
    %AA = ttg.local_alloc %A : (tensor<16x4xf64, #blocked0>) -> !ttg.memdesc<16x4xf64, #shared0, #smem>
    %BB = ttg.local_alloc %B : (tensor<4x16xf64, #blocked0>) -> !ttg.memdesc<4x16xf64, #shared0, #smem>
    %AA_DOT = ttg.local_load %AA : !ttg.memdesc<16x4xf64, #shared0, #smem> -> tensor<16x4xf64, #dot_operand_a>
    %BB_DOT = ttg.local_load %BB : !ttg.memdesc<4x16xf64, #shared0, #smem> -> tensor<4x16xf64, #dot_operand_b>
    %cst0 = arith.constant dense<0> : tensor<16x16xi32, #mma0>
    // expected-error@+2 {{unsupported MMA instruction for the given operand/result types}}
    // expected-error@+1 {{failed to legalize operation 'tt.dot' that was explicitly marked illegal}}
    %D = tt.dot %AA_DOT, %BB_DOT, %cst0 : tensor<16x4xf64, #dot_operand_a> * tensor<4x16xf64, #dot_operand_b> -> tensor<16x16xi32, #mma0>
    tt.return %D : tensor<16x16xi32, #mma0>
  }
}

// -----

#a = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#d = #ttng.tensor_memory_encoding<blockM = 128, blockN = 8, colStride = 1>
#b_mn = #ttg.shared_linear<{offset = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]]}, alignment = 16>
!a = !ttg.memdesc<128x128xf8E4M3FN, #a, #ttng.tensor_memory>
!b = !ttg.memdesc<128x1xf8E4M3FN, #b_mn, #ttg.shared_memory>
!d = !ttg.memdesc<128x1xf32, #d, #ttng.tensor_memory, mutable>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @mma_requires_contiguous_rhs_atom(%a: !a, %b: !b, %d: !d, %p: i1) {
    // expected-error @below {{instruction M/N must cover at least 16 contiguous elements of the shared-memory core}}
    // expected-error @below {{failed to find valid tcgen05.mma layout for operand B in shared memory}}
    // expected-error @below {{failed to legalize operation 'ttng.tc_gen5_mma' that was explicitly marked illegal}}
    ttng.tc_gen5_mma %a, %b, %d, %p, %p : !a, !b, !d
    tt.return
  }
}
