// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul --allocate-shared-memory-nv='compute-capability=120' --convert-triton-gpu-to-llvm='compute-capability=120' | FileCheck %s

// The fp4Padded shared encoding is lowered away by convert-to-llvm, so assert
// it at the ttgir stage here.
//
// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s --check-prefix=TTGIR

// Mixed-precision mxfp8 x mxfp4 on sm120 lowers to a kind::mxf8f6f4 block-scaled
// mma.sync. The packed fp4 operand (2 e2m1 per byte, [16,128] i8) is staged
// through fp4Padded shared memory and loaded into a kWidth=2 fp4Unpacked
// register dot operand.
//
// The staged fp4 operand is loaded with the cooperative fp4 ldmatrix mode
// (ldmatrix.m8n16.b8x16.b4x16_p64): the hardware decompresses each packed 4-bit
// e2m1 into bits [3:0] of an 8-bit container. MMAv2 moves each value into bits
// [5:2] immediately before the mixed-precision MMA.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_k = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_mmav2_dot_scaled_mixed
  // CHECK: nvvm.ldmatrix {{.*}}<b8x16.b4x16_p64>{{.*}}num = 4{{.*}}<m = 8, n = 16>
  // CHECK: %[[B_MASK:.+]] = llvm.mlir.constant(252645135 : i32) : i32
  // CHECK: %[[B_MASKED:.+]] = llvm.and %{{.*}}, %[[B_MASK]] : i32
  // CHECK: %[[B_SHIFT:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.shl %[[B_MASKED]], %[[B_SHIFT]] : i32
  // CHECK: mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0
  // The packed fp4 operand %b is staged through fp4Padded shared memory and
  // loaded into a kWidth=2 fp4Unpacked register dot operand (opIdx = 1).
  // TTGIR-DAG: #[[$SHARED:.+]] = #ttg.nvmma_shared<{{.*}}elementBitWidth = 8, fp4Padded = true{{.*}}>
  // TTGIR-LABEL: @sm120_mmav2_dot_scaled_mixed
  // TTGIR: ttg.local_alloc {{.*}}-> !ttg.memdesc<{{.*}}xi8, #[[$SHARED]], #smem>
  // TTGIR: ttg.local_load {{.*}}-> tensor<{{.*}}xi8, #ttg.dot_op<{opIdx = 1, {{.*}}kWidth = 2, fp4Unpacked = true}>>
  tt.func public @sm120_mmav2_dot_scaled_mixed(
    %a: tensor<128x32xf8E4M3FN, #blocked_k>,
    %sa: tensor<128x1xi8, #blocked>,
    %b: tensor<16x128xi8, #blocked>,
    %sb: tensor<128x1xi8, #blocked>,
    %out: !tt.ptr<f32>
  ){
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e4m3 rhs = e2m1 {fastMath = false}
      : tensor<128x32xf8E4M3FN, #blocked_k>, tensor<128x1xi8, #blocked>
        * tensor<16x128xi8, #blocked>, tensor<128x1xi8, #blocked>
        -> tensor<128x128xf32, #blocked>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.broadcast %out_splat : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %zero = arith.constant dense<0> : tensor<128x128xi1, #blocked>
    tt.store %out_ptrs, %d, %zero : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Loading an fp4Padded SMEM source into an fp4Unpacked B register operand is
// legal with transposed=false, but this layout cannot be divided by the
// b4x16_p64 ldmatrix tile. I.e., for operand B, b4x16_p64 requires packed K to
// be contiguous (transposed=true). But with transposed=false N is contiguous
// instead, forcing the legal load to fall back to scalar loads that unpack
// fp4Padded SMEM into fp4Unpacked registers.
//
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8, fp4Padded = true}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_fp4_unpacked_from_padded_smem_scalar_fallback
  // CHECK-NOT: <b8x16.b4x16_p64>
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3> -> {{(i8|vector<[0-9]+xi8>)}}
  // CHECK: llvm.and {{.*}} : i8
  // CHECK: llvm.lshr {{.*}} : i8
  // CHECK: llvm.and {{.*}} : i8
  // CHECK-NOT: <b8x16.b4x16_p64>
  // CHECK: llvm.call @vprintf
  // CHECK: llvm.return
  tt.func @sm120_fp4_unpacked_from_padded_smem_scalar_fallback(
    %src: !ttg.memdesc<16x128xi8, #shared, #smem, mutable>
  ) {
    %0 = ttg.local_load %src
      : !ttg.memdesc<16x128xi8, #shared, #smem, mutable>
        -> tensor<16x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2, fp4Unpacked = true}>>
    tt.print "fp4: " {hex = false, isSigned = array<i32: 0>} : %0 : tensor<16x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2, fp4Unpacked = true}>>
    tt.return
  }
}

// -----

// Mirror case: packed fp4 A is staged through fp4Padded shared memory while
// fp8 B remains in registers.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked_k = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.target" = "cuda:120", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sm120_mmav2_dot_scaled_mixed_fp4_a
  // CHECK: nvvm.ldmatrix {{.*}}<b8x16.b4x16_p64>{{.*}}num = 4{{.*}}<m = 8, n = 16>
  // CHECK: %[[A_MASK:.+]] = llvm.mlir.constant(252645135 : i32) : i32
  // CHECK: %[[A_MASKED:.+]] = llvm.and %{{.*}}, %[[A_MASK]] : i32
  // CHECK: %[[A_SHIFT:.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.shl %[[A_MASKED]], %[[A_SHIFT]] : i32
  // CHECK: mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0
  // TTGIR-DAG: #[[$SHARED_A:.+]] = #ttg.nvmma_shared<{{.*}}elementBitWidth = 8, fp4Padded = true{{.*}}>
  // TTGIR-LABEL: @sm120_mmav2_dot_scaled_mixed_fp4_a
  // TTGIR: ttg.local_alloc {{.*}}-> !ttg.memdesc<{{.*}}xi8, #[[$SHARED_A]], #smem>
  // TTGIR: ttg.local_load {{.*}}-> tensor<{{.*}}xi8, #ttg.dot_op<{opIdx = 0, {{.*}}kWidth = 2, fp4Unpacked = true}>>
  tt.func public @sm120_mmav2_dot_scaled_mixed_fp4_a(
    %a: tensor<128x16xi8, #blocked>,
    %sa: tensor<128x1xi8, #blocked>,
    %b: tensor<32x128xf8E4M3FN, #blocked_k>,
    %sb: tensor<128x1xi8, #blocked>,
    %out: !tt.ptr<f32>
  ){
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %d = tt.dot_scaled %a scale %sa, %b scale %sb, %c lhs = e2m1 rhs = e4m3 {fastMath = false}
      : tensor<128x16xi8, #blocked>, tensor<128x1xi8, #blocked>
        * tensor<32x128xf8E4M3FN, #blocked_k>, tensor<128x1xi8, #blocked>
        -> tensor<128x128xf32, #blocked>
    %out_splat = tt.splat %out : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %out_ptrs = tt.broadcast %out_splat : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x128x!tt.ptr<f32>, #blocked>
    %zero = arith.constant dense<0> : tensor<128x128xi1, #blocked>
    tt.store %out_ptrs, %d, %zero : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
