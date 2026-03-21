// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx1250 | FileCheck %s

// Test buffer atomic RMW fadd with f32 on gfx1250
// Verifies correct cache policy with SCOPE_DEV and fence generation

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_atomic_rmw_fadd_f32
  tt.func @buffer_atomic_rmw_fadd_f32(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset : tensor<128xi32, #blocked>{tt.divisibility=16:i32}, %values : tensor<128xf32, #blocked>) {
    // CHECK: rocdl.make.buffer.rsrc
    // There should be a single release fence before any atomics
    // CHECK: llvm.fence syncscope("agent") release

    // sizePerThread=4 => 4 atomic fadd calls
    // Cache policy = 16 (SCOPE_DEV only, no SC0 since return value is unused)
    // CHECK: llvm.mlir.constant(16 : i32)
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (f32, !llvm.ptr<8>, i32, i32, i32) -> f32

    // There should be a single acquire fence after all of the atomics
    // CHECK: llvm.fence syncscope("agent") acquire
    %ret = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %arg0[%offset] : tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// Test buffer atomic CAS with i32 on gfx1250
// Verifies correct resource descriptor creation, fence generation,
// and cache policy with SCOPE_DEV

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_atomic_cas_i32
  tt.func public @buffer_atomic_cas_i32(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %val = arith.constant dense<2> : tensor<256xi32, #blocked>
    %cmp = arith.constant dense<0> : tensor<256xi32, #blocked>
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %scalar_ptr = tt.addptr %arg0, %1 : !tt.ptr<i32>, i32

    // CHECK: rocdl.make.buffer.rsrc
    // CHECK: llvm.fence syncscope("agent") release
    // Cache policy = 17 (SC0 | SCOPE_DEV) because CAS return value is used
    // CHECK: llvm.mlir.constant(17 : i32)
    // CHECK: rocdl.raw.ptr.buffer.atomic.cmpswap {{.*}} : i32
    // CHECK: rocdl.raw.ptr.buffer.atomic.cmpswap {{.*}} : i32
    // CHECK: llvm.fence syncscope("agent") acquire
    %4 = amdg.buffer_atomic_cas acq_rel, gpu, %cmp, %val, %scalar_ptr[%offsets] : tensor<256xi32, #blocked>

    %5 = tt.addptr %arg1, %1 : !tt.ptr<i32>, i32
    amdg.buffer_store %4, %5[%offsets] : tensor<256xi32, #blocked>
    tt.return
  }
}

// -----

// Test buffer atomic RMW fadd with bf16 on gfx1250
// gfx1250 supports BUFFER_ATOMIC_PK_ADD_BF16 (packed bf16 fadd)
// Offsets must be contiguous (via tt.make_range) so axis analysis
// computes vec >= 2, which is required for packed v2bf16 atomics.

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_atomic_rmw_fadd_bf16
  tt.func @buffer_atomic_rmw_fadd_bf16(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %values : tensor<64xbf16, #blocked>) {
    %offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>

    // CHECK: llvm.fence syncscope("agent") release

    // sizePerThread=2, bf16 => packed v2bf16 atomic fadd
    // Cache policy = 16 (SCOPE_DEV only, no SC0 since return value is unused)
    // CHECK: llvm.mlir.constant(16 : i32)
    // CHECK: llvm.call_intrinsic "llvm.amdgcn.raw.ptr.buffer.atomic.fadd"({{.*}}) : (vector<2xbf16>, !llvm.ptr<8>, i32, i32, i32) -> vector<2xbf16>

    // CHECK: llvm.fence syncscope("agent") acquire
    %ret = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %values, %arg0[%offsets] : tensor<64xbf16, #blocked>
    tt.return
  }
}
