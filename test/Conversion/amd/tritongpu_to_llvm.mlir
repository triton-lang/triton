// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.store
    // CHECK: llvm.br
    // CHECK: rocdl.barrier
    // CHECK: llvm.load
    // CHECK: llvm.intr.masked.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.intr.masked.store
    // CHECK: llvm.intr.masked.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

// Smoke test to check that mfma 32 and dot operand layouts can work with small tensors, for example with shape 16x16
#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth=4}>
#dotop1 = #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth=4}>
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], hasLeadingOffset = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: small_mfma_tensor_conversions
  tt.func public @small_mfma_tensor_conversions(%arg0: tensor<16x16xf16, #mfma>, %arg1: tensor<16x16x!tt.ptr<f32>, #mfma>) {
    // CHECK-NOT: ttg.convert_layout
    %0 = ttg.local_alloc %arg0 : (tensor<16x16xf16, #mfma>) -> !ttg.memdesc<16x16xf16, #shared, #ttg.shared_memory>
    // CHECK-4: store {{.*}} vector<4xf16>
    %1 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #ttg.shared_memory> -> tensor<16x16xf16, #dotop0>
    // CHECK-2: load {{.*}} vector<4xf16>
    %2 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #ttg.shared_memory> -> tensor<16x16xf16, #dotop1>
    // CHECK-8: load {{.*}} vector<1xf16>
    %3 = ttg.local_load %0 : !ttg.memdesc<16x16xf16, #shared, #ttg.shared_memory> -> tensor<16x16xf16, #mfma>
    // CHECK-4: load {{.*}} vector<4xf16>
    %4 = tt.fp_to_fp %3 : tensor<16x16xf16, #mfma> -> tensor<16x16xf32, #mfma>

    %5 = tt.dot %1, %2, %4 : tensor<16x16xf16, #dotop0> * tensor<16x16xf16, #dotop1> -> tensor<16x16xf32, #mfma>
    // Store result to prevent DCE from removing all conversion related code
    %6 = ttg.local_alloc %5 : (tensor<16x16xf32, #mfma>) -> !ttg.memdesc<16x16xf32, #shared, #ttg.shared_memory>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f16x2
  tt.func @atomic_add_f16x2(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked1>, %arg2 : tensor<256xf16, #blocked1>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %base_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked1>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xi32, #blocked1>
    // CHECK: llvm.cond_br
    // CHECK-NOT: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xf16>
    // CHECK-NOT: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xf16, #blocked1>, tensor<256xi1, #blocked1>) -> tensor<256xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_bf16x2
  tt.func @atomic_add_bf16x2(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked2>, %arg2 : tensor<256xbf16, #blocked2>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %base_ptr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked2>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xi32, #blocked2>
    // CHECK: llvm.cond_br
    // CHECK-NOT: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xbf16>
    // CHECK-NOT: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xbf16, #blocked2>, tensor<256xi1, #blocked2>) -> tensor<256xbf16, #blocked2>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f16_dpp
  tt.func @atomic_add_f16_dpp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked1>, %arg2 : tensor<256xf16, #blocked1>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked1>
    %base_ptr = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked1>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xi32, #blocked1>
    // CHECK: llvm.cond_br
    // CHECK: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xf16>
    // CHECK: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<f16>, #blocked1>, tensor<256xf16, #blocked1>, tensor<256xi1, #blocked1>) -> tensor<256xf16, #blocked1>
    tt.return
  }
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_bf16_dpp
  tt.func @atomic_add_bf16_dpp(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1 : tensor<256xi1, #blocked2>, %arg2 : tensor<256xbf16, #blocked2>) {
    %range = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked2>
    %base_ptr = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked2>
    %ptr = tt.addptr %base_ptr, %range : tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xi32, #blocked2>
    // CHECK: llvm.cond_br
    // CHECK: rocdl.update.dpp
    // CHECK: llvm.atomicrmw fadd {{.*}} vector<2xbf16>
    // CHECK: rocdl.update.dpp
    %0 =  tt.atomic_rmw fadd, relaxed, gpu, %ptr, %arg2, %arg1 : (tensor<256x!tt.ptr<bf16>, #blocked2>, tensor<256xbf16, #blocked2>, tensor<256xi1, #blocked2>) -> tensor<256xbf16, #blocked2>
    tt.return
  }
}

// -----

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<64xf32, #blocked3>) {
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 274, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 273, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 322, 10, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK-NEXT: rocdl.update.dpp
    // CHECK-SAME: with 323, 15, 15, true : f32
    // CHECK-NEXT: llvm.intr.maxnum

    // CHECK: llvm.amdgcn.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<64xf32, #blocked3>) -> f32
    tt.return
  }
}

// -----

#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: reduce_xor_max
  tt.func @reduce_xor_max(%arg0: tensor<32xf32, #blocked4>) {
    // CHECK: rocdl.ds_swizzle
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 280, 15, 12, false : i32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 264, 15, 3, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 276, 15, 10, false : i32
    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 260, 15, 5, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 78, 15, 15, false : i32
    // CHECK: llvm.intr.maxnum

    // CHECK: rocdl.update.dpp
    // CHECK-SAME: with 177, 15, 15, false : i32
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked4>) -> f32
    tt.return
  }
}
