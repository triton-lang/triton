// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1170 --convert-builtin-func-to-llvm | FileCheck %s --check-prefixes=CHECK,GFX11
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1200 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GFX12

#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: reduce_dpp_max
  tt.func @reduce_dpp_max(%arg0: tensor<32xf32, #blocked3>) {
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

    // CHECK: rocdl.permlanex16
    // CHECK: llvm.intr.maxnum
    // CHECK: rocdl.readlane
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %1 = arith.maxnumf %arg1, %arg2 : f32
      tt.reduce.return %1 : f32
    }) : (tensor<32xf32, #blocked3>) -> f32
    tt.return
  }
}

#linear = #ttg.linear<{register = [[16, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>> {
  // This tensor has 64 elements with the last dimension across the lower and upper 16 lanes.
  // Therefore, we can reduce it with a 16 element butterfly shuffle.

  // CHECK-DAG: [[result0:%.*]] = llvm.mlir.undef
  // CHECK-DAG: [[select_lo:%.*]] = llvm.mlir.constant(1985229328 : i32)
  // CHECK-DAG: [[select_hi:%.*]] = llvm.mlir.constant(-19088744 : i32)
  // CHECK-DAG: [[reg0:%.*]] = llvm.extractvalue %arg0[0]
  // CHECK-DAG: [[reg1:%.*]] = llvm.extractvalue %arg0[1]
  // CHECK: [[permlane0:%.*]] = rocdl.permlanex16 [[reg0]], [[reg0]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum0:%.*]] = llvm.add [[reg0]], [[permlane0]]
  // CHECK: [[permlane1:%.*]] = rocdl.permlanex16 [[reg1]], [[reg1]], [[select_lo]], [[select_hi]], true, false
  // CHECK: [[sum1:%.*]] = llvm.add [[reg1]], [[permlane1]]
  // CHECK: [[result1:%.*]] = llvm.insertvalue [[sum0]], [[result0]][0]
  // CHECK: [[result2:%.*]] = llvm.insertvalue [[sum1]], [[result1]][1]

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 1 : i32} : (tensor<32x2xi32, #linear>) -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>

  // CHECK: llvm.return [[result2]]
  tt.return %0 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #linear}>>
}
}

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: @bf16_mulf
tt.func private @bf16_mulf(%arg0: tensor<64xbf16, #blocked>, %arg1: tensor<64xbf16, #blocked>) -> tensor<64xbf16, #blocked> {
  // v_dot2_bf16_bf16 truncates the bf16 result, so the multiply has to go
  // through fp32 and round to nearest even explicitly.
  // CHECK-NOT: llvm.amdgcn.fdot2.bf16.bf16
  // CHECK-COUNT-2: llvm.fmul {{.*}} : f32
  // CHECK-NOT: llvm.amdgcn.fdot2.bf16.bf16
  %0 = arith.mulf %arg0, %arg1 : tensor<64xbf16, #blocked>
  tt.return %0 : tensor<64xbf16, #blocked>
}
}

// -----

// GFX11-DAG: [[$REL_GLOBAL:#[A-Za-z0-9_]+]] = #llvm.mmra_tag<"amdgpu-synchronize-as":"global">
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX11-LABEL: @atomic_release_scalar_multi_wave
  // GFX12-LABEL: @atomic_release_scalar_multi_wave
  tt.func @atomic_release_scalar_multi_wave(%ptr: !tt.ptr<i32>, %val: i32) {
    // GFX11: llvm.fence syncscope("agent") release {llvm.mmra = [[$REL_GLOBAL]]}
    // GFX11-NEXT: llvm.fence syncscope("workgroup") release
    // GFX11-NEXT: rocdl.s.barrier
    // GFX11: llvm.atomicrmw add {{.*}} syncscope("agent") release
    // GFX12: rocdl.s.wait.storecnt 0
    // GFX12-NEXT: llvm.fence syncscope("workgroup") release
    // GFX12-NEXT: rocdl.s.barrier.signal
    // GFX12: llvm.atomicrmw add {{.*}} syncscope("agent") release
    %0 = tt.atomic_rmw add, release, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    // The system-scope release does not repeat global_wb in every wave on
    // gfx12; the atomic's own release writes back once.
    // GFX11: llvm.fence syncscope("agent") release {llvm.mmra = [[$REL_GLOBAL]]}
    // GFX11-NEXT: llvm.fence syncscope("workgroup") release
    // GFX11: llvm.atomicrmw add {{.*}} release
    // GFX12: rocdl.s.wait.storecnt 0
    // GFX12-NEXT: llvm.fence syncscope("workgroup") release
    // GFX12: llvm.atomicrmw add {{.*}} release
    %1 = tt.atomic_rmw add, release, sys, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    // GFX11-NOT: llvm.fence syncscope("agent")
    // GFX12-NOT: rocdl.s.wait.storecnt
    // GFX11: llvm.atomicrmw add {{.*}} syncscope("workgroup") release
    // GFX12: llvm.atomicrmw add {{.*}} syncscope("workgroup") release
    %2 = tt.atomic_rmw add, release, cta, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}
