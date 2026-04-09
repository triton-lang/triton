// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx90a -cse | FileCheck %s --check-prefix=GFX90A
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 -cse | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 -cse | FileCheck %s --check-prefix=GFX950
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 -cse | FileCheck %s --check-prefix=GFX1250

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // GFX942-LABEL: reduce_f16
  // GFX942: llvm.fadd {{.*}} : vector<2xf16>
  // GFX950-LABEL: reduce_f16
  // GFX950: llvm.fadd {{.*}} : vector<2xf16>
  tt.func public @reduce_f16(%arg0: tensor<1x256xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x256xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX90A-LABEL: reduce_f32
  // GFX90A: llvm.fadd {{.*}} : vector<2xf32>
  // GFX942-LABEL: reduce_f32
  // GFX942: llvm.fadd {{.*}} : vector<2xf32>
  // GFX950-LABEL: reduce_f32
  // GFX950: llvm.fadd {{.*}} : vector<2xf32>
  tt.func public @reduce_f32(%arg0: tensor<1x256xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x256xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}

// -----

#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: reduce_f16_tree_vectorize
  // GFX1250: llvm.fadd {{.*}} : vector<2xf16>
  tt.func public @reduce_f16_tree_vectorize(%arg0: tensor<1x128xf16, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) : (tensor<1x128xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_f32_tree_vectorize
  // GFX1250: llvm.fadd {{.*}} : vector<2xf32>
  tt.func public @reduce_f32_tree_vectorize(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // Ternary tree reduction for max/min: generates a chain of 3 dependent ops
  // per group so LLVM can fold into v_maximum3/v_minimum3/v_max3/v_min3.

  // GFX1250-LABEL: reduce_maximum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.maximum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.maximum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_maximum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maximumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_minimum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.minimum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.minimum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.minimum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_minimum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minimumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_maxnum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.maxnum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.maxnum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_maxnum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %max = arith.maxnumf %a, %b : f32
      tt.reduce.return %max : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // GFX1250-LABEL: reduce_minnum_f32_ternary
  // GFX1250: %[[A:.*]] = llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[B:.*]] = llvm.intr.minnum(%[[A]], %{{.*}}) : (f32, f32) -> f32
  // GFX1250-NEXT: %[[C:.*]] = llvm.intr.minnum(%[[B]], %{{.*}}) : (f32, f32) -> f32
  tt.func public @reduce_minnum_f32_ternary(%arg0: tensor<1x128xf32, #blocked_reduce>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %min = arith.minnumf %a, %b : f32
      tt.reduce.return %min : f32
    }) : (tensor<1x128xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }
}
