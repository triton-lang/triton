// RUN: triton-opt %s --convert-triton-gpu-to-llvm=compute-capability=70 2>&1 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-LABEL: clamp
module attributes {"ttg.target" = "cuda:70", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @clamp(%x : tensor<1024xf32, #blocked>, %limit : tensor<1024xf32, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32, #blocked>
    %neg_limit = arith.subf %cst, %limit : tensor<1024xf32, #blocked>

    // CHECK:      llvm.fcmp "une" %[[REG:[a-zA-Z0-9]+]], %[[REG]]
    // CHECK-NEXT: llvm.intr.maxnum
    // CHECK-NEXT: llvm.intr.minnum
    // CHECK-NEXT: llvm.mlir.constant
    // CHECK-NEXT: llvm.select
    %12 = tt.clampf %x, %neg_limit, %limit, propagateNan = all : tensor<1024xf32, #blocked>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: store_with_cache_attr
  tt.func @store_with_cache_attr(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    // CHECK-NOT: createpolicy.fractional
    // CHECK: st.global.L1::evict_last.b32
    tt.store %a_ptr_init, %cst_0, %cst evictionPolicy = evict_last cacheModifier = ca : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:70", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fastmath_minmax_emulated
  // CHECK-COUNT-2: llvm.fcmp "une" {{.*}} {fastmathFlags = #llvm.fastmath<nnan>} : f32
  // CHECK: llvm.intr.maxnum({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>} : (f32, f32) -> f32
  // CHECK: llvm.select {{.*}} {fastmathFlags = #llvm.fastmath<nnan>} : i1, f32
  tt.func @fastmath_minmax_emulated(
      %arg0: tensor<32xf32, #blocked1>,
      %arg1: tensor<32xf32, #blocked1>) {
    %maximum = arith.maximumf %arg0, %arg1 fastmath<nnan> : tensor<32xf32, #blocked1>
    tt.return
  }
}
