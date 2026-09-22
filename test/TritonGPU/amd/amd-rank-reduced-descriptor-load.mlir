// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-descriptor-encoding --tritonamdgpu-convert-tensor-ops | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$DESC_LAYOUT:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
  // CHECK-DAG: #[[$ALLOC_LAYOUT:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
  // CHECK-LABEL: @rank_reduced_descriptor_load_pad_limit
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i16>, <1x1024xi16, #[[$DESC_LAYOUT]]>
  // CHECK: amdg.async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<1x1024xi16, #[[$DESC_LAYOUT]]> -> !ttg.memdesc<1024xi16, #[[$ALLOC_LAYOUT]],
  tt.func public @rank_reduced_descriptor_load_pad_limit(
      %base: !tt.ptr<i16>,
      %stride: i64) -> tensor<1024xi16, #blocked> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c1024_i32 = arith.constant 1024 : i32
    %desc = tt.make_tensor_descriptor %base, [%c1_i32, %c1024_i32], [%stride, %c1_i64] : <i16>, <1x1024xi16>
    %result = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<1x1024xi16> -> tensor<1024xi16, #blocked>
    tt.return %result : tensor<1024xi16, #blocked>
  }
}

// -----

#blocked_multi_cta = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$MC_DESC_LAYOUT:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
  // CHECK-DAG: #[[$MC_ALLOC_LAYOUT:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = {{\[\[1\]\]}}}>
  // CHECK-LABEL: @multi_cta_rank_reduced_descriptor_load
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i16>, <1x2048xi16, #[[$MC_DESC_LAYOUT]]>
  // CHECK: amdg.async_tdm_copy_global_to_local {{.*}} : !tt.tensordesc<1x2048xi16, #[[$MC_DESC_LAYOUT]]> -> !ttg.memdesc<2048xi16, #[[$MC_ALLOC_LAYOUT]],
  tt.func public @multi_cta_rank_reduced_descriptor_load(
      %base: !tt.ptr<i16>,
      %stride: i64) -> tensor<2048xi16, #blocked_multi_cta> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2048_i32 = arith.constant 2048 : i32
    %desc = tt.make_tensor_descriptor %base, [%c1_i32, %c2048_i32], [%stride, %c1_i64] : <i16>, <1x2048xi16>
    %result = tt.descriptor_load %desc[%c0_i32, %c0_i32] : !tt.tensordesc<1x2048xi16> -> tensor<2048xi16, #blocked_multi_cta>
    tt.return %result : tensor<2048xi16, #blocked_multi_cta>
  }
}
