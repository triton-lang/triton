// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-descriptor-encoding | FileCheck %s
// Test that gather/scatter are assigned padded encodings

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>
// CHECK-LABEL: @descriptor_gather
tt.func public @descriptor_gather(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked> ) -> tensor<32x32xi8, #blocked1> {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[$PADDED]]>
  // CHECK: tt.descriptor_gather {{.*}} : (!tt.tensordesc<1x32xi8, #[[$PADDED]]>
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <1x32xi8>
  %2 = tt.descriptor_gather %1[%arg3, %c8_i32] : (!tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>
  tt.return %2 : tensor<32x32xi8, #blocked1>
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>
// CHECK-LABEL: @descriptor_scatter
tt.func public @descriptor_scatter(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %arg4: tensor<32x32xi8, #blocked1>) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[$PADDED]]>
  // CHECK: tt.descriptor_scatter {{.*}} : !tt.tensordesc<1x32xi8, #[[$PADDED]]>, {{.*}}
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c64_i32 = arith.constant 64 : i32
  %c8_i32 = arith.constant 8 : i32
  %0 = arith.extsi %arg2 : i32 to i64
  %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <1x32xi8>
  tt.descriptor_scatter %1[%arg3, %c8_i32], %arg4 : !tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32, tensor<32x32xi8, #blocked1>
  tt.return
}
}

// -----
// Test that descriptor gets the encoding last use of descriptor load
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+2] { order = [1, 0], shape = [256, 32] }>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[32:+2] {order = [2, 1, 0], shape = [1, 256, 32]}>
// CHECK-DAG: #[[$PADDED_ALLOC:.*]] = #ttg.padded_shared<[32:+2] {order = [1, 0], shape = [256, 32]}>
// CHECK-LABEL: @descriptor_load
tt.func public @descriptor_load(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f32>, <1x256x32xf32, #[[$PADDED]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<1x256x32xf32, #[[$PADDED]]> -> tensor<256x32xf32, #[[$BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<256x32xf32, #[[$BLOCKED]]>) -> !ttg.memdesc<256x32xf32, #[[$PADDED_ALLOC]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %arg4, %c1_i64] : <f32>, <1x256x32xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c1_i32, %c1_i32] : !tt.tensordesc<1x256x32xf32> -> tensor<256x32xf32, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<256x32xf32, #blocked>) -> !ttg.memdesc<256x32xf32, #shared, #smem>
  tt.return
}
}

// -----
// Test that host tensor descriptor in kernel argument gets the encoding
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[64:+8] { order = [1, 0], shape = [64, 64] }>
#smem = #ttg.shared_memory
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$PADDED:.*]] = #ttg.padded_shared<[64:+8] {order = [1, 0], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-LABEL: @descriptor_kernel_arg
tt.func public @descriptor_kernel_arg(%arg0: !tt.tensordesc<64x64xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: %arg0: !tt.tensordesc<64x64xf16, #[[$PADDED]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %arg0[{{.*}}] : !tt.tensordesc<64x64xf16, #[[$PADDED]]> -> tensor<64x64xf16, #[[$BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<64x64xf16, #[[$BLOCKED]]>) -> !ttg.memdesc<64x64xf16, #[[$PADDED]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %1 = tt.descriptor_load %arg0[%c1_i32, %c1_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
  tt.return
}
}

// -----
// Test propagation of descriptor encoding through while loop
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+16] { order = [1, 0], shape = [32, 32] }>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$PADDED_DESC:.*]] = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [1, 32]}>
// CHECK-DAG: #[[$PADDED_ALLOC:.*]] = #ttg.padded_shared<[32:+16] {order = [1, 0], shape = [32, 32]}>
// CHECK-LABEL: @descriptor_load_while
tt.func public @descriptor_load_while(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %cond: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64

    %0 = arith.extsi %arg2 : i32 to i64
    // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[$PADDED_DESC]]>
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <1x32xi8>

    %2 = scf.while (%arg4 = %1) : (!tt.tensordesc<1x32xi8>) -> (!tt.tensordesc<1x32xi8>) {
        scf.condition(%cond) %arg4 : !tt.tensordesc<1x32xi8>
    } do {
        ^bb0(%arg4: !tt.tensordesc<1x32xi8>):
          // CHECK: ^bb0(%[[ARG4:.*]]: !tt.tensordesc<1x32xi8, #[[$PADDED_DESC]]>):
          // CHECK: tt.descriptor_gather %[[ARG4]][{{.*}}] : (!tt.tensordesc<1x32xi8, #[[$PADDED_DESC]]>
          %3 = tt.descriptor_gather %arg4[%arg3, %c8_i32] : (!tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>

        scf.yield %arg4 : !tt.tensordesc<1x32xi8>
    }

  // CHECK: %[[GATHER:.*]] = tt.descriptor_gather {{.*}} : (!tt.tensordesc<1x32xi8, #[[$PADDED_DESC]]>
    %4 = tt.descriptor_gather %1[%arg3, %c8_i32] : (!tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>
    // CHECK: ttg.local_alloc %[[GATHER]] {{.*}} : (tensor<32x32xi8, #blocked1>) -> !ttg.memdesc<32x32xi8, #[[$PADDED_ALLOC]], #smem>
    %8 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<32x32xi8, #blocked1>) -> !ttg.memdesc<32x32xi8, #shared, #smem>

  tt.return
}
}

// -----
// Test propagation of descriptor encoding through dot operand
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[1, 0], [2, 0], [4, 0]]}, instrShape = [16, 16, 32]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[$PADDED_A:.*]] = #ttg.padded_shared<[128:+8] {
// CHECK-DAG: #[[$PADDED_B:.*]] = #ttg.padded_shared<[128:+16] {
// CHECK-LABEL: @descriptor_load_dot_operand
tt.func public @descriptor_load_dot_operand(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i64, %arg5: i64) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f16>, <512x32xf16, #[[$PADDED_A]]>
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f16>, <32x64xf16, #[[$PADDED_B]]
  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %cst = arith.constant dense<0.000000e+00> : tensor<512x64xf32, #mma>
  %0 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%arg4, %c1_i64] : <f16>, <512x32xf16>
  %1 = tt.make_tensor_descriptor %arg1, [%arg3, %arg2], [%arg5, %c1_i64] : <f16>, <32x64xf16>
  %2 = tt.descriptor_load %0[%c0_i32, %c0_i32] : !tt.tensordesc<512x32xf16> -> tensor<512x32xf16, #blocked>
  %3 = tt.descriptor_load %1[%c0_i32, %c0_i32] : !tt.tensordesc<32x64xf16> -> tensor<32x64xf16, #blocked1>
  %4 = ttg.convert_layout %2 : tensor<512x32xf16, #blocked> -> tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
  %5 = ttg.convert_layout %3 : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
  %6 = tt.dot %4, %5, %cst : tensor<512x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<512x64xf32, #mma>
  tt.return
}
}

// -----
// Test propagation of descriptor encoding through for and if (load in both then and else)
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+2] { order = [1, 0], shape = [64, 32] }>
#shared1 = #ttg.padded_shared<[32:+8] { order = [1, 0], shape = [64, 32] }>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250"} {
// CHECK-DAG: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[$PADDED1:.*]] = #ttg.padded_shared<[32:+2] {order = [1, 0], shape = [64, 32]}>
// CHECK-DAG: #[[$PADDED2:.*]] = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [64, 32]}>
// CHECK-DAG: #[[$PADDED_FALLBACK:.*]] = #ttg.padded_shared<[32:+4] {order = [2, 1, 0], shape = [1, 64, 32]}>
// CHECK-LABEL: @descriptor_fallback
tt.func public @descriptor_fallback(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %cond: i1) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %rng = arith.constant 5 : index
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f32>, <1x64x32xf32, #[[$PADDED_FALLBACK]]>
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%c1_i64, %arg3, %arg4] : <f32>, <1x64x32xf32>
  // CHECK: scf.for {{.*}} -> (!tt.tensordesc<1x64x32xf32, #[[$PADDED_FALLBACK]]>)
  %1 = scf.for %iv = %c0 to %rng step %c1 iter_args(%iter_desc = %0) -> (!tt.tensordesc<1x64x32xf32>) {
    // CHECK: scf.if {{.*}} -> (!tt.tensordesc<1x64x32xf32, #[[$PADDED_FALLBACK]]>)
    %2 = scf.if %cond -> (!tt.tensordesc<1x64x32xf32>) {
      // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<1x64x32xf32, #[[$PADDED_FALLBACK]]> -> tensor<64x32xf32, #[[$BLOCKED]]>
      %3 = tt.descriptor_load %iter_desc[%c1_i32, %c1_i32, %c1_i32] : !tt.tensordesc<1x64x32xf32> -> tensor<64x32xf32, #blocked>
      %4 = ttg.local_alloc %3 : (tensor<64x32xf32, #blocked>) -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
      scf.yield %iter_desc : !tt.tensordesc<1x64x32xf32>
    } else {
      // CHECK: tt.descriptor_load {{.*}} : !tt.tensordesc<1x64x32xf32, #[[$PADDED_FALLBACK]]> -> tensor<64x32xf32, #[[$BLOCKED]]>
      %5 = tt.descriptor_load %iter_desc[%c0_i32, %c0_i32, %c1_i32] : !tt.tensordesc<1x64x32xf32> -> tensor<64x32xf32, #blocked>
      %6 = ttg.local_alloc %5 : (tensor<64x32xf32, #blocked>) -> !ttg.memdesc<64x32xf32, #shared1, #smem, mutable>
      scf.yield %iter_desc : !tt.tensordesc<1x64x32xf32>
    }
    scf.yield %2 : !tt.tensordesc<1x64x32xf32>
  }
  tt.return
}
}
