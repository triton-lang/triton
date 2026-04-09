// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding | FileCheck %s
// Test that gather/scatter are assigned swizzled encodings

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_gather(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked> ) -> tensor<32x32xi8, #blocked1> {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[NVMMA_32]]>
  // CHECK: tt.descriptor_gather {{.*}} : (!tt.tensordesc<1x32xi8, #[[NVMMA_32]]>
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_scatter(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %arg4: tensor<32x32xi8, #blocked1>) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[NVMMA_32]]>
  // CHECK: tt.descriptor_scatter {{.*}} : !tt.tensordesc<1x32xi8, #[[NVMMA_32]]>, {{.*}}
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

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[SWIZZLE_MMA:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
// CHECK-DAG: #[[SWIZZLE_2D:.*]] = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
tt.func public @tma_scatter(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: tt.make_tensor_descriptor {{.*}} : <f32>, <1x256x32xf32, #[[SWIZZLE_MMA]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load {{.*}} : !tt.tensordesc<1x256x32xf32, #[[SWIZZLE_MMA]]> -> tensor<256x32xf32, #[[BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<256x32xf32, #[[BLOCKED]]>) -> !ttg.memdesc<256x32xf32, #[[SWIZZLE_2D]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %arg1, %arg2], [%arg3, %arg4, %c1_i64] : <f32>, <1x256x32xf32>
  %1 = tt.descriptor_load %0[%c1_i32, %c1_i32, %c1_i32] : !tt.tensordesc<1x256x32xf32> -> tensor<256x32xf32, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<256x32xf32, #blocked>) -> !ttg.memdesc<256x32xf32, #shared, #smem>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[NVMMA_64:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
tt.func public @descriptor_kernel_arg(%arg0: !tt.tensordesc<64x64xf16>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64) {
  // CHECK: %arg0: !tt.tensordesc<64x64xf16, #[[NVMMA_64]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %arg0[{{.*}}] : !tt.tensordesc<64x64xf16, #[[NVMMA_64]]> -> tensor<64x64xf16, #[[BLOCKED]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<64x64xf16, #[[BLOCKED]]>) -> !ttg.memdesc<64x64xf16, #[[NVMMA_64]], #smem>
  %c1_i32 = arith.constant 1 : i32
  %1 = tt.descriptor_load %arg0[%c1_i32, %c1_i32] : !tt.tensordesc<64x64xf16> -> tensor<64x64xf16, #blocked>
  %2 = ttg.local_alloc %1 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED_16x128:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[NVMMA_128:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
// CHECK-DAG: #[[NVMMA_TRANSPOSED_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
tt.func public @descriptor_ignores_transposed_local_alloc(%arg0: !tt.tensordesc<16x128xf16>) {
  // CHECK: %arg0: !tt.tensordesc<16x128xf16, #[[NVMMA_128]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %arg0{{.*}} : !tt.tensordesc<16x128xf16, #[[NVMMA_128]]> -> tensor<16x128xf16, #[[BLOCKED_16x128]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<16x128xf16, #[[BLOCKED_16x128]]>) -> !ttg.memdesc<16x128xf16, #[[NVMMA_TRANSPOSED_32]], #smem>
  %c0 = arith.constant 0 : i32
  %0 = tt.descriptor_load %arg0[%c0, %c0] : !tt.tensordesc<16x128xf16> -> tensor<16x128xf16, #blocked>
  %1 = ttg.local_alloc %0 : (tensor<16x128xf16, #blocked>) -> !ttg.memdesc<16x128xf16, #shared, #smem>
  tt.return
}
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
// CHECK-DAG: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[NVMMA_32:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
tt.func public @tma_load_while(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: tensor<32xi32, #blocked>, %cond: i1) {
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64

    %0 = arith.extsi %arg2 : i32 to i64
    // CHECK: tt.make_tensor_descriptor {{.*}} : <i8>, <1x32xi8, #[[NVMMA_32]]>
    %1 = tt.make_tensor_descriptor %arg0, [%arg1, %arg2], [%0, %c1_i64] : <i8>, <1x32xi8>

    %2 = scf.while (%arg4 = %1) : (!tt.tensordesc<1x32xi8>) -> (!tt.tensordesc<1x32xi8>) {
        scf.condition(%cond) %arg4 : !tt.tensordesc<1x32xi8>
    } do {
        ^bb0(%arg4: !tt.tensordesc<1x32xi8>):
          // CHECK: ^bb0(%[[ARG4:.*]]: !tt.tensordesc<1x32xi8, #[[NVMMA_32]]>):
          // CHECK: tt.descriptor_gather %[[ARG4]][{{.*}}] : (!tt.tensordesc<1x32xi8, #[[NVMMA_32]]>
          %3 = tt.descriptor_gather %arg4[%arg3, %c8_i32] : (!tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>

        scf.yield %arg4 : !tt.tensordesc<1x32xi8>
    }

  // CHECK: %[[GATHER:.*]] = tt.descriptor_gather {{.*}} : (!tt.tensordesc<1x32xi8, #[[NVMMA_32]]>
    %4 = tt.descriptor_gather %1[%arg3, %c8_i32] : (!tt.tensordesc<1x32xi8>, tensor<32xi32, #blocked>, i32) -> tensor<32x32xi8, #blocked1>
    // CHECK: ttg.local_alloc %[[GATHER]] {{.*}} : (tensor<32x32xi8, #blocked1>) -> !ttg.memdesc<32x32xi8, #[[NVMMA_32]], #smem>
    %8 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<32x32xi8, #blocked1>) -> !ttg.memdesc<32x32xi8, #shared, #smem>

  tt.return
}
}

// -----

#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 4, 8, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared_linear_base = #ttg.shared_linear<{offset = [[0, 0, 0, 0, 1], [0, 0, 0, 0, 2], [0, 0, 0, 0, 4], [0, 0, 0, 0, 8], [0, 0, 0, 1, 0], [0, 0, 0, 2, 0], [0, 0, 0, 4, 0], [0, 0, 1, 0, 0], [0, 0, 2, 0, 0], [0, 0, 4, 0, 0], [0, 0, 8, 0, 0], [0, 0, 16, 0, 0], [0, 0, 32, 0, 0], [0, 0, 64, 0, 0], [0, 0, 128, 0, 0]]}, alignment = 128>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
// CHECK-DAG: #[[NVMMA_0:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8, rank = 5}>
// CHECK-DAG: #[[BLOCKED5D:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 16], threadsPerWarp = [1, 1, 4, 8, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
// CHECK-DAG: #[[SL_BASE:.*]] = #ttg.shared_linear<{{.*}}alignment = 128>
tt.func public @descriptor_arg_from_shared_linear_use(%b_desc: !tt.tensordesc<1x1x256x8x16xf8E4M3FN>) {
  // CHECK: %arg0: !tt.tensordesc<1x1x256x8x16xf8E4M3FN, #[[NVMMA_0]]>
  // CHECK: %[[LOAD:.*]] = tt.descriptor_load %arg0
  // CHECK-SAME: : !tt.tensordesc<1x1x256x8x16xf8E4M3FN, #[[NVMMA_0]]> -> tensor<1x1x256x8x16xf8E4M3FN, #[[BLOCKED5D]]>
  // CHECK: ttg.local_alloc %[[LOAD]] : (tensor<1x1x256x8x16xf8E4M3FN, #[[BLOCKED5D]]>) -> !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #[[SL_BASE]], #smem>
  %c0 = arith.constant 0 : i32
  %b = tt.descriptor_load %b_desc[%c0, %c0, %c0, %c0, %c0] : !tt.tensordesc<1x1x256x8x16xf8E4M3FN> -> tensor<1x1x256x8x16xf8E4M3FN, #blocked2>
  %b_s = ttg.local_alloc %b : (tensor<1x1x256x8x16xf8E4M3FN, #blocked2>) -> !ttg.memdesc<1x1x256x8x16xf8E4M3FN, #shared_linear_base, #smem>
  tt.return
}
}
