// RUN: triton-opt %s -split-input-file -tritonamdgpu-gluon-in-thread-transpose | FileCheck %s

#blockedA = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#blockedB = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#sharedA = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#sharedB = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = true}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[$BLOCKED_A:#.*]] = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK-DAG: [[$BLOCKED_B:#.*]] = #ttg.blocked<{sizePerThread = [16, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK-DAG: [[$TRANSPOSED_A:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0{{]]}}, lane = {{\[\[}}4, 0], [8, 0], [16, 0], [0, 16], [0, 32], [0, 64{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$TRANSPOSED_B:#.*]] = #ttg.linear<{register = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2{{]]}}, lane = {{\[\[}}0, 4], [0, 8], [0, 16], [16, 0], [32, 0], [64, 0{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$SHARED_A:#.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK-DAG: [[$SHARED_B:#.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
// CHECK-LABEL: legacyInThreadTranspose

// CHECK-DAG: [[DATA_A:%.*]] = tt.load {{.*}} : tensor<32x128x!tt.ptr<f16>, [[$BLOCKED_A]]>
// CHECK-DAG: [[DATA_B:%.*]] = tt.load {{.*}} : tensor<128x32x!tt.ptr<f16>, [[$BLOCKED_B]]>
// CHECK-DAG: [[TRANSPOSED_A:%.*]] = amdg.in_thread_transpose [[DATA_A]] : tensor<32x128xf16, [[$BLOCKED_A]]> -> tensor<32x128xf16, [[$TRANSPOSED_A]]>
// CHECK-DAG: [[TRANSPOSED_B:%.*]] = amdg.in_thread_transpose [[DATA_B]] : tensor<128x32xf16, [[$BLOCKED_B]]> -> tensor<128x32xf16, [[$TRANSPOSED_B]]>
// CHECK-DAG: [[BUF_A:%.*]] = ttg.local_alloc [[TRANSPOSED_A]] : (tensor<32x128xf16, [[$TRANSPOSED_A]]>) -> !ttg.memdesc<32x128xf16, [[$SHARED_A]], #smem>
// CHECK-DAG: [[BUF_B:%.*]] = ttg.local_alloc [[TRANSPOSED_B]] : (tensor<128x32xf16, [[$TRANSPOSED_B]]>) -> !ttg.memdesc<128x32xf16, [[$SHARED_B]], #smem>
// CHECK-DAG: ttg.local_load [[BUF_A]]
// CHECK-DAG: ttg.local_load [[BUF_B]]
  tt.func public @legacyInThreadTranspose(%arg0: tensor<32x128x!tt.ptr<f16>, #blockedA>, %arg1: tensor<128x32x!tt.ptr<f16>, #blockedB>, %c: tensor<32x32xf32, #mma>) {
    %dataA = tt.load %arg0 : tensor<32x128x!tt.ptr<f16>, #blockedA>
    %dataB = tt.load %arg1 : tensor<128x32x!tt.ptr<f16>, #blockedB>

    %bufA = ttg.local_alloc %dataA : (tensor<32x128xf16, #blockedA>) -> !ttg.memdesc<32x128xf16, #sharedA, #smem>
    %a = ttg.local_load %bufA : !ttg.memdesc<32x128xf16, #sharedA, #smem> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>

    %bufB = ttg.local_alloc %dataB : (tensor<128x32xf16, #blockedB>) -> !ttg.memdesc<128x32xf16, #sharedB, #smem>
    %b = ttg.local_load %bufB : !ttg.memdesc<128x32xf16, #sharedB, #smem> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>

    %d = tt.dot %a, %b, %c : tensor<32x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK-DAG: [[$TRANSPOSED:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0{{]]}}, lane = {{\[\[}}4, 0], [8, 0], [16, 0], [0, 16], [0, 32], [0, 64{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$SHARED:#.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: minimalAllocInThreadTranspose
// CHECK-SAME: ([[DATA:%.*]]: tensor<32x128xf16, [[$BLOCKED]]>)
// CHECK: [[TRANSPOSED:%.*]] = amdg.in_thread_transpose [[DATA]] : tensor<32x128xf16, [[$BLOCKED]]> -> tensor<32x128xf16, [[$TRANSPOSED]]>
// CHECK: [[BUF:%.*]] = ttg.local_alloc [[TRANSPOSED]] : (tensor<32x128xf16, [[$TRANSPOSED]]>) -> !ttg.memdesc<32x128xf16, [[$SHARED]], #smem>
// CHECK: tt.return [[BUF]]
  tt.func public @minimalAllocInThreadTranspose(%data: tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem> {
    %buf = ttg.local_alloc %data : (tensor<32x128xf16, #blocked>) -> !ttg.memdesc<32x128xf16, #shared, #smem>
    tt.return %buf : !ttg.memdesc<32x128xf16, #shared, #smem>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-DAG: [[$BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
// CHECK-DAG: [[$TRANSPOSED:#.*]] = #ttg.linear<{register = {{\[\[}}0, 1], [0, 2], [0, 4], [0, 8], [1, 0], [2, 0{{]]}}, lane = {{\[\[}}4, 0], [8, 0], [16, 0], [0, 16], [0, 32], [0, 64{{]]}}, warp = [], block = []}>
// CHECK-DAG: [[$SHARED:#.*]] = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: minimalLocalStoreInThreadTranspose
// CHECK-SAME: ([[DATA:%.*]]: tensor<32x128xf16, [[$BLOCKED]]>, [[BUF:%.*]]: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>)
// CHECK-DAG: [[TRANSPOSED:%.*]] = amdg.in_thread_transpose [[DATA]] : tensor<32x128xf16, [[$BLOCKED]]> -> tensor<32x128xf16, [[$TRANSPOSED]]>
// CHECK-DAG: ttg.local_store [[TRANSPOSED]], [[BUF]] : tensor<32x128xf16, [[$TRANSPOSED]]> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
// CHECK: tt.return [[BUF]]
  tt.func public @minimalLocalStoreInThreadTranspose(%data: tensor<32x128xf16, #blocked>, %buf: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>) -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> {
    ttg.local_store %data, %buf : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %buf : !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+4] {order = [0, 1], shape = [32, 128]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: paddedSharedEncoding
// CHECK: amdg.in_thread_transpose
  tt.func public @paddedSharedEncoding(%data: tensor<32x128xf16, #blocked>, %buf: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>) -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> {
    ttg.local_store %data, %buf : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %buf : !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: skinnySizePerThreadNegative
// CHECK-NOT: amdg.in_thread_transpose
  tt.func public @skinnySizePerThreadNegative(%data: tensor<32x128xf16, #blocked>, %buf: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>) -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> {
    ttg.local_store %data, %buf : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %buf : !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 16], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {

// CHECK-LABEL: sameOrderNegative
// CHECK-NOT: amdg.in_thread_transpose
  tt.func public @sameOrderNegative(%data: tensor<32x128xf16, #blocked>, %buf: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>) -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable> {
    ttg.local_store %data, %buf : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    tt.return %buf : !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
  }
}
