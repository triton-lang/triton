// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK: global internal constant @tensor_constant_1([34359738368, 68719476736]) {addr_space = 0 : i32} : !llvm.array<2 x i64>
// CHECK: global internal constant @tensor_constant_0([0, 42]) {addr_space = 0 : i32} : !llvm.array<2 x i64>

// CHECK-LABEL: @experimental_buffer_descriptors_tmem
// CHECK: llvm.mlir.constant(4294967295 : i64) : i64
tt.func private @experimental_buffer_descriptors_tmem() {
  tti.experimental_buffer_descriptors [0, 42], [8, 16], tensor_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK: global internal constant @tensor_constant_1([17179869184, 51539607552])
// CHECK: global internal constant @tensor_constant_0([0, 42])

// CHECK-LABEL: @experimental_buffer_descriptors_shared
// CHECK: llvm.mlir.constant(16777215 : i64) : i64
tt.func private @experimental_buffer_descriptors_shared() {
  tti.experimental_buffer_descriptors [0, 42], [4, 12], shared_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_lock_acquire
// CHECK: 09atom.global.acquire.gpu.cas.b32
// CHECK: nvvm.barrier
tt.func private @experimental_lock_acquire(
  %lock: !tt.ptr<i32>,
  %pred: i1
) {
  tti.experimental_lock_acquire %lock, %pred : !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_lock_release
// CHECK: nvvm.barrier
// CHECK: atom.global.release.gpu.exch.b32
tt.func private @experimental_lock_release(
  %lock: !tt.ptr<i32>,
  %pred: i1
) {
  tti.experimental_lock_release %lock, %pred : !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_memdesc_to_i32
// CHECK:  llvm.ptrtoint %1 : !llvm.ptr<3> to i32
tt.func private @experimental_memdesc_to_i32(
  %memdesc: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
) {
  tti.experimental_memdesc_to_i32 %memdesc : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
  tt.return
}
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_gsan_tensordesc_info
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.inttoptr
// CHECK-NOT: llvm.lshr
// CHECK: %[[DESC:.*]] = llvm.load %arg0 : !llvm.ptr -> !llvm.struct<
// CHECK: %[[BASE:.*]] = llvm.extractvalue %[[DESC]][0] : !llvm.struct<
// CHECK: %[[SHAPE0:.*]] = llvm.extractvalue %[[DESC]][8] : !llvm.struct<
// CHECK: llvm.zext %[[SHAPE0]] : i32 to i64
// CHECK: llvm.add %{{.*}}, %{{.*}} : i64
// CHECK: %[[SHAPE1:.*]] = llvm.extractvalue %[[DESC]][7] : !llvm.struct<
// CHECK: llvm.zext %[[SHAPE1]] : i32 to i64
// CHECK: llvm.add %{{.*}}, %{{.*}} : i64
// CHECK: %[[STRIDE:.*]] = llvm.extractvalue %[[DESC]][2] : !llvm.struct<
// CHECK: llvm.zext %[[STRIDE]] : i32 to i64
// CHECK: llvm.mul %{{.*}}, %{{.*}} : i64
// CHECK: llvm.udiv %{{.*}}, %{{.*}} : i64
tt.func private @experimental_gsan_tensordesc_info(
  %desc: !tt.tensordesc<32x32xf32, #shared>
) {
  %0:5 = "tti.experimental_gsan_tensordesc_info"(%desc) : (!tt.tensordesc<32x32xf32, #shared>) -> (!tt.ptr<f32, 1>, i64, i64, i64, i64)
  tt.return
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_fpsan_embed
// CHECK-NOT: tti.experimental_fpsan_embed
// CHECK: %[[RAW:.*]] = llvm.bitcast %arg0 : f32 to i32
// CHECK-NOT: llvm.inline_asm
// CHECK: llvm.mul %[[RAW]],
// CHECK: llvm.xor
tt.func private @experimental_fpsan_embed(%arg0: f32) -> i32 {
  %0 = tti.experimental_fpsan_embed %arg0 : (f32) -> i32
  tt.return %0 : i32
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_fpsan_unembed
// CHECK-NOT: tti.experimental_fpsan_unembed
// CHECK: llvm.mul %arg0,
// CHECK: llvm.xor
// CHECK: llvm.bitcast %{{.*}} : i32 to f32
tt.func private @experimental_fpsan_unembed(%arg0: i32) -> f32 {
  %0 = tti.experimental_fpsan_unembed %arg0 : (i32) -> f32
  tt.return %0 : f32
}
}

// -----

#local_gather_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>
#local_gather_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_local_gather
// CHECK: nvvm.mapa
// CHECK: llvm.load {{.*}} : !llvm.ptr<7> -> i32
tt.func private @experimental_local_gather(%out: !tt.ptr<i32>) {
  %src = ttg.local_alloc {allocation.offset = [0 : i32, 256 : i32]} : () -> !ttg.memdesc<2x32xi32, #local_gather_shared, #ttg.shared_memory, mutable>
  %idx = arith.constant dense<0> : tensor<2x32xi32, #local_gather_blocked>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %g = tti.experimental_local_gather %src[%idx] offsets = [%c1, %c0] {axis = 0 : i32} : !ttg.memdesc<2x32xi32, #local_gather_shared, #ttg.shared_memory, mutable>, tensor<2x32xi32, #local_gather_blocked> -> tensor<2x32xi32, #local_gather_blocked>
  %ptrs = tt.splat %out : !tt.ptr<i32> -> tensor<2x32x!tt.ptr<i32>, #local_gather_blocked>
  %offs = arith.constant dense<0> : tensor<2x32xi32, #local_gather_blocked>
  %out_ptrs = tt.addptr %ptrs, %offs : tensor<2x32x!tt.ptr<i32>, #local_gather_blocked>, tensor<2x32xi32, #local_gather_blocked>
  tt.store %out_ptrs, %g : tensor<2x32x!tt.ptr<i32>, #local_gather_blocked>
  tt.return
}
}
