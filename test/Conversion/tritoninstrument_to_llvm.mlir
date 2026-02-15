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
// CHECK: llvm.mlir.constant(4294967295 : i64) : i64
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
// CHECK: nvvm.barrier0
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
// CHECK: nvvm.barrier0
// CHECK: atom.global.gpu.acq_rel.exch.b32
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
