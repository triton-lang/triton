// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_buffer_descriptors_tmem
// CHECK: llvm.mlir.constant(4294967295 : i64) : i64
// CHECK: llvm.mlir.constant(34359738368 : i64) : i64
// CHECK: llvm.mlir.constant(68719476736 : i64) : i64
tt.func private @experimental_buffer_descriptors_tmem() {
  tti.experimental_buffer_descriptors [0, 42], [8, 16], tensor_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_buffer_descriptors_shared
// CHECK: llvm.mlir.constant(4294967295 : i64) : i64
// CHECK: llvm.mlir.constant(17179869184 : i64) : i64
// CHECK: llvm.mlir.constant(51539607552 : i64) : i64
tt.func private @experimental_buffer_descriptors_shared() {
  tti.experimental_buffer_descriptors [0, 42], [4, 12], shared_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_assert_in_thread_any
// CHECK: %[[E0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(i1, i1)>
// CHECK: %[[E1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(i1, i1)>
// CHECK: %[[INIT:.+]] = llvm.mlir.constant(false) : i1
// CHECK: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
// CHECK: %[[OR0:.+]] = llvm.or %[[INIT]], %[[E0]] : i1
// CHECK: %[[OR1:.+]] = llvm.or %[[OR0]], %[[E1]] : i1
// CHECK: %[[XOR:.+]] = llvm.xor %[[OR1]]

// CHECK: @__assertfail
tt.func private @experimental_assert_in_thread_any(
  %condition: tensor<2xi1, #blocked>,
  %message: !llvm.ptr<8>
) {
  tti.experimental_assert_in_thread %condition, "test" {check_any = true} : tensor<2xi1, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_assert_in_thread_all
// CHECK: %[[E0:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(i1, i1)>
// CHECK: %[[E1:.+]] = llvm.extractvalue %arg0[1] : !llvm.struct<(i1, i1)>
// CHECK: %[[INIT:.+]] = llvm.mlir.constant(true) : i1
// CHECK: %[[FALSE:.+]] = llvm.mlir.constant(false) : i1
// CHECK: %[[AND0:.+]] = llvm.and %[[INIT]], %[[E0]] : i1
// CHECK: %[[AND1:.+]] = llvm.and %[[AND0]], %[[E1]] : i1
// CHECK: %[[XOR:.+]] = llvm.xor %[[AND1]]

// CHECK: @__assertfail
tt.func private @experimental_assert_in_thread_all(
  %condition: tensor<2xi1, #blocked>,
  %message: !llvm.ptr<8>
) {
  tti.experimental_assert_in_thread %condition, "test" {check_any = false} : tensor<2xi1, #blocked>
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
