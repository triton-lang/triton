// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_buffer_pointers_tmem
// CHECK:nvgpu.tensor_memory_base
tt.func private @experimental_buffer_pointers_tmem() {
  tti.experimental_buffer_pointers [0, 42], tensor_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_set_waiting
// CHECK: st.global
tt.func private @experimental_set_waiting(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %waiting: !tt.ptr<i32>
) {
  %phase = arith.constant 1 : i32
  tti.experimental_set_waiting %mbar, 5, %phase{%barriers, %waiting(tensor<2xi32, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_check_all_active_waiting
// CHECK: @__assertfail
tt.func private @experimental_check_all_active_waiting(
  %barriers: tensor<2xi64, #blocked>,
  %waiting: !tt.ptr<i32>,
  %states: !tt.ptr<i32>
) {
  // active mask 3 -> two active base threads (bits 0 and 1)
  tti.experimental_check_all_active_waiting 3, %barriers, %waiting(tensor<2xi32, #blocked>), %states(tensor<2xi32, #blocked>) : tensor<2xi64, #blocked>, !tt.ptr<i32>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_init_barrier_state
// CHECK: shl
// CHECK: st.global
tt.func private @experimental_init_barrier_state(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %states: !tt.ptr<i32>
) {
  tti.experimental_init_barrier_state %mbar, 3{%barriers, %states(tensor<2xi32, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_verify_barrier_arrive
// CHECK: @__assertfail
tt.func private @experimental_verify_barrier_arrive(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %states: !tt.ptr<i32>,
  %pred: i1
) {
  tti.experimental_verify_barrier_arrive %mbar, 2{%barriers, %states(tensor<2xi32, #blocked>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_update_barrier_state
// CHECK: st.global
tt.func private @experimental_update_barrier_state(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %states: !tt.ptr<i32>,
  %pred: i1
) {
  tti.experimental_update_barrier_state %mbar, 2{%barriers, %states(tensor<2xi32, #blocked>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_waiting
// CHECK: st.global
tt.func private @experimental_clear_waiting(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %waiting: !tt.ptr<i32>
) {
  tti.experimental_clear_waiting %mbar, 7{%barriers, %waiting(tensor<2xi32, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i32>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_buffer_pointers_shared
// CHECK: llvm.ptrtoint %arg0
tt.func private @experimental_buffer_pointers_shared() {
  tti.experimental_buffer_pointers [0, 42], shared_mem : tensor<2xi64, #blocked>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_set_write_visibility
// CHECK: st.global
tt.func private @experimental_set_write_visibility(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_set_write_visibility %buf, 42{%buffers, %writeVisibility(tensor<2xi64, #blocked>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_set_read_visibility
// CHECK: st.global
tt.func private @experimental_set_read_visibility(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_set_read_visibility %buf, 42{%buffers, %readVisibility(tensor<2x64xi64, #blocked3>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_copy_write_visibility
// CHECK: st.global
tt.func private @experimental_copy_write_visibility(
  %writeVis: !tt.ptr<i64>
) {
  tti.experimental_copy_write_visibility 0, 6{%writeVis(tensor<2xi64, #blocked>)} : !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blockedThreads = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_copy_read_visibility
// CHECK: st.global
tt.func private @experimental_copy_read_visibility(
  %readVis: !tt.ptr<i64>
) {
  tti.experimental_copy_read_visibility 0, 6{%readVis(tensor<2x64xi64, #blockedThreads>)} : !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_write_tracking
// CHECK: st.global
tt.func private @experimental_clear_write_tracking(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_clear_write_tracking %buf{%buffers, %writeTracking(tensor<2x4xi8, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_read_visibility
// CHECK: st.global
tt.func private @experimental_clear_read_visibility(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_clear_read_visibility %buf{%buffers, %readVisibility(tensor<2x64xi64, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_read_tracking
// CHECK: st.global
tt.func private @experimental_clear_read_tracking(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_clear_read_tracking %buf{%buffers, %readTracking(tensor<2x4xi64, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_track_visible_writes
// CHECK: st.global
tt.func private @experimental_track_visible_writes(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_track_visible_writes %mbar, 16{%barriers, %writeVisibility(tensor<2xi64, #blocked>), %writeTracking(tensor<2x4xi8, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i64>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_track_visible_reads
// CHECK: st.global
tt.func private @experimental_track_visible_reads(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_track_visible_reads %mbar, 16{%barriers, %readVisibility(tensor<2x64xi64, #blocked3>), %readTracking(tensor<2x4xi64, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i64>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_transfer_visible_writes
// CHECK: st.global
tt.func private @experimental_transfer_visible_writes(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_transfer_visible_writes %mbar, 16{%barriers, %writeVisibility(tensor<2xi64, #blocked>), %writeTracking(tensor<2x4xi8, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i64>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_transfer_visible_reads
// CHECK: st.global
tt.func private @experimental_transfer_visible_reads(
  %mbar: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_transfer_visible_reads %mbar, 16{%barriers, %readVisibility(tensor<2x64xi64, #blocked3>), %readTracking(tensor<2x4xi64, #blocked2>)}, %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i64>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_verify_write_visibility
// CHECK: @__assertfail
tt.func private @experimental_verify_write_visibility(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_verify_write_visibility %buf, 16{%buffers, %writeVisibility(tensor<2xi64, #blocked>)}, "", %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_verify_read_visibility
// CHECK: @__assertfail
tt.func private @experimental_verify_read_visibility(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeVisibility: !tt.ptr<i64>,
  %writeTracking: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>,
  %readTracking: !tt.ptr<i64>,
  %pred: i1
) {
  tti.experimental_verify_read_visibility %buf, 16{%buffers, %readVisibility(tensor<2x64xi64, #blocked3>)}, "", %pred : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_stage_access_for_commit
// CHECK: st.global
tt.func private @experimental_stage_access_for_commit(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %writeCommits: !tt.ptr<i8>
) {
  tti.experimental_stage_access_for_commit %buf, 0{%buffers, %writeCommits(tensor<2x16xi8, #blocked2>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_check_outstanding_commits
// CHECK: @__assertfail
tt.func private @experimental_check_outstanding_commits(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %writeCommits: !tt.ptr<i8>
) {
  tti.experimental_check_outstanding_commits %buf{%buffers, %writeCommits(tensor<2x16xi8, #blocked2>)}, "dummy" : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_commit_accesses
// CHECK: st.global
tt.func private @experimental_commit_accesses(
  %writeCommits: !tt.ptr<i8>
) {
  tti.experimental_commit_accesses 0{%writeCommits(tensor<2x16xi8, #blocked2>)} : !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_outstanding_commits_transfer_writes
// CHECK: st.global
tt.func private @experimental_clear_outstanding_commits_transfer_writes(
  %outstandingCommits: !tt.ptr<i8>,
  %writeVisibility: !tt.ptr<i64>
) {
  tti.experimental_clear_outstanding_commits_transfer_writes 0, 4294967296{%outstandingCommits(tensor<2x16xi8, #blocked2>)}, %writeVisibility(tensor<2xi64, #blocked>) {outstandingNum = 42 : i32} : !tt.ptr<i8>, !tt.ptr<i64>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 64], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_outstanding_commits_transfer_reads
// CHECK: st.global
tt.func private @experimental_clear_outstanding_commits_transfer_reads(
  %outstandingCommits: !tt.ptr<i8>,
  %readVisibility: !tt.ptr<i64>
) {
  tti.experimental_clear_outstanding_commits_transfer_reads 0, 4294967296{%outstandingCommits(tensor<2x16xi8, #blocked2>)}, %readVisibility(tensor<2x64xi64, #blocked3>) {outstandingNum = 42 : i32} : !tt.ptr<i8>, !tt.ptr<i64>
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
// CHECK: nvvm.barrier0
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
// CHECK: nvvm.barrier0
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
