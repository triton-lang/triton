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
// CHECK-LABEL: @experimental_check_write_state
// CHECK: @__assertfail
tt.func private @experimental_check_write_state(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeBars: !tt.ptr<i8>,
  %writeState: !tt.ptr<i8>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_check_write_state %buf {%buffers, %writeBars(tensor<2x4xi8, #blocked2>), %writeState(tensor<2xi8, #blocked>)} pipelined false : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_check_read_barriers
// CHECK: @__assertfail
tt.func private @experimental_check_read_barriers(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_check_read_barriers %buf {%buffers, %readBars(tensor<2x2xi8, #blocked2>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_set_write_state
// CHECK: st.global
tt.func private @experimental_set_write_state(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeBars: !tt.ptr<i8>,
  %writeState: !tt.ptr<i8>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_set_write_state %buf {%buffers, %writeState(tensor<2xi8, #blocked>)} pipelined false : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
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
// CHECK-LABEL: @experimental_check_barrier_writes_cleared
// CHECK: @__assertfail
tt.func private @experimental_check_barrier_writes_cleared(
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeBars: !tt.ptr<i8>,
  %writeState: !tt.ptr<i8>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_check_barrier_writes_cleared %mbar {%barriers, %writeBars(tensor<2x4xi8, #blocked2>)} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i8>
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
// CHECK-LABEL: @experimental_commit_write_with_barrier
// CHECK: st.global
tt.func private @experimental_commit_write_with_barrier(
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeBars: !tt.ptr<i8>,
  %writeState: !tt.ptr<i8>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_commit_write_with_barrier %mbar {%barriers, %writeBars(tensor<2x4xi8, #blocked2>), %writeState(tensor<2xi8, #blocked>)} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i8>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_set_read_barrier
// CHECK: st.global
tt.func private @experimental_set_read_barrier(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<2xi64, #blocked>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_set_read_barrier %buf, %mbar {%buffers, %barriers, %readBars(tensor<2x2xi8, #blocked2>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<2xi64, #blocked>, tensor<2xi64, #blocked>, !tt.ptr<i8>
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
// CHECK-LABEL: @experimental_clear_write_barrier
// CHECK: st.global
tt.func private @experimental_clear_write_barrier(
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<4xi64, #blocked1>,
  %writeBars: !tt.ptr<i8>,
  %writeState: !tt.ptr<i8>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_clear_write_barrier %mbar {%barriers, %writeBars(tensor<2x4xi8, #blocked2>), %writeState(tensor<2xi8, #blocked>)} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<4xi64, #blocked1>, !tt.ptr<i8>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_read_barrier
// CHECK: st.global
tt.func private @experimental_clear_read_barrier(
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %barriers: tensor<2xi64, #blocked>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_clear_read_barrier %mbar {%barriers, %readBars(tensor<2x2xi8, #blocked2>)} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
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
  tti.experimental_stage_access_for_commit %buf {%buffers, %writeCommits(tensor<2xi8, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
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
  tti.experimental_check_outstanding_commits %buf {%buffers, %writeCommits(tensor<2xi8, #blocked>)}, "dummy" : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_commit_accesses
// CHECK: st.global
tt.func private @experimental_commit_accesses(
  %writeCommits: !tt.ptr<i8>
) {
  tti.experimental_commit_accesses {%writeCommits(tensor<2xi8, #blocked>)} : !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_outstanding_commits
// CHECK: st.global
tt.func private @experimental_clear_outstanding_commits(
  %outstandingCommits: !tt.ptr<i8>
) {
  tti.experimental_clear_outstanding_commits {%outstandingCommits(tensor<2xi8, #blocked>)}, 42 : !tt.ptr<i8>
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
