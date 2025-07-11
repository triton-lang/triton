// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_check_outstanding_writes
// CHECK: @__assertfail
tt.func private @experimental_check_outstanding_writes(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %writeBars: !tt.ptr<i64>
) {
  tti.experimental_check_outstanding_writes %buf {%buffers, %writeBars(tensor<2xi64, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
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
// CHECK-LABEL: @experimental_check_outstanding_reads
// CHECK: @__assertfail
tt.func private @experimental_check_outstanding_reads(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_check_outstanding_reads %buf {%buffers, %readBars(tensor<2x2xi8, #blocked2>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_mark_as_write
// CHECK: st.global
tt.func private @experimental_mark_as_write(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %writeBars: !tt.ptr<i64>
) {
  tti.experimental_mark_as_write %buf, %mbar {%buffers, %writeBars(tensor<2xi64, #blocked>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<2xi64, #blocked>, !tt.ptr<i64>
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
// CHECK-LABEL: @experimental_mark_as_read
// CHECK: st.global
tt.func private @experimental_mark_as_read(
  %buf: !ttg.memdesc<32x32xf32, #shared, #smem, mutable>,
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %buffers: tensor<2xi64, #blocked>,
  %barriers: tensor<2xi64, #blocked>,
  %readBars: !tt.ptr<i8>
) {
  tti.experimental_mark_as_read %buf, %mbar {%buffers, %barriers, %readBars(tensor<2x2xi8, #blocked2>)} : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, tensor<2xi64, #blocked>, tensor<2xi64, #blocked>, !tt.ptr<i8>
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:90"} {
// CHECK-LABEL: @experimental_clear_write_barrier
// CHECK: st.global
tt.func private @experimental_clear_write_barrier(
  %mbar: !ttg.memdesc<1xi64, #shared1, #smem, mutable>,
  %writeBars: !tt.ptr<i64>
) {
  tti.experimental_clear_write_barrier %mbar {%writeBars(tensor<2xi64, #blocked>)} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !tt.ptr<i64>
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
