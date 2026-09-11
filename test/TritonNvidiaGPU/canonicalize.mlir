// RUN: triton-opt %s -canonicalize | FileCheck %s --check-prefixes=CHECK,BARRIER
// RUN: triton-opt %s -gluon-canonicalize | FileCheck %s --check-prefix=BARRIER

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0], [64, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.target" = "cuda:80"} {

// CHECK-LABEL: @test_dce_tmem_alloc
tt.func @test_dce_tmem_alloc(%arg: tensor<128x4xi8, #linear>) {
  // CHECK-NOT: ttng.tmem_alloc
  %a = ttng.tmem_alloc %arg : (tensor<128x4xi8, #linear>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
  // CHECK-NEXT: tt.return
  tt.return
}

// CHECK-LABEL: @reinterpret_fold
tt.func @reinterpret_fold(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> {
  %0 = ttg.memdesc_reinterpret %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
  // CHECK-NEXT: return %arg0
  tt.return %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>
}

// CHECK-LABEL: @preserve_ld_acquire
llvm.func @preserve_ld_acquire(%arg0: !llvm.ptr<1>) {
  // CHECK: nvg.ld_acquire acquire, gpu, %arg0 : (!llvm.ptr<1>) -> i32
  %0 = nvg.ld_acquire acquire, gpu, %arg0 : (!llvm.ptr<1>) -> i32
  llvm.return
}

}  // end module

#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2], [4]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 4 : i32} {
// BARRIER-LABEL: @canonicalize_fromCTA
tt.func @canonicalize_fromCTA(%barrier: !ttg.memdesc<8xi64, #barrier, #smem, mutable>, %pred: i1) {
  // BARRIER-NEXT: ttng.barrier_expect %arg0, 16, %arg1 :
  ttng.barrier_expect %barrier, 16 {fromCTA = 7 : i32}, %pred : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
  // BARRIER-NEXT: ttng.barrier_expect %arg0, 16 {fromCTA = 5 : i32}, %arg1 :
  ttng.barrier_expect %barrier, 16 {fromCTA = 5 : i32}, %pred : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
  // BARRIER-NEXT: ttng.arrive_barrier %arg0, 1, %arg1 :
  ttng.arrive_barrier %barrier, 1, %pred {fromCTA = 7 : i32} : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
  // BARRIER-NEXT: ttng.arrive_barrier %arg0, 1, %arg1 {fromCTA = 5 : i32} :
  ttng.arrive_barrier %barrier, 1, %pred {fromCTA = 5 : i32} : !ttg.memdesc<8xi64, #barrier, #smem, mutable>
  tt.return
}
}

#local_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#shared_tma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#offsets_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#offsets = #ttg.slice<{dim = 0, parent = #offsets_parent}>
#data = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// BARRIER-LABEL: @false_barrier_predicates
// BARRIER-NEXT: %[[FALSE_WAIT:.*]] = arith.constant false
// BARRIER-NEXT: %[[PHASE:.*]] = arith.constant 0 : i32
// BARRIER-NEXT: ttng.wait_barrier %arg0, %[[PHASE]], %[[FALSE_WAIT]] deps %arg2 :
// BARRIER-NEXT: ttng.wait_barrier %arg0, %[[PHASE]], %arg1 :
// BARRIER-NEXT: ttng.tc_gen5_commit %arg0, %arg1 :
// BARRIER-NEXT: tt.return
tt.func @false_barrier_predicates(%bar: !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>, %pred: i1, %mem: !ttg.memdesc<16x64xf16, #shared_tma, #smem, mutable>) {
  %false = arith.constant false
  %phase = arith.constant 0 : i32
  ttng.barrier_expect %bar, 1048576, %false : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  ttng.arrive_barrier %bar, 1, %false : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  ttng.wait_barrier %bar, %phase, %false : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  ttng.tc_gen5_commit %bar, %false : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  // Keep the dependency-bearing wait so %mem stays live until the wait.
  ttng.wait_barrier %bar, %phase, %false deps %mem : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>, !ttg.memdesc<16x64xf16, #shared_tma, #smem, mutable>
  ttng.wait_barrier %bar, %phase, %pred : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  ttng.tc_gen5_commit %bar, %pred : !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  tt.return
}

// BARRIER-LABEL: @false_tma_predicates
// BARRIER-NEXT: tt.return
tt.func @false_tma_predicates(%desc: !tt.tensordesc<16x64xf16, #shared_tma>, %gather: !tt.tensordesc<1x64xf16, #shared_tma>, %mem: !ttg.memdesc<16x64xf16, #shared_tma, #smem, mutable>, %bar: !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>, %rows: tensor<16xi32, #offsets>) {
  %false = arith.constant false
  %zero = arith.constant 0 : i32
  ttng.async_tma_copy_global_to_local %desc[%zero, %zero] %mem, %bar, %false : !tt.tensordesc<16x64xf16, #shared_tma>, !ttg.memdesc<1xi64, #local_barrier, #smem, mutable> -> !ttg.memdesc<16x64xf16, #shared_tma, #smem, mutable>
  ttng.async_tma_gather %gather[%rows, %zero] %mem, %bar, %false : !tt.tensordesc<1x64xf16, #shared_tma>, tensor<16xi32, #offsets>, i32, !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>, !ttg.memdesc<16x64xf16, #shared_tma, #smem, mutable>, i1
  tt.return
}

// BARRIER-LABEL: @false_tmem_store
// BARRIER-NEXT: tt.return
tt.func @false_tmem_store(%value: tensor<128x128xf32, #data>, %mem: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>) {
  %false = arith.constant false
  ttng.tmem_store %value, %mem, %false : tensor<128x128xf32, #data> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

// MMA with completion barriers still synchronizes its partition when false.
// BARRIER-LABEL: @false_mma_keeps_synchronization
// BARRIER: %[[FALSE:.*]] = arith.constant false
// BARRIER: ttng.tc_gen5_mma %arg0, %arg1, %arg2, %[[FALSE]], %[[FALSE]], %arg3[
// BARRIER-NEXT: tt.return
tt.func @false_mma_keeps_synchronization(%a: !ttg.memdesc<128x128xf16, #shared_a, #smem>, %b: !ttg.memdesc<128x128xf16, #shared_b, #smem>, %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>) {
  %false = arith.constant false
  %true = arith.constant true
  ttng.tc_gen5_mma %a, %b, %d, %false, %false, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #shared_a, #smem>, !ttg.memdesc<128x128xf16, #shared_b, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>
  tt.return
}

// BARRIER-LABEL: @false_tmem_store_keeps_token
// BARRIER: %[[TOKEN:.*]] = ttng.tmem_store
// BARRIER-NEXT: tt.return %[[TOKEN]] : !ttg.async.token
tt.func @false_tmem_store_keeps_token(%value: tensor<128x128xf32, #data>, %mem: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %dep: !ttg.async.token) -> !ttg.async.token {
  %false = arith.constant false
  %token = ttng.tmem_store %value, %mem[%dep], %false : tensor<128x128xf32, #data> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return %token : !ttg.async.token
}
}
