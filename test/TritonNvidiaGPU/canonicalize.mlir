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
