// RUN: triton-opt %s -canonicalize | FileCheck %s --check-prefixes=CHECK,BARRIER,COMMON
// RUN: triton-opt %s -gluon-canonicalize | FileCheck %s --check-prefixes=BARRIER,COMMON

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

// The predicate disables MMA, while the independent completion predicate stays true.
// BARRIER-LABEL: @false_mma_keeps_completion
// BARRIER: %[[FALSE:.*]] = arith.constant false
// BARRIER: ttng.tc_gen5_mma %arg0, %arg1, %arg2, %[[FALSE]], %[[FALSE]], %arg3[
// BARRIER-NEXT: tt.return
tt.func @false_mma_keeps_completion(%a: !ttg.memdesc<128x128xf16, #shared_a, #smem>, %b: !ttg.memdesc<128x128xf16, #shared_b, #smem>, %d: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %bar: !ttg.memdesc<1xi64, #local_barrier, #smem, mutable>) {
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

#div_layout = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-warps" = 4 : i32} {
// COMMON-LABEL: @fdiv_constant_in_range
// COMMON: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// COMMON: %[[RECIP_MIN:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_MIN]] : f32
// COMMON: %[[RECIP_MAX:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_MAX]] : f32
// COMMON: %[[RECIP_NEG_MIN:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_NEG_MIN]] : f32
// COMMON: %[[RECIP_NEG_MAX:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_NEG_MAX]] : f32
// COMMON: %[[RECIP_THREE:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_THREE]] : f32
// COMMON: %[[RECIP_NEG_THREE:.*]] = tt.approx_divf %[[ONE]], %{{.*}} : f32
// COMMON-NEXT: arith.mulf %arg0, %[[RECIP_NEG_THREE]] : f32
// COMMON-NEXT: tt.return
tt.func @fdiv_constant_in_range(%x: f32) -> (f32, f32, f32, f32, f32, f32) {
  %min = arith.constant 0x00800000 : f32
  %max = arith.constant 0x7E800000 : f32
  %neg_min = arith.constant 0x80800000 : f32
  %neg_max = arith.constant 0xFE800000 : f32
  %three = arith.constant 3.0 : f32
  %neg_three = arith.constant -3.0 : f32
  %a = arith.divf %x, %min : f32
  %b = arith.divf %x, %max : f32
  %c = arith.divf %x, %neg_min : f32
  %d = arith.divf %x, %neg_max : f32
  %e = arith.divf %x, %three : f32
  %f = arith.divf %x, %neg_three : f32
  tt.return %a, %b, %c, %d, %e, %f : f32, f32, f32, f32, f32, f32
}

// COMMON-LABEL: @fdiv_constant_out_of_range
// COMMON-COUNT-9: arith.divf
// COMMON-NEXT: tt.return
tt.func @fdiv_constant_out_of_range(%x: f32) -> (f32, f32, f32, f32, f32, f32, f32, f32, f32) {
  %below_min = arith.constant 0x007FFFFF : f32
  %above_max = arith.constant 0x7E800001 : f32
  %neg_below_min = arith.constant 0x807FFFFF : f32
  %neg_above_max = arith.constant 0xFE800001 : f32
  %zero = arith.constant 0.0 : f32
  %neg_zero = arith.constant -0.0 : f32
  %inf = arith.constant 0x7F800000 : f32
  %neg_inf = arith.constant 0xFF800000 : f32
  %nan = arith.constant 0x7FC00000 : f32
  %a = arith.divf %x, %below_min : f32
  %b = arith.divf %x, %above_max : f32
  %c = arith.divf %x, %neg_below_min : f32
  %d = arith.divf %x, %neg_above_max : f32
  %e = arith.divf %x, %zero : f32
  %f = arith.divf %x, %neg_zero : f32
  %g = arith.divf %x, %inf : f32
  %h = arith.divf %x, %neg_inf : f32
  %i = arith.divf %x, %nan : f32
  tt.return %a, %b, %c, %d, %e, %f, %g, %h, %i : f32, f32, f32, f32, f32, f32, f32, f32, f32
}

// COMMON-LABEL: @fdiv_constant_tensor
// COMMON-DAG: %[[ONE:.*]] = arith.constant 1.000000e+00 : f32
// COMMON-DAG: %[[THREE:.*]] = arith.constant 3.000000e+00 : f32
// COMMON-DAG: %[[ONES:.*]] = arith.constant dense<1.000000e+00> : tensor<4xf32,
// COMMON: %[[SCALAR_RECIP:.*]] = tt.approx_divf %[[ONE]], %[[THREE]] : f32
// COMMON-NEXT: %[[SPLAT:.*]] = tt.splat %[[SCALAR_RECIP]] : f32 -> tensor<4xf32,
// COMMON-NEXT: %[[A:.*]] = arith.mulf %arg0, %[[SPLAT]] : tensor<4xf32,
// COMMON-NEXT: %[[TENSOR_RECIP:.*]] = tt.approx_divf %[[ONES]], %{{.*}} : tensor<4xf32,
// COMMON-NEXT: %[[B:.*]] = arith.mulf %arg0, %[[TENSOR_RECIP]] : tensor<4xf32,
// COMMON-NEXT: %[[C:.*]] = arith.divf
// COMMON-NEXT: tt.return %[[A]], %[[B]], %[[C]]
tt.func @fdiv_constant_tensor(%x: tensor<4xf32, #div_layout>) -> (tensor<4xf32, #div_layout>, tensor<4xf32, #div_layout>, tensor<4xf32, #div_layout>) {
  %splat = arith.constant dense<3.0> : tensor<4xf32, #div_layout>
  %in_range = arith.constant dense<[0x00800000, 0x7E800000, 0x80800000, 0xFE800000]> : tensor<4xf32, #div_layout>
  %out_of_range = arith.constant dense<[3.0, -3.0, 0.0, 4.0]> : tensor<4xf32, #div_layout>
  %a = arith.divf %x, %splat : tensor<4xf32, #div_layout>
  %b = arith.divf %x, %in_range : tensor<4xf32, #div_layout>
  %c = arith.divf %x, %out_of_range : tensor<4xf32, #div_layout>
  tt.return %a, %b, %c : tensor<4xf32, #div_layout>, tensor<4xf32, #div_layout>, tensor<4xf32, #div_layout>
}

// COMMON-LABEL: @fdiv_runtime_denominator
// COMMON: arith.divf
tt.func @fdiv_runtime_denominator(%x: f32, %y: f32) -> f32 {
  %a = arith.divf %x, %y : f32
  tt.return %a : f32
}

// COMMON-LABEL: @fdiv_constant_f64
// COMMON: arith.divf
tt.func @fdiv_constant_f64(%x: f64) -> f64 {
  %three = arith.constant 3.0 : f64
  %a = arith.divf %x, %three : f64
  tt.return %a : f64
}

// COMMON-LABEL: @fdiv_constant_precise
// COMMON: tt.precise_divf
tt.func @fdiv_constant_precise(%x: f32) -> f32 {
  %three = arith.constant 3.0 : f32
  %a = tt.precise_divf %x, %three : f32
  tt.return %a : f32
}
}

module attributes {"ttg.target" = "hip:gfx942"} {
// COMMON-LABEL: @fdiv_constant_amd
// COMMON: arith.divf
// COMMON-NEXT: tt.return
tt.func @fdiv_constant_amd(%x: f32) -> f32 {
  %three = arith.constant 3.0 : f32
  %a = arith.divf %x, %three : f32
  tt.return %a : f32
}
}

module {
// COMMON-LABEL: @fdiv_constant_no_target
// COMMON: arith.divf
// COMMON-NEXT: tt.return
tt.func @fdiv_constant_no_target(%x: f32) -> f32 {
  %three = arith.constant 3.0 : f32
  %a = arith.divf %x, %three : f32
  tt.return %a : f32
}
}
