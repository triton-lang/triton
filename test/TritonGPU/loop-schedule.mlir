// RUN: triton-opt %s -split-input-file -tritongpu-loop-scheduling=num-stages=3 | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#CLs0 = #triton_gpu.slice<{parent=#C, dim=0}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
// CHECK-LABLE: @matmul_loop_load_acc
// CHECK: tt.load %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
// CHECK: tt.load %{{.*}} {loop.cluster = 3 : i32, loop.stage = 0 : i32}
// CHECK: tt.load %{{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
// CHECK: tt.dot {{.*}} {loop.cluster = 1 : i32, loop.stage = 2 : i32}
tt.func @matmul_loop_load_acc(%lb : index, %ub : index, %step : index,
                  %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
                  %C : !tt.ptr<f32> {tt.divisibility = 16 : i32},
                  %c_init: tensor<128x128xf32, #C>) -> tensor<128x128xf32, #C> {

  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
  // C ptrs
  %c_ptr_splat = tt.splat %C : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #C>
  %c_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #CLs0>
  %c_tmp1 = tt.expand_dims %c_tmp0 {axis = 0 : i32} : tensor<128xi32, #CLs0> -> tensor<1x128xi32, #C>
  %c_offs = tt.broadcast %c_tmp1 : tensor<1x128xi32, #C> -> tensor<128x128xi32, #C>
  %c_ptr_init = tt.addptr %c_ptr_splat, %c_offs : tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xi32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c_off = arith.constant dense<4> : tensor<128x128xi32, #C>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %c_ptr = %c_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xf32, #C>) {
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %c_ = tt.load %c_ptr : tensor<128x128x!tt.ptr<f32>, #C>
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    %next_c_ptr = tt.addptr %c_ptr, %c_off : tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xi32, #C>
    scf.yield %next_a_ptr, %next_b_ptr, %next_c_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128x!tt.ptr<f32>, #C>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}
}
