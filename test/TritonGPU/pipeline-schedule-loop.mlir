// RUN: triton-opt %s -split-input-file -tritongpu-test-pipeline-schedule-loop -canonicalize | FileCheck %s

#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #triton_gpu.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 2, maxPhase = 4, order = [0, 1], hasLeadingOffset = true}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>

module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.num-ctas" = 1 : i32} {
// CHECK-LABEL: @default_stages
tt.func @default_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: triton_gpu.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %b_ = tt.load %b_ptr {tt.latency = 2 : i32} : tensor<32x128x!tt.ptr<f16>, #BL>
    // CHECK: triton_gpu.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @small_load
// We should *not* assign latency to the load of b_ptr.
tt.func @small_load(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32}
    %a_ = tt.load %a_ptr {tt.latency = 2 : i32} : tensor<128x32x!tt.ptr<f16>, #AL>
    // CHECK: triton_gpu.convert_layout {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %a = triton_gpu.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}}
    // CHECK-NOT: loop.stage
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = triton_gpu.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    // CHECK: tt.dot {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32}
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}
}
