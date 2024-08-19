// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=4 -debug-only=triton-matmul-loop-pipeline 2>&1 | FileCheck %s --check-prefix=DEBUG
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=4 | FileCheck %s

// DEBUG: Final coarse schedule:
// DEBUG: Ops in stage 2
// DEBUG: triton_nvidia_gpu.warp_group_dot
// DEBUG: Ops in stage 3
// DEBUG: triton_nvidia_gpu.wait_barrier
// DEBUG: triton_nvidia_gpu.wait_barrier
// DEBUG: Original loop:

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_tma(%arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg6: f32, %arg8: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg9: i32, %arg10: i32, %arg11: i64, %arg14: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %25 = tt.experimental_descriptor_load %arg3[%arg9, %c0_i32] : !tt.ptr<i8> -> tensor<128x128xf16, #blocked1>
    %26 = triton_gpu.local_alloc %25 : (tensor<128x128xf16, #blocked1>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
    %27 = arith.extsi %arg14 : i32 to i64
    %28 = tt.splat %arg6 : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %29 = tt.splat %arg6 : f32 -> tensor<128x128xf32, #mma>
    %30 = arith.extsi %arg17 : i32 to i64
    // CHECK: tt.experimental_descriptor_load
    // CHECK: %[[QLOC:.+]] = triton_gpu.local_alloc {{.*}}tt.memdesc<128x128xf16
    // CHECK: %[[KLOC:.+]] = triton_gpu.local_alloc {{.*}}tt.memdesc<4x128x128xf16
    // CHECK: %[[VLOC:.+]] = triton_gpu.local_alloc {{.*}}tt.memdesc<4x128x128xf16
    // CHECK: %[[KBAR:.+]] = triton_gpu.local_alloc {{.*}}tt.memdesc<4xi64
    // CHECK: %[[VBAR:.+]] = triton_gpu.local_alloc {{.*}}tt.memdesc<4xi64
    // stage 0 iteration 0
    // CHECK: %[[K0:.+]] = triton_gpu.memdesc_subview %[[KLOC]][%c0_i32, %c0_i32, %c0_i32]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local{{.*}} %[[K0]]
    // stage 0 iteration 1
    // CHECK: %[[K1:.+]] = triton_gpu.memdesc_subview %[[KLOC]][%c1_i32, %c0_i32, %c0_i32]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local{{.*}} %[[K1]]
    // stage 1 iteration 0
    // CHECK: %[[V0:.+]] = triton_gpu.memdesc_subview %[[VLOC]][%c0_i32, %c0_i32, %c0_i32]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local{{.*}} %[[V0]]
    // stage 2 iteration 0
    // CHECK: %[[FIRSTDOT:.+]] = triton_nvidia_gpu.warp_group_dot
    // stage 0 iteration 2
    // CHECK: %[[K2:.+]] = triton_gpu.memdesc_subview %[[KLOC]][%c2_i32, %c0_i32, %c0_i32]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local{{.*}} %[[K2]]
    // stage 1 iteration 1
    // CHECK: %[[V1:.+]] = triton_gpu.memdesc_subview %[[VLOC]][%c1_i32, %c0_i32, %c0_i32]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local{{.*}} %[[V1]]
    // CHECK: scf.for {{.*}} %[[ARG:.+]] = %[[FIRSTDOT]]
    // CHECK: %[[KBARSUB:.+]] = triton_gpu.memdesc_subview %[[KBAR]][%[[KBARIDX:.+]]]
    // CHECK: triton_nvidia_gpu.wait_barrier %[[KBARSUB]]
    // CHECK: %[[KLOOP:.+]] = triton_gpu.memdesc_subview %[[KLOC]]
    // CHECK: tt.trans %[[KLOOP]]
    // CHECK: %[[FIRSTDOTLOOP:.+]] = triton_nvidia_gpu.warp_group_dot
    // CHECK: %[[WAIT:.+]]:{{[0-9]+}} = triton_nvidia_gpu.warp_group_dot_wait
    // CHECK: "tt.reduce"(%[[ARG]])
    // CHECK: %[[VBARSUB:.+]] = triton_gpu.memdesc_subview %[[VBAR]][%[[KBARIDX]]]
    // CHECK: triton_nvidia_gpu.wait_barrier %[[VBARSUB]]
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local
    // CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local
    // CHECK: scf.yield {{.*}}%[[WAIT]]#0
    // arg26 is acc
    %31:1 = scf.for %arg24 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %cst_2) -> (tensor<128x128xf32, #mma>)  : i32 {
      %48 = arith.divsi %arg11, %27 : i64
      %49 = arith.trunci %48 : i64 to i32
      %50 = arith.addi %arg24, %49 : i32
      // loads in different stages
      %51 = tt.experimental_descriptor_load %arg4[%50, %c0_i32] {loop.stage = 0 : i32} : !tt.ptr<i8> -> tensor<128x128xf16, #blocked1>
      %52 = triton_gpu.local_alloc %51 : (tensor<128x128xf16, #blocked1>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
      %53 = tt.trans %52 {order = array<i32: 1, 0>} : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory>
      %54 = triton_nvidia_gpu.warp_group_dot %26, %53, %cst_2 {inputPrecision = 0 : i32, loop.stage = 2} : !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x128xf16, #shared1, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      %55 = "tt.reduce"(%54) <{axis = 1 : i32}> ({
      ^bb0(%arg28: f32 loc(unknown), %arg29: f32 loc(unknown)):
        %80 = arith.maxnumf %arg28, %arg29 : f32
        tt.reduce.return %80 : f32
      }) : (tensor<128x128xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %56 = arith.mulf %55, %28 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %58 = arith.mulf %54, %29 : tensor<128x128xf32, #mma>
      %59 = tt.expand_dims %56 {axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %60 = tt.broadcast %59 : tensor<128x1xf32, #mma> -> tensor<128x128xf32, #mma>
      %61 = arith.subf %58, %60 : tensor<128x128xf32, #mma>
      %62 = math.exp2 %61 : tensor<128x128xf32, #mma>
      %71 = arith.divsi %arg11, %30 : i64
      %72 = arith.extsi %arg24 : i32 to i64
      %73 = arith.addi %71, %72 : i64
      %74 = arith.trunci %73 : i64 to i32
      %75 = tt.experimental_descriptor_load %arg5[%74, %c0_i32] {loop.stage = 1 : i32} : !tt.ptr<i8> -> tensor<128x128xf16, #blocked1>
      %76 = triton_gpu.local_alloc %75 : (tensor<128x128xf16, #blocked1>) -> !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory>
      %77 = arith.truncf %62 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
      %78 = triton_gpu.convert_layout %77 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      %79 = triton_nvidia_gpu.warp_group_dot %78, %76, %arg26 {inputPrecision = 0 : i32, loop.stage = 3 : i32} : tensor<128x128xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<128x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x128xf32, #mma>
      scf.yield %79 : tensor<128x128xf32, #mma>
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    %42 = arith.truncf %31#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %43 = triton_gpu.convert_layout %42 : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked1>
    tt.experimental_descriptor_store %arg8[%arg10, %c0_i32], %43 : !tt.ptr<i8>, tensor<128x128xf16, #blocked1>
    tt.return
  }
}
