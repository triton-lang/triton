// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-code-partition=num-buffers=1 | FileCheck %s

// CHECK-LABEL: @matmul_kernel_one_consumer
// CHECK: %[[#TASKID:]] = triton_nvidia_gpu.get_async_task_id : i32
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[#WG0:]] = arith.cmpi eq, %[[#TASKID]], %c0_i32 : i32
// CHECK: scf.if %[[#WG0]]
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[#WG1:]] = arith.cmpi eq, %[[#TASKID]], %c1_i32 : i32
// CHECK: scf.if %[[#WG1]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_gpu.local_load
// CHECK: triton_gpu.local_load
// CHECK: tt.dot
// CHECK: triton_nvidia_gpu.consumer_release


#blocked = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_one_consumer(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c255_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 255 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 127 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
    %cst_0 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<256x128xf16, #blocked1>
    %cst_1 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf16, #blocked2>
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 256 : i32
    %cst_2 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<256> : tensor<128x256xi32, #blocked2>
    %0 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %1 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %2 = arith.divsi %1, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %3 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %4 = arith.divsi %3, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %5 = arith.muli %4, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %6 = arith.divsi %0, %5 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %7 = arith.muli %6, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %8 = arith.subi %2, %7 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %9 = arith.minsi %8, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %10 = arith.remsi %0, %5 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %11 = arith.remsi %10, %9 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %12 = arith.addi %7, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %13 = arith.divsi %10, %9 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %14 = arith.muli %12, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %15 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {async_task_id = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.splat %14 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.splat %14 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %20 = arith.addi %18, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %21 = arith.addi %19, %16 {async_task_id = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.splat %arg3 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %23 = arith.remsi %20, %22 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %24 = arith.muli %13, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %25 = tt.splat %24 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %17 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %23 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %30 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked2>
    %31 = arith.muli %29, %30 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
    %32 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = tt.expand_dims %32 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %34 = tt.broadcast %31 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %35 = tt.broadcast %33 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %36 = arith.addi %34, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256xi32, #blocked2>
    %37 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
    %38 = tt.addptr %37, %36 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
    %39 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %40 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %41 = tt.expand_dims %39 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %42 = tt.expand_dims %40 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<256x1xi32, #blocked1>
    %43 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked1>
    %44 = arith.muli %41, %43 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1>
    %45 = tt.expand_dims %28 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %46 = tt.broadcast %44 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %47 = tt.broadcast %45 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %48 = arith.addi %46, %47 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128xi32, #blocked1>
    %49 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked1>
    %50 = tt.addptr %49, %48 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
    %51 = arith.addi %arg5, %c255_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %52 = arith.divsi %51, %c256_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
    %53 = arith.muli %arg7, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %54 = tt.splat %53 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x128xi32, #blocked1>
    %55:3 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %38, %arg12 = %50) -> (tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %74 = arith.muli %arg9, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %75 = arith.subi %arg5, %74 {async_task_id = dense<0> : vector<1xi32>} : i32
      %76 = tt.splat %75 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x256xi32, #blocked2>
      %77 = arith.cmpi slt, %33, %76 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked2>
      %78 = tt.broadcast %77 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %79 = tt.load %arg11, %78, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>
      %80 = tt.splat %75 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked1>
      %81 = arith.cmpi slt, %42, %80 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked1>
      %82 = tt.broadcast %81 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi1, #blocked1> -> tensor<256x128xi1, #blocked1>
      %83 = tt.load %arg12, %82, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>
      %84 = triton_gpu.convert_layout %79 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>
      %85 = triton_gpu.convert_layout %83 {async_task_id = dense<1> : vector<1xi32>} : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>
      %86 = tt.dot %84, %85, %arg10, inputPrecision = tf32 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
      %87 = tt.addptr %arg11, %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
      %88 = tt.addptr %arg12, %54 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
      scf.yield {async_task_id = dense<[0, 1]> : vector<2xi32>} %86, %87, %88 : tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>
    } {async_task_id = dense<[0, 1]> : vector<2xi32>}
    %56 = arith.truncf %55#0 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %57 = tt.expand_dims %21 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %58 = tt.splat %arg8 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %59 = arith.muli %58, %57 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    %60 = tt.splat %arg2 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %61 = tt.addptr %60, %59 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %62 = tt.expand_dims %26 {async_task_id = dense<1> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %63 = tt.broadcast %61 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x128x!tt.ptr<f16>, #blocked1>
    %64 = tt.broadcast %62 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<128x128xi32, #blocked1>
    %65 = tt.addptr %63, %64 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked1>, tensor<128x128xi32, #blocked1>
    %66 = tt.splat %arg3 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %67 = arith.cmpi slt, %57, %66 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    %68 = tt.splat %arg4 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<1x128xi32, #blocked1>
    %69 = arith.cmpi slt, %62, %68 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked1>
    %70 = tt.broadcast %67 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %71 = tt.broadcast %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi1, #blocked1> -> tensor<128x128xi1, #blocked1>
    %72 = arith.andi %70, %71 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked1>
    %73 = triton_gpu.convert_layout %56 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
    tt.store %65, %73, %72 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----


// CHECK-LABEL: @matmul_kernel_two_consumers
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_gpu.async_copy_global_to_local
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: scf.if
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: triton_nvidia_gpu.consumer_release

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_two_consumers(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
    %cst_0 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_1 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x128xf16, #blocked1>
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
    %c63_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i32
    %cst_2 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.divsi %1, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = arith.divsi %3, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %5 = arith.muli %4, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %6 = arith.divsi %0, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %7 = arith.muli %6, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %8 = arith.subi %2, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %9 = arith.minsi %8, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %10 = arith.remsi %0, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %11 = arith.remsi %10, %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %12 = arith.addi %7, %11 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %13 = arith.divsi %10, %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %14 = arith.muli %12, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %15 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %24 = tt.make_range {async_task_id = dense<2> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %25 = arith.addi %17, %23 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.addi %18, %24 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.remsi %25, %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %28 = arith.muli %13, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %29 = tt.make_range {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %30 = tt.splat %28 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %31 = arith.addi %30, %29 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %32 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %33 = arith.remsi %31, %32 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %34 = tt.expand_dims %22 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %35 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
    %36 = arith.muli %34, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked>
    %37 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %38 = tt.expand_dims %37 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %39 = tt.broadcast %36 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %40 = tt.broadcast %38 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %41 = arith.addi %39, %40 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64xi32, #blocked>
    %42 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x64x!tt.ptr<f16>, #blocked>
    %43 = tt.addptr %42, %41 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %44 = tt.expand_dims %27 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %45 = arith.muli %44, %35 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked>
    %46 = tt.broadcast %45 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %47 = arith.addi %46, %40 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64xi32, #blocked>
    %48 = tt.addptr %42, %47 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
    %49 = tt.expand_dims %16 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %50 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %51 = arith.muli %49, %50 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %52 = tt.expand_dims %33 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %53 = tt.broadcast %51 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %54 = tt.broadcast %52 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %55 = arith.addi %53, %54 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128xi32, #blocked1>
    %56 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %57 = tt.addptr %56, %55 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %58 = arith.addi %arg5, %c63_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %59 = arith.divsi %58, %c64_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %60 = tt.expand_dims %37 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %61 = tt.expand_dims %16 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %62 = arith.muli %arg7, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %63 = tt.splat %62 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x128xi32, #blocked1>
    %true = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
    %false = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
    %true_3 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
    %false_4 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
    %64:5 = scf.for %arg9 = %c0_i32 to %59 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %43, %arg13 = %57, %arg14 = %48) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %93 = arith.muli %arg9, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %94 = arith.subi %arg5, %93 {async_task_id = dense<0> : vector<1xi32>} : i32
      %95 = tt.splat %94 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x64xi32, #blocked>
      %96 = arith.cmpi slt, %60, %95 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked>
      %97 = tt.broadcast %96 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %98 = tt.load %arg12, %97, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %99 = triton_gpu.local_alloc %98 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x64xf16, #blocked>) -> !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>
      %100 = tt.splat %94 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
      %101 = arith.cmpi slt, %61, %100 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
      %102 = tt.broadcast %101 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
      %103 = tt.load %arg13, %102, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %104 = triton_gpu.local_alloc %103 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xf16, #blocked1>) -> !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory>
      %105 = tt.load %arg14, %97, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %106 = triton_gpu.local_alloc %105 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x64xf16, #blocked>) -> !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory>
      %107 = triton_nvidia_gpu.warp_group_dot %99, %104, %arg10 {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma>
      %108 = triton_nvidia_gpu.warp_group_dot %106, %104, %arg11 {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x128xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma>
      %109 = tt.addptr %arg12, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %110 = tt.addptr %arg14, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %111 = tt.addptr %arg13, %63 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
      scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %107, %108, %109, %111, %110 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    %65 = arith.truncf %64#0 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %66 = arith.truncf %64#1 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %67 = tt.expand_dims %20 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %68 = tt.splat %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %69 = arith.muli %68, %67 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %70 = tt.splat %arg2 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %71 = tt.addptr %70, %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %72 = tt.expand_dims %31 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %73 = tt.broadcast %71 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %74 = tt.broadcast %72 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked1> -> tensor<64x128xi32, #blocked1>
    %75 = tt.addptr %73, %74 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %76 = tt.expand_dims %26 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %77 = arith.muli %68, %76 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %78 = tt.addptr %70, %77 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %79 = tt.broadcast %78 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<f16>, #blocked1> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    %80 = tt.addptr %79, %74 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
    %81 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %82 = arith.cmpi slt, %67, %81 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %83 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<1x128xi32, #blocked1>
    %84 = arith.cmpi slt, %72, %83 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked1>
    %85 = tt.broadcast %82 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %86 = tt.broadcast %84 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %87 = arith.andi %85, %86 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xi1, #blocked1>
    %88 = arith.cmpi slt, %76, %81 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %89 = tt.broadcast %88 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
    %90 = arith.andi %89, %86 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xi1, #blocked1>
    %91 = triton_gpu.convert_layout %65 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %75, %91, %87 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    %92 = triton_gpu.convert_layout %66 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %80, %92, %90 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}


// -----

// CHECK-LABEL: @_matmul_layernorm_persistent_one_producer_one_consumer_one_epilog
// CHECK: %[[#TASKID:]] = triton_nvidia_gpu.get_async_task_id : i32
// CHECK: %c0_i32_0 = arith.constant 0 : i32
// CHECK: %[[#WG0:]] = arith.cmpi eq, %[[#TASKID]], %c0_i32_0 : i32
// CHECK: scf.if %[[#WG0]]
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.for
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: triton_nvidia_gpu.barrier_expect
// CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local
// CHECK: triton_nvidia_gpu.async_tma_copy_global_to_local
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[#WG1:]] = arith.cmpi eq, %[[#TASKID]], %c1_i32 : i32
// CHECK: scf.if %[[#WG1]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.producer_acquire
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.wait_barrier
// CHECK: triton_gpu.local_load
// CHECK: triton_gpu.local_load
// CHECK: triton_nvidia_gpu.warp_group_dot
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: triton_gpu.local_store
// CHECK: triton_nvidia_gpu.producer_commit
// CHECK: %c2_i32 = arith.constant 2 : i32
// CHECK: %[[#WG2:]] = arith.cmpi eq, %[[#TASKID]], %c2_i32 : i32
// CHECK: scf.if %[[#WG2]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: scf.for
// CHECK: scf.for
// CHECK: triton_gpu.local_load
// CHECK: triton_nvidia_gpu.consumer_wait
// CHECK: triton_nvidia_gpu.consumer_release
// CHECK: tt.experimental_descriptor_store

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_matmul_layernorm_persistent_one_producer_one_consumer_one_epilog(%arg0: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg1: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg2: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg4: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: f32) attributes {noinline = false} {
    %c63_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c132_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 132 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
    %c255_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 255 : i32
    %cst = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %cst_0 = arith.constant {async_task_id = dense<2> : vector<1xi32>} dense<1.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %0 = arith.addi %arg7, %c63_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.divsi %0, %c64_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.addi %arg5, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.divsi %2, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = arith.addi %arg6, %c255_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %5 = arith.divsi %4, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %6 = arith.muli %3, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %7 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %8 = arith.sitofp %arg6 {async_task_id = dense<2> : vector<1xi32>} : i32 to f32
    %9 = tt.splat %8 {async_task_id = dense<2> : vector<1xi32>} : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %10 = tt.splat %arg11 {async_task_id = dense<2> : vector<1xi32>} : f32 -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    scf.for %arg12 = %7 to %6 step %c132_i32  : i32 {
      %11 = arith.muli %arg12, %c128_i32 {async_task_id = dense<[0, 2]> : vector<2xi32>} : i32
      %true = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
      %false = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
      %12 = scf.for %arg13 = %c0_i32 to %1 step %c1_i32 iter_args(%arg14 = %cst) -> (tensor<128x256xf32, #mma>)  : i32 {
        %45 = arith.muli %arg13, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
        %46 = tt.experimental_descriptor_load %arg0[%11, %45] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<128x64xf16, #blocked>
        %47 = triton_gpu.local_alloc %46 {async_task_id = dense<1> : vector<1xi32>} : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
        %48 = tt.experimental_descriptor_load %arg1[%45, %c0_i32] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x256xf16, #blocked1>
        %49 = triton_gpu.local_alloc %48 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>
        %50 = triton_nvidia_gpu.warp_group_dot %47, %49, %arg14 {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %50 : tensor<128x256xf32, #mma>
      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
      %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %45 = arith.addf %arg13, %arg14 {async_task_id = dense<2> : vector<1xi32>} : f32
        tt.reduce.return %45 {async_task_id = dense<2> : vector<1xi32>} : f32
      }) {async_task_id = dense<2> : vector<1xi32>} : (tensor<128x256xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %14 = arith.divf %13, %9 {async_task_id = dense<2> : vector<1xi32>} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %15 = tt.expand_dims %14 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %16 = tt.broadcast %15 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma>
      %17 = arith.subf %12, %16 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma>
      %18 = arith.mulf %17, %17 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma>
      %19 = "tt.reduce"(%18) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %45 = arith.addf %arg13, %arg14 {async_task_id = dense<2> : vector<1xi32>} : f32
        tt.reduce.return %45 {async_task_id = dense<2> : vector<1xi32>} : f32
      }) {async_task_id = dense<2> : vector<1xi32>} : (tensor<128x256xf32, #mma>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %20 = arith.divf %19, %9 {async_task_id = dense<2> : vector<1xi32>} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %21 = arith.addf %20, %10 {async_task_id = dense<2> : vector<1xi32>} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %22 = math.sqrt %21 {async_task_id = dense<2> : vector<1xi32>} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %23 = arith.divf %cst_0, %22 {async_task_id = dense<2> : vector<1xi32>} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %24 = tt.experimental_descriptor_load %arg3[%c0_i32] {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256xf16, #blocked2>
      %25 = tt.experimental_descriptor_load %arg4[%c0_i32] {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<256xf16, #blocked2>
      %26 = tt.expand_dims %23 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %27 = tt.broadcast %26 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma>
      %28 = arith.mulf %17, %27 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma>
      %29 = triton_gpu.convert_layout %24 {async_task_id = dense<2> : vector<1xi32>} : tensor<256xf16, #blocked2> -> tensor<256xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %30 = tt.expand_dims %29 {async_task_id = dense<2> : vector<1xi32>, axis = 0 : i32} : tensor<256xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xf16, #blocked1>
      %31 = triton_gpu.convert_layout %30 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf16, #blocked1> -> tensor<1x256xf16, #blocked3>
      %32 = arith.extf %31 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf16, #blocked3> to tensor<1x256xf32, #blocked3>
      %33 = triton_gpu.convert_layout %32 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf32, #blocked3> -> tensor<1x256xf32, #mma>
      %34 = tt.broadcast %33 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
      %35 = arith.mulf %28, %34 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma>
      %36 = triton_gpu.convert_layout %25 {async_task_id = dense<2> : vector<1xi32>} : tensor<256xf16, #blocked2> -> tensor<256xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %37 = tt.expand_dims %36 {async_task_id = dense<2> : vector<1xi32>, axis = 0 : i32} : tensor<256xf16, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xf16, #blocked1>
      %38 = triton_gpu.convert_layout %37 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf16, #blocked1> -> tensor<1x256xf16, #blocked3>
      %39 = arith.extf %38 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf16, #blocked3> to tensor<1x256xf32, #blocked3>
      %40 = triton_gpu.convert_layout %39 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf32, #blocked3> -> tensor<1x256xf32, #mma>
      %41 = tt.broadcast %40 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
      %42 = arith.addf %35, %41 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma>
      %43 = arith.truncf %42 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %44 = triton_gpu.convert_layout %43 {async_task_id = dense<2> : vector<1xi32>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.experimental_descriptor_store %arg2[%11, %c0_i32], %44 {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<i8, 0>, tensor<128x256xf16, #blocked1>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    tt.return
  }
}

// -----

// Verify that we can reuse buffers between two for loops
// CHECK-LABEL: @_attn_bwd_ws
// CHECK-DAG: triton_gpu.local_alloc  {allocation.shareGroup = 0 : i32} : () -> !tt.memdesc<2x64x128xbf16
// CHECK-DAG: triton_gpu.local_alloc  {allocation.shareGroup = 1 : i32} : () -> !tt.memdesc<2x64x128xbf16
// CHECK-DAG: triton_gpu.local_alloc  {allocation.shareGroup = 0 : i32} : () -> !tt.memdesc<2x64x128xbf16
// CHECK-DAG: triton_gpu.local_alloc  {allocation.shareGroup = 1 : i32} : () -> !tt.memdesc<2x64x128xbf16

// CHECK: %[[TID:.*]] = triton_nvidia_gpu.get_async_task_id : i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[TWG0:.*]] = arith.cmpi eq, %[[TID]], %[[ZERO]] : i32
// CHECK: scf.if %[[TWG0]]
// CHECK: triton_nvidia_gpu.reg_dealloc 40
// CHECK: scf.if
// CHECK: scf.yield

// CHECK: %[[IF_IDX:.*]] = scf.if
// CHECK: arith.divui %c0{{.*}}
// CHECK: arith.subi %c0{{.*}}
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX:.*]] = arith.addi %c0
// CHECK: scf.yield {{.*}} %[[NEW_IDX]]
// CHECK: scf.yield {{.*}} %c0_

// CHECK: scf.if
// CHECK: arith.divui %[[IF_IDX]]
// CHECK: arith.subi %[[IF_IDX]]
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX2:.*]] = arith.addi %[[IF_IDX]]
// CHECK: scf.yield {{.*}} %[[NEW_IDX2]]
// CHECK: scf.yield {{.*}} %[[IF_IDX]]

// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[TWG1:.*]] = arith.cmpi eq, %[[TID]], %[[ONE]] : i32
// CHECK: scf.if %[[TWG1]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: scf.if
// CHECK: scf.yield

// CHECK: %[[IF_IDX_WG1:.*]] = scf.if
// CHECK: arith.divui %c0{{.*}}
// CHECK: arith.subi %c0{{.*}}
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX_WG1:.*]] = arith.addi %c0
// CHECK: scf.yield {{.*}} %[[NEW_IDX_WG1]]
// CHECK: scf.yield {{.*}} %c0_

// CHECK: scf.if
// CHECK: arith.divui %[[IF_IDX_WG1]]
// CHECK: arith.subi %[[IF_IDX_WG1]]
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX2_WG1:.*]] = arith.addi %[[IF_IDX_WG1]]
// CHECK: scf.yield {{.*}} %[[NEW_IDX2_WG1]]
// CHECK: scf.yield {{.*}} %[[IF_IDX_WG1]]

// CHECK: %[[TWO:.*]] = arith.constant 2 : i32
// CHECK: %[[TWG2:.*]] = arith.cmpi eq, %[[TID]], %[[TWO]] : i32
// CHECK: scf.if %[[TWG2]]
// CHECK: triton_nvidia_gpu.reg_alloc 232
// CHECK: scf.if
// CHECK: scf.yield

// CHECK: %[[IF_IDX_WG2:.*]] = scf.if
// CHECK: arith.divui %c0{{.*}}
// CHECK: arith.subi %c0{{.*}}
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX_WG2:.*]] = arith.addi %c0
// CHECK: scf.yield {{.*}} %[[NEW_IDX_WG2]]
// CHECK: scf.yield {{.*}} %c0_

// CHECK: scf.if
// CHECK: arith.divui %[[IF_IDX_WG2]]
// CHECK: arith.subi %[[IF_IDX_WG2]]
// CHECK: scf.for
// CHECK: scf.yield
// CHECK: arith.addi
// CHECK: %[[NEW_IDX2_WG2:.*]] = arith.addi %[[IF_IDX_WG2]]
// CHECK: scf.yield {{.*}} %[[NEW_IDX2_WG2]]
// CHECK: scf.yield {{.*}} %[[IF_IDX_WG2]]

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_bwd_ws(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg6: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg7: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg8: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg9: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg10: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg11: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg12: !tt.ptr<i8, 0> {tt.nv_tma_desc = 1 : i32}, %arg14: f32, %arg15: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg16: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg17: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg18: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg19: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg20: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}, %arg28: i32 {tt.divisibility = 16 : i32}, %arg29: i32, %arg30: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} false
    %cst = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %cst_0 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<128> : tensor<1x128xi32, #blocked>
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c64_i64 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i64
    %c63_i64 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i64
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c0_i64 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i64
    %c1_i64 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i64
    %cst_1 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} dense<0.000000e+00> : tensor<64x128xf32, #mma1>
    %cst_2 = arith.constant {async_task_id = dense<[1, 2]> : vector<2xi32>} dense<0.693147182> : tensor<64x128xf32, #mma1>
    %0 = tt.get_program_id z {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.divsi %0, %arg29 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.remsi %0, %arg29 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = tt.addptr %arg1, %1 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>, i32
    %5 = tt.load %4 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>
    %6 = tt.addptr %4, %c1_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>, i32
    %7 = tt.load %6 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>
    %8 = arith.subi %7, %5 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    %9 = tt.addptr %arg3, %1 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>, i32
    %10 = tt.load %9 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>
    %11 = tt.addptr %9, %c1_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>, i32
    %12 = tt.load %11 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>
    %13 = arith.subi %12, %10 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    %14 = arith.muli %3, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %15 = tt.make_range {async_task_id = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {async_task_id = dense<2> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {async_task_id = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %18 = tt.make_range {async_task_id = dense<2> : vector<1xi32>, end = 128 : i32, start = 64 : i32} : tensor<64xi32, #blocked1>
    %19 = arith.extsi %14 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
    %20 = arith.cmpi sle, %19, %13 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    %21 = arith.cmpi sle, %19, %8 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    %22 = arith.ori %20, %21 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i1
    %23:5 = scf.if %22 -> (!tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<f32>, !tt.ptr<f32>) {
      %27 = tt.addptr %arg16, %1 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>, i32
      %28 = tt.load %27 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<i64>
      %29 = arith.extsi %2 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %30 = arith.extsi %arg26 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %31 = arith.muli %29, %30 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %32 = arith.addi %31, %28 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %33 = arith.extsi %arg24 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %34 = arith.muli %29, %33 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %35 = arith.extsi %arg22 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %36 = arith.muli %5, %35 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %37 = arith.addi %34, %36 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %38 = arith.muli %2, %arg25 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %39 = arith.extsi %arg23 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %40 = arith.muli %10, %39 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %41 = arith.extsi %38 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
      %42 = arith.addi %41, %40 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %43 = tt.addptr %arg17, %37 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<bf16>, i64
      %44 = tt.addptr %arg18, %42 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<bf16>, i64
      %45 = tt.addptr %arg19, %42 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<bf16>, i64
      %46 = tt.addptr %arg20, %32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<f32>, i64
      %47 = tt.addptr %arg21, %32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : !tt.ptr<f32>, i64
      scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %43, %44, %45, %46, %47 : !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<f32>, !tt.ptr<f32>
    } else {
      scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %arg17, %arg18, %arg19, %arg20, %arg21 : !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<bf16>, !tt.ptr<f32>, !tt.ptr<f32>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    %24 = arith.extsi %14 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 to i64
    %25 = arith.cmpi slt, %24, %13 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    scf.if %25 {
      %27 = tt.splat %14 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %28 = tt.splat %14 {async_task_id = dense<2> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %29 = arith.addi %27, %15 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %30 = arith.addi %28, %16 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %31 = arith.extsi %14 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %32 = arith.addi %10, %31 {async_task_id = dense<0> : vector<1xi32>} : i64
      %33 = arith.trunci %32 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %34 = arith.addi %33, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %35 = arith.addi %33, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %36 = arith.muli %2, %arg25 {async_task_id = dense<0> : vector<1xi32>} : i32
      %37 = tt.experimental_descriptor_load %arg6[%33, %36] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %38 = tt.experimental_descriptor_load %arg6[%35, %36] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %39 = triton_gpu.local_alloc %37 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %40 = triton_gpu.local_alloc %38 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %41 = tt.experimental_descriptor_load %arg7[%33, %36] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %42 = tt.experimental_descriptor_load %arg7[%34, %36] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %43 = triton_gpu.local_alloc %41 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %44 = triton_gpu.local_alloc %42 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %45 = arith.addi %8, %c63_i64 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %46 = arith.divsi %45, %c64_i64 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %47 = arith.extsi %2 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %48 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %49 = arith.addi %5, %c64_i64 {async_task_id = dense<0> : vector<1xi32>} : i64
      %50 = arith.trunci %49 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %51 = arith.extsi %arg24 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %52 = arith.muli %47, %51 {async_task_id = dense<0> : vector<1xi32>} : i64
      %53 = arith.trunci %52 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %54 = tt.splat %8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i64 -> tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %55 = tt.splat %23#3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %56 = tt.splat %arg14 {async_task_id = dense<1> : vector<1xi32>} : f32 -> tensor<64x64xf32, #mma>
      %57 = tt.splat %arg14 {async_task_id = dense<2> : vector<1xi32>} : f32 -> tensor<64x64xf32, #mma>
      %58 = tt.splat %23#4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
      %59:3 = scf.for %arg31 = %c0_i64 to %46 step %c1_i64 iter_args(%arg32 = %c0_i32, %arg33 = %cst_1, %arg35 = %cst_1) -> (i32, tensor<64x128xf32, #mma1>, tensor<64x128xf32, #mma1>)  : i64 {
        %111 = tt.splat %arg32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %112 = arith.addi %111, %48 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %113 = tt.experimental_descriptor_load %arg5[%50, %53] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
        %114 = triton_gpu.local_alloc %113 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %115 = tt.trans %114 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %116 = arith.extsi %112 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>> to tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %117 = arith.cmpi slt, %116, %54 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64xi64, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %118 = tt.addptr %55, %112 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %119 = tt.load %118, %117 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %120 = triton_nvidia_gpu.warp_group_dot %39, %115, %cst, %false {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %121 = triton_nvidia_gpu.warp_group_dot %40, %115, %cst, %false {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %122 = arith.mulf %120, %56 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %123 = arith.mulf %121, %57 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %124 = tt.experimental_descriptor_load %arg8[%50, %53] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
        %125 = triton_gpu.local_alloc %124 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %126 = tt.trans %125 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %127 = triton_nvidia_gpu.warp_group_dot %43, %126, %cst, %false {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %128 = triton_nvidia_gpu.warp_group_dot %44, %126, %cst, %false {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %129 = tt.expand_dims %119 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<64xf32, #triton_gpu.slice<{dim = 0, parent = #mma}>> -> tensor<1x64xf32, #mma>
        %130 = tt.broadcast %129 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x64xf32, #mma> -> tensor<64x64xf32, #mma>
        %131 = tt.broadcast %129 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x64xf32, #mma> -> tensor<64x64xf32, #mma>
        %132 = arith.subf %122, %130 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %133 = arith.subf %123, %131 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %134 = math.exp2 %132 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %135 = math.exp2 %133 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %136 = arith.truncf %134 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
        %137 = arith.truncf %135 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
        %138 = tt.addptr %58, %112 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>, tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %139 = tt.load %138, %117 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<64x!tt.ptr<f32>, #triton_gpu.slice<{dim = 0, parent = #mma}>>
        %140 = triton_gpu.convert_layout %136 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %141 = triton_gpu.convert_layout %137 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %142 = triton_nvidia_gpu.warp_group_dot %140, %125, %arg33 {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma1>
        %143 = triton_nvidia_gpu.warp_group_dot %141, %125, %arg35 {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma1>
        %157 = arith.addi %arg32, %c64_i32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32
        scf.yield {async_task_id = dense<[1, 2]> : vector<2xi32>} %157, %142, %143 : i32, tensor<64x128xf32, #mma1>, tensor<64x128xf32, #mma1>
      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, tt.num_stages = 2 : i32}
      %60 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %61 = tt.expand_dims %60 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %62 = arith.cmpi slt, %61, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked>
      %63 = tt.expand_dims %29 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %64 = tt.expand_dims %30 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %65 = arith.extsi %63 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
      %66 = arith.extsi %64 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
      %67 = tt.splat %13 {async_task_id = dense<1> : vector<1xi32>} : i64 -> tensor<64x1xi64, #blocked>
      %68 = tt.splat %13 {async_task_id = dense<2> : vector<1xi32>} : i64 -> tensor<64x1xi64, #blocked>
      %69 = arith.cmpi slt, %65, %67 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi64, #blocked>
      %70 = arith.cmpi slt, %66, %68 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi64, #blocked>
      %71 = tt.broadcast %62 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %72 = tt.broadcast %62 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %73 = tt.broadcast %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %74 = tt.broadcast %70 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %75 = arith.andi %71, %73 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xi1, #blocked>
      %76 = arith.andi %72, %74 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xi1, #blocked>
      %77 = tt.splat %arg23 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
      %78 = tt.splat %arg23 {async_task_id = dense<2> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
      %79 = arith.muli %63, %77 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked>
      %80 = arith.muli %64, %78 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked>
      %81 = tt.splat %23#2 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
      %82 = tt.splat %23#2 {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
      %83 = tt.addptr %81, %79 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
      %84 = tt.addptr %82, %80 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
      %85 = tt.broadcast %83 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %86 = tt.broadcast %84 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %87 = tt.broadcast %61 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %88 = tt.broadcast %61 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %89 = tt.addptr %85, %87 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %90 = tt.addptr %86, %88 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %91 = arith.truncf %59#1 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma1> to tensor<64x128xbf16, #mma1>
      %92 = arith.truncf %59#2 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma1> to tensor<64x128xbf16, #mma1>
      %93 = triton_gpu.convert_layout %91 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xbf16, #mma1> -> tensor<64x128xbf16, #blocked>
      %94 = triton_gpu.convert_layout %92 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xbf16, #mma1> -> tensor<64x128xbf16, #blocked>
      tt.store %89, %93, %75 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
      tt.store %90, %94, %76 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    %26 = arith.cmpi slt, %24, %8 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
    scf.if %26 {
      %27 = tt.splat %14 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %28 = tt.splat %14 {async_task_id = dense<2> : vector<1xi32>} : i32 -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %29 = tt.splat %14 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<64xi32, #blocked1>
      %30 = tt.splat %14 {async_task_id = dense<2> : vector<1xi32>} : i32 -> tensor<64xi32, #blocked1>
      %31 = arith.addi %27, %15 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %32 = arith.addi %28, %16 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %33 = arith.addi %29, %17 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #blocked1>
      %34 = arith.addi %30, %18 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #blocked1>
      %35 = arith.extsi %2 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %36 = arith.extsi %14 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %37 = arith.addi %5, %36 {async_task_id = dense<0> : vector<1xi32>} : i64
      %38 = arith.trunci %37 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %39 = arith.addi %38, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %40 = arith.addi %38, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %41 = arith.extsi %arg24 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %42 = arith.muli %35, %41 {async_task_id = dense<0> : vector<1xi32>} : i64
      %43 = arith.trunci %42 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %44 = tt.experimental_descriptor_load %arg9[%38, %43] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %45 = tt.experimental_descriptor_load %arg9[%40, %43] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %46 = triton_gpu.local_alloc %44 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %47 = triton_gpu.local_alloc %45 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %48 = arith.extsi %arg28 {async_task_id = dense<0> : vector<1xi32>} : i32 to i64
      %49 = arith.muli %35, %48 {async_task_id = dense<0> : vector<1xi32>} : i64
      %50 = arith.trunci %49 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %51 = tt.experimental_descriptor_load %arg12[%38, %50] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %52 = tt.experimental_descriptor_load %arg12[%39, %50] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
      %53 = triton_gpu.local_alloc %51 {async_task_id = dense<1> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %54 = triton_gpu.local_alloc %52 {async_task_id = dense<2> : vector<1xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
      %55 = arith.extsi %33 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi32, #blocked1> to tensor<64xi64, #blocked1>
      %56 = arith.extsi %34 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi32, #blocked1> to tensor<64xi64, #blocked1>
      %57 = tt.splat %8 {async_task_id = dense<1> : vector<1xi32>} : i64 -> tensor<64xi64, #blocked1>
      %58 = tt.splat %8 {async_task_id = dense<2> : vector<1xi32>} : i64 -> tensor<64xi64, #blocked1>
      %59 = arith.cmpi slt, %55, %57 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xi64, #blocked1>
      %60 = arith.cmpi slt, %56, %58 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xi64, #blocked1>
      %61 = tt.splat %23#3 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %62 = tt.splat %23#3 {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %63 = tt.addptr %61, %33 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %64 = tt.addptr %62, %34 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %65 = tt.load %63, %59 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>
      %66 = tt.load %64, %60 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>
      %67 = triton_gpu.convert_layout %65 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xf32, #blocked1> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %68 = triton_gpu.convert_layout %66 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xf32, #blocked1> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %69 = tt.expand_dims %67 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xf32, #blocked3>
      %70 = tt.expand_dims %68 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xf32, #blocked3>
      %71 = arith.addi %13, %c63_i64 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %72 = arith.divsi %71, %c64_i64 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i64
      %73 = tt.splat %23#4 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %74 = tt.splat %23#4 {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %75 = tt.addptr %73, %33 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %76 = tt.addptr %74, %34 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      %77 = tt.load %75, %59 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>
      %78 = tt.load %76, %60 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x!tt.ptr<f32>, #blocked1>
      %79 = arith.trunci %10 {async_task_id = dense<0> : vector<1xi32>} : i64 to i32
      %80 = arith.muli %2, %arg25 {async_task_id = dense<0> : vector<1xi32>} : i32
      %81 = tt.splat %arg14 {async_task_id = dense<1> : vector<1xi32>} : f32 -> tensor<64x64xf32, #mma>
      %82 = tt.splat %arg14 {async_task_id = dense<2> : vector<1xi32>} : f32 -> tensor<64x64xf32, #mma>
      %83 = tt.broadcast %69 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %84 = tt.broadcast %70 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %85 = triton_gpu.convert_layout %83 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #mma>
      %86 = triton_gpu.convert_layout %84 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #mma>
      %87 = triton_gpu.convert_layout %77 {async_task_id = dense<1> : vector<1xi32>} : tensor<64xf32, #blocked1> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %88 = triton_gpu.convert_layout %78 {async_task_id = dense<2> : vector<1xi32>} : tensor<64xf32, #blocked1> -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
      %89 = tt.expand_dims %87 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xf32, #blocked3>
      %90 = tt.expand_dims %88 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xf32, #blocked3>
      %91 = tt.broadcast %89 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %92 = tt.broadcast %90 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %93 = triton_gpu.convert_layout %91 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #mma>
      %94 = triton_gpu.convert_layout %92 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #mma>
      %95 = tt.splat %arg14 {async_task_id = dense<1> : vector<1xi32>} : f32 -> tensor<64x128xf32, #mma1>
      %96 = tt.splat %arg14 {async_task_id = dense<2> : vector<1xi32>} : f32 -> tensor<64x128xf32, #mma1>
      %97:2 = scf.for %arg31 = %c0_i64 to %72 step %c1_i64 iter_args(%arg32 = %cst_1, %arg33 = %cst_1) -> (tensor<64x128xf32, #mma1>, tensor<64x128xf32, #mma1>)  : i64 {
        %135 = tt.experimental_descriptor_load %arg10[%79, %80] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
        %136 = triton_gpu.local_alloc %135 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %137 = tt.trans %136 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %138 = tt.experimental_descriptor_load %arg11[%79, %80] {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<i8, 0> -> tensor<64x128xbf16, #blocked2>
        %139 = triton_gpu.local_alloc %138 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x128xbf16, #blocked2>) -> !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory>
        %140 = tt.trans %139 {async_task_id = dense<[1, 2]> : vector<2xi32>, order = array<i32: 1, 0>} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory>
        %141 = triton_nvidia_gpu.warp_group_dot %46, %137, %cst, %false {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %142 = triton_nvidia_gpu.warp_group_dot %47, %137, %cst, %false {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %143 = arith.mulf %141, %81 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %144 = arith.mulf %142, %82 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %145 = triton_nvidia_gpu.warp_group_dot %53, %140, %cst, %false {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %146 = triton_nvidia_gpu.warp_group_dot %54, %140, %cst, %false {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<128x64xbf16, #shared1, #triton_gpu.shared_memory> -> tensor<64x64xf32, #mma>
        %147 = arith.subf %143, %85 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %148 = arith.subf %144, %86 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %149 = math.exp2 %147 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %150 = math.exp2 %148 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %151 = arith.subf %145, %93 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %152 = arith.subf %146, %94 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %153 = arith.mulf %149, %151 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %154 = arith.mulf %150, %152 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma>
        %155 = arith.truncf %153 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
        %156 = arith.truncf %154 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xf32, #mma> to tensor<64x64xbf16, #mma>
        %157 = triton_gpu.convert_layout %155 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %158 = triton_gpu.convert_layout %156 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x64xbf16, #mma> -> tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
        %159 = triton_nvidia_gpu.warp_group_dot %157, %136, %cst_1, %false {async_task_id = dense<1> : vector<1xi32>, inputPrecision = 0 : i32} : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma1>
        %160 = triton_nvidia_gpu.warp_group_dot %158, %136, %cst_1, %false {async_task_id = dense<2> : vector<1xi32>, inputPrecision = 0 : i32} : tensor<64x64xbf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * !tt.memdesc<64x128xbf16, #shared, #triton_gpu.shared_memory> -> tensor<64x128xf32, #mma1>
        %161 = arith.mulf %159, %95 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma1>
        %162 = arith.mulf %160, %96 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma1>
        %163 = arith.addf %arg32, %161 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma1>
        %164 = arith.addf %arg33, %162 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma1>
        scf.yield {async_task_id = dense<[1, 2]> : vector<2xi32>} %163, %164 : tensor<64x128xf32, #mma1>, tensor<64x128xf32, #mma1>
      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, tt.num_stages = 2 : i32}
      %98 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %99 = tt.expand_dims %98 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %100 = arith.cmpi slt, %99, %cst_0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x128xi32, #blocked>
      %101 = tt.expand_dims %31 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %102 = tt.expand_dims %32 {async_task_id = dense<2> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %103 = arith.extsi %101 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
      %104 = arith.extsi %102 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
      %105 = tt.splat %8 {async_task_id = dense<1> : vector<1xi32>} : i64 -> tensor<64x1xi64, #blocked>
      %106 = tt.splat %8 {async_task_id = dense<2> : vector<1xi32>} : i64 -> tensor<64x1xi64, #blocked>
      %107 = arith.cmpi slt, %103, %105 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi64, #blocked>
      %108 = arith.cmpi slt, %104, %106 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi64, #blocked>
      %109 = tt.broadcast %100 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %110 = tt.broadcast %100 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x128xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %111 = tt.broadcast %107 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %112 = tt.broadcast %108 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
      %113 = arith.andi %109, %111 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xi1, #blocked>
      %114 = arith.andi %110, %112 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xi1, #blocked>
      %115 = tt.splat %arg22 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
      %116 = tt.splat %arg22 {async_task_id = dense<2> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked>
      %117 = arith.muli %101, %115 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked>
      %118 = arith.muli %102, %116 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1xi32, #blocked>
      %119 = tt.splat %23#0 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
      %120 = tt.splat %23#0 {async_task_id = dense<2> : vector<1xi32>} : !tt.ptr<bf16> -> tensor<64x1x!tt.ptr<bf16>, #blocked>
      %121 = tt.addptr %119, %117 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
      %122 = tt.addptr %120, %118 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked>, tensor<64x1xi32, #blocked>
      %123 = tt.broadcast %121 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %124 = tt.broadcast %122 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x1x!tt.ptr<bf16>, #blocked> -> tensor<64x128x!tt.ptr<bf16>, #blocked>
      %125 = tt.broadcast %99 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %126 = tt.broadcast %99 {async_task_id = dense<2> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
      %127 = tt.addptr %123, %125 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %128 = tt.addptr %124, %126 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>, tensor<64x128xi32, #blocked>
      %129 = arith.mulf %97#0, %cst_2 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma1>
      %130 = arith.mulf %97#1, %cst_2 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma1>
      %131 = arith.truncf %129 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xf32, #mma1> to tensor<64x128xbf16, #mma1>
      %132 = arith.truncf %130 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xf32, #mma1> to tensor<64x128xbf16, #mma1>
      %133 = triton_gpu.convert_layout %131 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128xbf16, #mma1> -> tensor<64x128xbf16, #blocked>
      %134 = triton_gpu.convert_layout %132 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128xbf16, #mma1> -> tensor<64x128xbf16, #blocked>
      tt.store %127, %133, %113 {async_task_id = dense<1> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
      tt.store %128, %134, %114 {async_task_id = dense<2> : vector<1xi32>} : tensor<64x128x!tt.ptr<bf16>, #blocked>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    tt.return
  }
}
