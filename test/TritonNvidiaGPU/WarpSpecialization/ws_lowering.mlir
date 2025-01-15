// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-lowering=num-consumer-groups=1 | FileCheck %s

// CHECK:  %[[#PBARRIER:]] = triton_gpu.local_alloc  : () -> !tt.memdesc<1xi64
// CHECK:  %[[#CBARRIER:]] = triton_gpu.local_alloc  : () -> !tt.memdesc<1xi64
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#PBARRIER]][%c0_i32]
// CHECK:  triton_nvidia_gpu.init_barrier %[[#]], 128
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#CBARRIER]][%c0_i32]
// CHECK:  triton_nvidia_gpu.init_barrier %[[#]], 1
// CHECK:  scf.for
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#CBARRIER]]
// CHECK:  triton_nvidia_gpu.wait_barrier %[[#]]
// CHECK:  triton_gpu.async_copy_global_to_local
// CHECK:  triton_gpu.async_copy_global_to_local
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#PBARRIER]]
// CHECK:  triton_nvidia_gpu.mbarrier_arrive %[[#]]
// CHECK:  scf.for
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#PBARRIER]]
// CHECK:  triton_nvidia_gpu.wait_barrier %[[#]]
// CHECK:  triton_gpu.local_load
// CHECK:  triton_gpu.local_load
// CHECK:  tt.dot
// CHECK:  %[[#]] = triton_gpu.memdesc_subview %[[#CBARRIER]]
// CHECK:  triton_nvidia_gpu.mbarrier_arrive %[[#]]



#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x128x256xf16, #shared, #triton_gpu.shared_memory, mutable>
    %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<1x256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
    %2 = triton_nvidia_gpu.create_token {num = 1 : i32} : tensor<1x!triton_nvidia_gpu.token>
    %3 = triton_nvidia_gpu.get_async_task_id : i32
    %c0_i32 = arith.constant 0 : i32
    %4 = arith.cmpi eq, %3, %c0_i32 : i32
    scf.if %4 {
      %c255_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 255 : i32
      %c127_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 127 : i32
      %c1_i32_0 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
      %c0_i32_1 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %cst = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<256x128xf16, #blocked>
      %cst_2 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf16, #blocked1>
      %c8_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 8 : i32
      %c128_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 128 : i32
      %c256_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 256 : i32
      %cst_3 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<256> : tensor<128x256xi32, #blocked1>
      %6 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %7 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %8 = arith.divsi %7, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %10 = arith.divsi %9, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.muli %10, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = arith.divsi %6, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %13 = arith.muli %12, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %14 = arith.subi %8, %13 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %15 = arith.minsi %14, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %16 = arith.remsi %6, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %17 = arith.remsi %16, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %18 = arith.addi %13, %17 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %19 = arith.divsi %16, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %20 = arith.muli %18, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %21 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %22 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %23 = tt.splat %20 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %24 = arith.addi %23, %21 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %25 = tt.splat %arg3 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %26 = arith.remsi %24, %25 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %27 = arith.muli %19, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %28 = tt.splat %27 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %22 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %31 = arith.remsi %29, %30 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %32 = tt.expand_dims %26 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %33 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked1>
      %34 = arith.muli %32, %33 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
      %35 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %36 = tt.expand_dims %35 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %37 = tt.broadcast %34 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %38 = tt.broadcast %36 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %39 = arith.addi %37, %38 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256xi32, #blocked1>
      %40 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      %41 = tt.addptr %40, %39 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
      %42 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %43 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %44 = tt.expand_dims %42 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
      %45 = tt.expand_dims %43 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
      %46 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked>
      %47 = arith.muli %44, %46 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked>
      %48 = tt.expand_dims %31 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %49 = tt.broadcast %47 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked> -> tensor<256x128xi32, #blocked>
      %50 = tt.broadcast %48 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<256x128xi32, #blocked>
      %51 = arith.addi %49, %50 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128xi32, #blocked>
      %52 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<256x128x!tt.ptr<f16>, #blocked>
      %53 = tt.addptr %52, %51 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked>, tensor<256x128xi32, #blocked>
      %54 = arith.addi %arg5, %c255_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %55 = arith.divsi %54, %c256_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %56 = arith.muli %arg7, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
      %57 = tt.splat %56 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x128xi32, #blocked>
      %c1_i32_4 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
      %c0_i32_5 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %false = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} false
      %58:4 = scf.for %arg9 = %c0_i32_1 to %55 step %c1_i32_0 iter_args(%arg10 = %41, %arg11 = %53, %arg12 = %false, %arg13 = %c0_i32_5) -> (tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<256x128x!tt.ptr<f16>, #blocked>, i1, i32)  : i32 {
        %59 = arith.muli %arg9, %c256_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
        %60 = arith.subi %arg5, %59 {async_task_id = dense<0> : vector<1xi32>} : i32
        %61 = tt.splat %60 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x256xi32, #blocked1>
        %62 = arith.cmpi slt, %36, %61 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked1>
        %63 = tt.broadcast %62 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
        triton_nvidia_gpu.producer_acquire %2, %arg13, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<1x!triton_nvidia_gpu.token>, i32, i1
        %c0_i32_6 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
        %c1_i32_7 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 1 : i32
        %64 = triton_gpu.memdesc_subview %0[%arg13, %c0_i32_6, %c0_i32_6] {async_task_id = dense<0> : vector<1xi32>} : !tt.memdesc<1x128x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x256xf16, #shared, #triton_gpu.shared_memory, mutable>
        %65 = triton_gpu.async_copy_global_to_local %arg10, %64 mask %63 other %cst_2 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1> -> <128x256xf16, #shared, #triton_gpu.shared_memory, mutable>
        %66 = tt.splat %60 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256x1xi32, #blocked>
        %67 = arith.cmpi slt, %45, %66 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi32, #blocked>
        %68 = tt.broadcast %67 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x1xi1, #blocked> -> tensor<256x128xi1, #blocked>
        %c0_i32_8 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
        %c1_i32_9 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 1 : i32
        %69 = triton_gpu.memdesc_subview %1[%arg13, %c0_i32_8, %c0_i32_8] {async_task_id = dense<0> : vector<1xi32>} : !tt.memdesc<1x256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
        %70 = triton_gpu.async_copy_global_to_local %arg11, %69 mask %68 other %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked> -> <256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
        triton_nvidia_gpu.producer_commit %2, %arg13 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x!triton_nvidia_gpu.token>, i32
        %71 = tt.addptr %arg10, %cst_3 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
        %72 = tt.addptr %arg11, %57 {async_task_id = dense<0> : vector<1xi32>} : tensor<256x128x!tt.ptr<f16>, #blocked>, tensor<256x128xi32, #blocked>
        %c1_i32_10 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 1 : i32
        %c0_i32_11 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
        %true = arith.constant {async_task_id = dense<0> : vector<1xi32>} true
        %73 = arith.addi %arg13, %c1_i32_10 {async_task_id = dense<0> : vector<1xi32>} : i32
        %74 = arith.cmpi uge, %73, %c1_i32_4 {async_task_id = dense<0> : vector<1xi32>} : i32
        %75 = arith.cmpi ult, %73, %c1_i32_4 {async_task_id = dense<0> : vector<1xi32>} : i32
        %76 = arith.subi %73, %c1_i32_4 {async_task_id = dense<0> : vector<1xi32>} : i32
        %77 = arith.select %74, %76, %73 {async_task_id = dense<0> : vector<1xi32>} : i32
        %78 = arith.xori %arg12, %true {async_task_id = dense<0> : vector<1xi32>} : i1
        %79 = arith.andi %74, %78 {async_task_id = dense<0> : vector<1xi32>} : i1
        %80 = arith.andi %75, %arg12 {async_task_id = dense<0> : vector<1xi32>} : i1
        %81 = arith.ori %79, %80 {async_task_id = dense<0> : vector<1xi32>} : i1
        scf.yield {async_task_id = dense<0> : vector<1xi32>} %71, %72, %81, %77 : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<256x128x!tt.ptr<f16>, #blocked>, i1, i32
      } {async_task_id = dense<0> : vector<1xi32>}
    } {async_task_id = dense<0> : vector<1xi32>}
    %c1_i32 = arith.constant 1 : i32
    %5 = arith.cmpi eq, %3, %c1_i32 : i32
    scf.if %5 {
      %cst = arith.constant {async_task_id = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #blocked2>
      %c255_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 255 : i32
      %c127_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 127 : i32
      %c1_i32_0 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
      %c0_i32_1 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %cst_2 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<256x128xf16, #blocked>
      %cst_3 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<0.000000e+00> : tensor<128x256xf16, #blocked1>
      %c8_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 8 : i32
      %c128_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 128 : i32
      %c256_i32 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 256 : i32
      %cst_4 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} dense<256> : tensor<128x256xi32, #blocked1>
      %6 = tt.get_program_id x {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %7 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %8 = arith.divsi %7, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = arith.addi %arg4, %c127_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %10 = arith.divsi %9, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.muli %10, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = arith.divsi %6, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %13 = arith.muli %12, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %14 = arith.subi %8, %13 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %15 = arith.minsi %14, %c8_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %16 = arith.remsi %6, %11 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %17 = arith.remsi %16, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %18 = arith.addi %13, %17 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %19 = arith.divsi %16, %15 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %20 = arith.muli %18, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %21 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %22 = tt.make_range {async_task_id = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %23 = tt.make_range {async_task_id = dense<[0, 1]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %24 = tt.splat %20 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %25 = tt.splat %20 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %26 = arith.addi %24, %21 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %27 = arith.addi %25, %22 {async_task_id = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %28 = tt.splat %arg3 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %29 = arith.remsi %26, %28 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %30 = arith.muli %19, %c128_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %31 = tt.splat %30 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %32 = arith.addi %31, %23 {async_task_id = dense<[0, 1]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %33 = arith.addi %arg5, %c255_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %34 = arith.divsi %33, %c256_i32 {async_task_id = dense<[0, 1]> : vector<2xi32>} : i32
      %c1_i32_5 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 1 : i32
      %c0_i32_6 = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %false = arith.constant {async_task_id = dense<[0, 1]> : vector<2xi32>} false
      %35:3 = scf.for %arg9 = %c0_i32_1 to %34 step %c1_i32_0 iter_args(%arg10 = %cst, %arg11 = %false, %arg12 = %c0_i32_6) -> (tensor<128x128xf32, #blocked2>, i1, i32)  : i32 {
        %c0_i32_7 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 0 : i32
        %c1_i32_8 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 1 : i32
        %c0_i32_9 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 0 : i32
        %c1_i32_10 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 1 : i32
        triton_nvidia_gpu.consumer_wait %2, %arg12, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!triton_nvidia_gpu.token>, i32, i1
        %54 = triton_gpu.memdesc_subview %0[%arg12, %c0_i32_7, %c0_i32_7] {async_task_id = dense<1> : vector<1xi32>} : !tt.memdesc<1x128x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<128x256xf16, #shared, #triton_gpu.shared_memory, mutable>
        %55 = triton_gpu.local_load %54 {async_task_id = dense<1> : vector<1xi32>} : !tt.memdesc<128x256xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<128x256xf16, #blocked1>
        %56 = triton_gpu.convert_layout %55 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #blocked1> -> tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>
        %57 = triton_gpu.memdesc_subview %1[%arg12, %c0_i32_9, %c0_i32_9] {async_task_id = dense<1> : vector<1xi32>} : !tt.memdesc<1x256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable>
        %58 = triton_gpu.local_load %57 {async_task_id = dense<1> : vector<1xi32>} : !tt.memdesc<256x128xf16, #shared, #triton_gpu.shared_memory, mutable> -> tensor<256x128xf16, #blocked>
        %59 = triton_gpu.convert_layout %58 {async_task_id = dense<1> : vector<1xi32>} : tensor<256x128xf16, #blocked> -> tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>
        %60 = tt.dot %56, %59, %arg10, inputPrecision = tf32 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x256xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<256x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x128xf32, #blocked2>
        triton_nvidia_gpu.consumer_release %2, %arg12 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x!triton_nvidia_gpu.token>, i32
        %c1_i32_11 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 1 : i32
        %c0_i32_12 = arith.constant {async_task_id = dense<1> : vector<1xi32>} 0 : i32
        %true = arith.constant {async_task_id = dense<1> : vector<1xi32>} true
        %61 = arith.addi %arg12, %c1_i32_11 {async_task_id = dense<1> : vector<1xi32>} : i32
        %62 = arith.cmpi uge, %61, %c1_i32_5 {async_task_id = dense<1> : vector<1xi32>} : i32
        %63 = arith.cmpi ult, %61, %c1_i32_5 {async_task_id = dense<1> : vector<1xi32>} : i32
        %64 = arith.subi %61, %c1_i32_5 {async_task_id = dense<1> : vector<1xi32>} : i32
        %65 = arith.select %62, %64, %61 {async_task_id = dense<1> : vector<1xi32>} : i32
        %66 = arith.xori %arg11, %true {async_task_id = dense<1> : vector<1xi32>} : i1
        %67 = arith.andi %62, %66 {async_task_id = dense<1> : vector<1xi32>} : i1
        %68 = arith.andi %63, %arg11 {async_task_id = dense<1> : vector<1xi32>} : i1
        %69 = arith.ori %67, %68 {async_task_id = dense<1> : vector<1xi32>} : i1
        scf.yield {async_task_id = dense<1> : vector<1xi32>} %60, %69, %65 : tensor<128x128xf32, #blocked2>, i1, i32
      } {async_task_id = dense<1> : vector<1xi32>}
      %36 = arith.truncf %35#0 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf32, #blocked2> to tensor<128x128xf16, #blocked2>
      %37 = tt.expand_dims %27 {async_task_id = dense<1> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %38 = tt.splat %arg8 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked>
      %39 = arith.muli %38, %37 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked>
      %40 = tt.splat %arg2 {async_task_id = dense<1> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
      %41 = tt.addptr %40, %39 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
      %42 = tt.expand_dims %32 {async_task_id = dense<1> : vector<1xi32>, axis = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %43 = tt.broadcast %41 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
      %44 = tt.broadcast %42 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %45 = tt.addptr %43, %44 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
      %46 = tt.splat %arg3 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked>
      %47 = arith.cmpi slt, %37, %46 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked>
      %48 = tt.splat %arg4 {async_task_id = dense<1> : vector<1xi32>} : i32 -> tensor<1x128xi32, #blocked>
      %49 = arith.cmpi slt, %42, %48 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi32, #blocked>
      %50 = tt.broadcast %47 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %51 = tt.broadcast %49 {async_task_id = dense<1> : vector<1xi32>} : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
      %52 = arith.andi %50, %51 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked>
      %53 = triton_gpu.convert_layout %36 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128xf16, #blocked2> -> tensor<128x128xf16, #blocked>
      tt.store %45, %53, %52 {async_task_id = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16>, #blocked>
    } {async_task_id = dense<1> : vector<1xi32>}
    tt.return
  }
}
