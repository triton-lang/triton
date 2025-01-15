// RUN: triton-opt %s -split-input-file --tritongpu-warp-spec-data-partition=num-consumer-groups=2 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_ws_cooperative_kernel
// CHECK: %[[#GA1:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
// CHECK: %[[#GA2:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
// CHECK: %[[#LA1:]] = triton_gpu.local_alloc %[[#GA1]]
// CHECK: %[[#LA2:]] = triton_gpu.local_alloc %[[#GA2]]
// CHECK: %[[#GB:]] = tt.load {{.*}} : tensor<64x256x!tt.ptr<f16>
// CHECK: %[[#LB:]] = triton_gpu.local_alloc %[[#GB]]
// CHECK: %[[#C1:]] = triton_nvidia_gpu.warp_group_dot %[[#LA1]], %[[#LB]], {{.*}} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x256xf32, #mma>
// CHECK: %[[#C2:]] = triton_nvidia_gpu.warp_group_dot %[[#LA2]], %[[#LB]], {{.*}} : !tt.memdesc<64x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<64x256xf32, #mma>
// CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
// CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>



#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "cuda:90", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_ws_cooperative_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<64> : tensor<128x64xi32, #blocked>
    %c0_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 1 : i32
    %c255_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 255 : i32
    %c63_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 63 : i32
    %c64_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 64 : i32
    %c256_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 256 : i32
    %c128_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 8 : i32
    %c127_i32 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} 127 : i32
    %cst_0 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_1 = arith.constant {async_task_id = dense<0> : vector<1xi32>} dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    %cst_2 = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = arith.addi %arg3, %c127_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %1 = arith.divsi %0, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %2 = arith.addi %arg4, %c255_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %3 = arith.divsi %2, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %4 = arith.muli %1, %3 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %5 = tt.get_program_id x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %6 = tt.get_num_programs x {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %7 = arith.muli %3, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %8 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.make_range {async_task_id = dense<[1, 2]> : vector<2xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %10 = tt.splat %arg3 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.make_range {async_task_id = dense<[0, 1, 2]> : vector<3xi32>, end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %12 = tt.splat %arg4 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %13 = tt.splat %arg6 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128x1xi32, #blocked>
    %14 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.expand_dims %14 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %16 = tt.broadcast %15 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %17 = tt.splat %arg0 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %18 = tt.make_range {async_task_id = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.expand_dims %18 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %20 = tt.splat %arg7 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
    %21 = arith.muli %19, %20 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
    %22 = tt.broadcast %21 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %23 = tt.splat %arg1 {async_task_id = dense<0> : vector<1xi32>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %24 = arith.addi %arg5, %c63_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %25 = arith.divsi %24, %c64_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
    %26 = tt.expand_dims %14 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %27 = tt.expand_dims %18 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %28 = arith.muli %arg7, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
    %29 = tt.splat %28 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x256xi32, #blocked1>
    %30 = tt.splat %arg8 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %31 = tt.splat %arg2 {async_task_id = dense<[1, 2]> : vector<2xi32>} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %32 = tt.splat %arg3 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128x1xi32, #blocked1>
    %33 = tt.splat %arg4 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<1x256xi32, #blocked1>
    scf.for %arg9 = %5 to %4 step %6  : i32 {
      %34 = arith.divsi %arg9, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %35 = arith.muli %34, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %36 = arith.subi %1, %35 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %37 = arith.minsi %36, %c8_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %38 = arith.remsi %arg9, %7 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %39 = arith.remsi %38, %37 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %40 = arith.addi %35, %39 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %41 = arith.divsi %38, %37 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %42 = arith.muli %40, %c128_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %43 = tt.splat %42 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %44 = tt.splat %42 {async_task_id = dense<[1, 2]> : vector<2xi32>} : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %45 = arith.addi %43, %8 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %46 = arith.addi %44, %9 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %47 = arith.remsi %45, %10 {async_task_id = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %48 = arith.muli %41, %c256_i32 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32
      %49 = tt.splat %48 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %50 = arith.addi %49, %11 {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %51 = arith.remsi %50, %12 {async_task_id = dense<0> : vector<1xi32>} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %52 = tt.expand_dims %47 {async_task_id = dense<0> : vector<1xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %53 = arith.muli %52, %13 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked>
      %54 = tt.broadcast %53 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked> -> tensor<128x64xi32, #blocked>
      %55 = arith.addi %54, %16 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64xi32, #blocked>
      %56 = tt.addptr %17, %55 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
      %57 = tt.expand_dims %51 {async_task_id = dense<0> : vector<1xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %58 = tt.broadcast %57 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
      %59 = arith.addi %22, %58 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256xi32, #blocked1>
      %60 = tt.addptr %23, %59 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %true = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} true
      %false = arith.constant {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} false
      %61:3 = scf.for %arg10 = %c0_i32 to %25 step %c1_i32 iter_args(%arg11 = %cst_2, %arg12 = %56, %arg13 = %60) -> (tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>)  : i32 {
        %76 = arith.muli %arg10, %c64_i32 {async_task_id = dense<0> : vector<1xi32>} : i32
        %77 = arith.subi %arg5, %76 {async_task_id = dense<0> : vector<1xi32>} : i32
        %78 = tt.splat %77 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<1x64xi32, #blocked>
        %79 = arith.cmpi slt, %26, %78 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked>
        %80 = tt.broadcast %79 {async_task_id = dense<0> : vector<1xi32>} : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
        %81 = tt.load %arg12, %80, %cst_0 {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>
        %82 = triton_gpu.local_alloc %81 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<128x64xf16, #blocked>) -> !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory>
        %83 = tt.splat %77 {async_task_id = dense<0> : vector<1xi32>} : i32 -> tensor<64x1xi32, #blocked1>
        %84 = arith.cmpi slt, %27, %83 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
        %85 = tt.broadcast %84 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
        %86 = tt.load %arg13, %85, %cst_1 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        %87 = triton_gpu.local_alloc %86 {async_task_id = dense<[1, 2]> : vector<2xi32>} : (tensor<64x256xf16, #blocked1>) -> !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory>
        %88 = triton_nvidia_gpu.warp_group_dot %82, %87, %arg11 {async_task_id = dense<[1, 2]> : vector<2xi32>, inputPrecision = 0 : i32} : !tt.memdesc<128x64xf16, #shared, #triton_gpu.shared_memory> * !tt.memdesc<64x256xf16, #shared, #triton_gpu.shared_memory> -> tensor<128x256xf32, #mma>
        %89 = tt.addptr %arg12, %cst {async_task_id = dense<0> : vector<1xi32>} : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
        %90 = tt.addptr %arg13, %29 {async_task_id = dense<0> : vector<1xi32>} : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
        scf.yield {async_task_id = dense<[0, 1, 2]> : vector<3xi32>} %88, %89, %90 : tensor<128x256xf32, #mma>, tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<64x256x!tt.ptr<f16>, #blocked1>
      } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
      %62 = arith.truncf %61#0 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %63 = tt.expand_dims %46 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %64 = arith.muli %30, %63 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked1>
      %65 = tt.addptr %31, %64 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
      %66 = tt.expand_dims %50 {async_task_id = dense<[1, 2]> : vector<2xi32>, axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
      %67 = tt.broadcast %65 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
      %68 = tt.broadcast %66 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
      %69 = tt.addptr %67, %68 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
      %70 = arith.cmpi slt, %63, %32 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi32, #blocked1>
      %71 = arith.cmpi slt, %66, %33 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi32, #blocked1>
      %72 = tt.broadcast %70 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x1xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
      %73 = tt.broadcast %71 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<1x256xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
      %74 = arith.andi %72, %73 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xi1, #blocked1>
      %75 = triton_gpu.convert_layout %62 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.store %69, %75, %74 {async_task_id = dense<[1, 2]> : vector<2xi32>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    } {async_task_id = dense<[0, 1, 2]> : vector<3xi32>}
    tt.return
  }
}
