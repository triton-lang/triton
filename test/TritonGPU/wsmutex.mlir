// RUN: triton-opt -triton-nvidia-gpu-ws-mutex='compute-capability=90' %s | FileCheck %s


// CHECK: scf.if
// CHECK: triton_nvidia_gpu.create_mutex
// CHECK: triton_nvidia_gpu.create_mutex
// CHECK: scf.for
// CHECK: triton_nvidia_gpu.lock
// CHECK: agent.mutex_role = 0 : i32
// CHECK: triton_nvidia_gpu.unlock
// CHECK: triton_nvidia_gpu.lock
// CHECK: agent.mutex_role = 1 : i32
// CHECK: triton_nvidia_gpu.unlock

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"async.num-agents" = 2 : i32, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @static_persistent_warp_specialized_matmul_kernel_0d1d2d3de4de5de6de7c8c9de10de11c(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %0 = triton_gpu.alloc_tensor : tensor<3x64x16xf16, #shared>
    %1 = triton_gpu.alloc_tensor : tensor<3x16x64xf16, #shared1>
    %2 = triton_nvidia_gpu.create_token {num = 3 : i32} : tensor<3x!triton_nvidia_gpu.token>
    %3 = triton_nvidia_gpu.get_agent_id : i32
    %c0_i32 = arith.constant 0 : i32
    %4 = arith.cmpi eq, %3, %c0_i32 : i32
    scf.if %4 {
      %cst = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<16> : tensor<16x64xi32, #blocked>
      %cst_0 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<16> : tensor<64x16xi32, #blocked1>
      %c63_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 63 : i32
      %c114_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 114 : i32
      %c16_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 16 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %c64_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 64 : i32
      %6 = tt.get_program_id x {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
      %7 = arith.addi %arg3, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %8 = arith.divsi %7, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = arith.addi %arg4, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %10 = arith.divsi %9, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.muli %8, %10 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %13 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %14 = tt.splat %arg3 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %15 = tt.splat %arg4 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %16 = tt.splat %arg6 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<64x1xi32, #blocked1>
      %17 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %18 = tt.expand_dims %17 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x16xi32, #blocked1>
      %19 = tt.broadcast %18 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x16xi32, #blocked1>) -> tensor<64x16xi32, #blocked1>
      %20 = tt.splat %arg0 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<64x16x!tt.ptr<f16, 1>, #blocked1>
      %21 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %22 = tt.expand_dims %21 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<16x1xi32, #blocked>
      %23 = tt.splat %arg7 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<1x64xi32, #blocked>
      %24 = tt.broadcast %22 {async_agent = dense<0> : vector<1xi32>} : (tensor<16x1xi32, #blocked>) -> tensor<16x64xi32, #blocked>
      %25 = tt.splat %arg1 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<16x64x!tt.ptr<f16, 1>, #blocked>
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_2 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %26 = scf.for %arg9 = %6 to %11 step %c114_i32 iter_args(%arg10 = %c0_i32_2) -> (i32)  : i32 {
        %27 = arith.divsi %arg9, %10 {async_agent = dense<0> : vector<1xi32>} : i32
        %28 = arith.remsi %arg9, %10 {async_agent = dense<0> : vector<1xi32>} : i32
        %29 = arith.muli %27, %c64_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %30 = tt.splat %29 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        %31 = arith.addi %30, %12 {async_agent = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        %32 = arith.remsi %31, %14 {async_agent = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        %33 = arith.muli %28, %c64_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %34 = tt.splat %33 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %35 = arith.addi %34, %13 {async_agent = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %36 = arith.remsi %35, %15 {async_agent = dense<0> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
        %37 = tt.expand_dims %32 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
        %38 = arith.muli %37, %16 {async_agent = dense<0> : vector<1xi32>} : tensor<64x1xi32, #blocked1>
        %39 = tt.broadcast %38 {async_agent = dense<0> : vector<1xi32>} : (tensor<64x1xi32, #blocked1>) -> tensor<64x16xi32, #blocked1>
        %40 = arith.addi %39, %19 {async_agent = dense<0> : vector<1xi32>} : tensor<64x16xi32, #blocked1>
        %41 = tt.addptr %20, %40 {async_agent = dense<0> : vector<1xi32>} : tensor<64x16x!tt.ptr<f16, 1>, #blocked1>, tensor<64x16xi32, #blocked1>
        %42 = tt.expand_dims %36 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x64xi32, #blocked>
        %43 = arith.muli %42, %23 {async_agent = dense<0> : vector<1xi32>} : tensor<1x64xi32, #blocked>
        %44 = tt.broadcast %43 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x64xi32, #blocked>) -> tensor<16x64xi32, #blocked>
        %45 = arith.addi %24, %44 {async_agent = dense<0> : vector<1xi32>} : tensor<16x64xi32, #blocked>
        %46 = tt.addptr %25, %45 {async_agent = dense<0> : vector<1xi32>} : tensor<16x64x!tt.ptr<f16, 1>, #blocked>, tensor<16x64xi32, #blocked>
        %c3_i32_3 = arith.constant {async_agent = dense<0> : vector<1xi32>} 3 : i32
        %47 = arith.subi %arg5, %c0_i32_1 {async_agent = dense<0> : vector<1xi32>} : i32
        %48 = arith.divui %47, %c16_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %49 = arith.muli %arg10, %48 {async_agent = dense<0> : vector<1xi32>} : i32
        %c3_i32_4 = arith.constant {async_agent = dense<0> : vector<1xi32>} 3 : i32
        %50:3 = scf.for %arg11 = %c0_i32_1 to %arg5 step %c16_i32 iter_args(%arg12 = %41, %arg13 = %46, %arg14 = %49) -> (tensor<64x16x!tt.ptr<f16, 1>, #blocked1>, tensor<16x64x!tt.ptr<f16, 1>, #blocked>, i32)  : i32 {
          %52 = arith.remsi %arg14, %c3_i32_4 {async_agent = dense<0> : vector<1xi32>} : i32
          triton_nvidia_gpu.producer_acquire %2, %52 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %53 = triton_gpu.insert_slice %arg12, %0, %52 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x16x!tt.ptr<f16, 1>, #blocked1> -> tensor<3x64x16xf16, #shared>
          %54 = triton_gpu.insert_slice %arg13, %1, %52 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x64x!tt.ptr<f16, 1>, #blocked> -> tensor<3x16x64xf16, #shared1>
          triton_nvidia_gpu.producer_commit %2, %52 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %55 = tt.addptr %arg12, %cst_0 {async_agent = dense<0> : vector<1xi32>} : tensor<64x16x!tt.ptr<f16, 1>, #blocked1>, tensor<64x16xi32, #blocked1>
          %56 = tt.addptr %arg13, %cst {async_agent = dense<0> : vector<1xi32>} : tensor<16x64x!tt.ptr<f16, 1>, #blocked>, tensor<16x64xi32, #blocked>
          %c1_i32_6 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
          %57 = arith.addi %arg14, %c1_i32_6 {async_agent = dense<0> : vector<1xi32>} : i32
          scf.yield {async_agent = dense<0> : vector<1xi32>} %55, %56, %57 : tensor<64x16x!tt.ptr<f16, 1>, #blocked1>, tensor<16x64x!tt.ptr<f16, 1>, #blocked>, i32
        } {async_agent = dense<0> : vector<1xi32>}
        %c1_i32_5 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
        %51 = arith.addi %arg10, %c1_i32_5 {async_agent = dense<0> : vector<1xi32>} : i32
        scf.yield {async_agent = dense<0> : vector<1xi32>} %51 : i32
      } {async_agent = dense<0> : vector<1xi32>}
    } {async_agent = dense<0> : vector<1xi32>}
    %c1_i32 = arith.constant 1 : i32
    %5 = arith.cmpi eq, %3, %c1_i32 : i32
    scf.if %5 {
      %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<64x64xf32, #mma>
      %c63_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 63 : i32
      %c114_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 114 : i32
      %c16_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 16 : i32
      %c0_i32_0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %c64_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 64 : i32
      %6 = tt.get_program_id x {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
      %7 = arith.addi %arg3, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %8 = arith.divsi %7, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = arith.addi %arg4, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %10 = arith.divsi %9, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.muli %8, %10 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %13 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %14 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64x1xi32, #blocked2>
      %15 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked2>
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %16 = scf.for %arg9 = %6 to %11 step %c114_i32 iter_args(%arg10 = %c0_i32_1) -> (i32)  : i32 {
        %17 = arith.divsi %arg9, %10 {async_agent = dense<1> : vector<1xi32>} : i32
        %18 = arith.remsi %arg9, %10 {async_agent = dense<1> : vector<1xi32>} : i32
        %19 = arith.muli %17, %c64_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %20 = tt.splat %19 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
        %21 = arith.addi %20, %13 {async_agent = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
        %22 = arith.muli %18, %c64_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %23 = tt.splat %22 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
        %24 = arith.addi %23, %12 {async_agent = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
        %c3_i32_2 = arith.constant {async_agent = dense<1> : vector<1xi32>} 3 : i32
        %25 = arith.subi %arg5, %c0_i32_0 {async_agent = dense<1> : vector<1xi32>} : i32
        %26 = arith.divui %25, %c16_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %27 = arith.muli %arg10, %26 {async_agent = dense<1> : vector<1xi32>} : i32
        %c3_i32_3 = arith.constant {async_agent = dense<1> : vector<1xi32>} 3 : i32
        %28:2 = scf.for %arg11 = %c0_i32_0 to %arg5 step %c16_i32 iter_args(%arg12 = %cst, %arg13 = %27) -> (tensor<64x64xf32, #mma>, i32)  : i32 {
          %38 = arith.remsi %arg13, %c3_i32_3 {async_agent = dense<1> : vector<1xi32>} : i32
          triton_nvidia_gpu.consumer_wait %2, %38 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %39 = triton_gpu.extract_slice %0[%38, 0, 0] [1, 64, 16] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x64x16xf16, #shared> to tensor<64x16xf16, #shared>
          %40 = triton_gpu.convert_layout %39 {async_agent = dense<1> : vector<1xi32>} : (tensor<64x16xf16, #shared>) -> tensor<64x16xf16, #shared>
          %41 = triton_gpu.extract_slice %1[%38, 0, 0] [1, 16, 64] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x16x64xf16, #shared1> to tensor<16x64xf16, #shared1>
          %42 = triton_gpu.convert_layout %41 {async_agent = dense<1> : vector<1xi32>} : (tensor<16x64xf16, #shared1>) -> tensor<16x64xf16, #shared1>
          %43 = tt.dot %40, %42, %arg12 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
          triton_nvidia_gpu.consumer_release %2, %38 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %c1_i32_5 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
          %44 = arith.addi %arg13, %c1_i32_5 {async_agent = dense<1> : vector<1xi32>} : i32
          scf.yield {async_agent = dense<1> : vector<1xi32>} %43, %44 : tensor<64x64xf32, #mma>, i32
        } {async_agent = dense<1> : vector<1xi32>}
        %29 = tt.expand_dims %21 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
        %30 = arith.muli %29, %14 {async_agent = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked2>
        %31 = tt.addptr %15, %30 {async_agent = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f32, 1>, #blocked2>, tensor<64x1xi32, #blocked2>
        %32 = tt.expand_dims %24 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2>
        %33 = tt.broadcast %31 {async_agent = dense<1> : vector<1xi32>} : (tensor<64x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked2>
        %34 = tt.broadcast %32 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
        %35 = tt.addptr %33, %34 {async_agent = dense<1> : vector<1xi32>} : tensor<64x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x64xi32, #blocked2>
        %36 = triton_gpu.convert_layout %28#0 {async_agent = dense<1> : vector<1xi32>} : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked2>
        tt.store %35, %36 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked2>
        %c1_i32_4 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
        %37 = arith.addi %arg10, %c1_i32_4 {async_agent = dense<1> : vector<1xi32>} : i32
        scf.yield {async_agent = dense<1> : vector<1xi32>} %37 : i32
      } {async_agent = dense<1> : vector<1xi32>}
    } {async_agent = dense<1> : vector<1xi32>}
    tt.return
  }
}
