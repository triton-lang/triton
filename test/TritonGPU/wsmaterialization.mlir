// RUN: triton-opt -split-input-file -triton-nvidia-gpu-ws-materialization='compute-capability=90' %s | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"async.num-agents" = 2 : i32, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @simple_gemm
  // CHECK: triton_nvidia_gpu.alloc_mbarrier
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_nvidia_gpu.insert_slice_async_v2
  // CHECK: triton_nvidia_gpu.insert_slice_async_v2
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_gpu.extract_slice
  // CHECK: triton_gpu.extract_slice
  // CHECK: triton_nvidia_gpu.dot_async
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: triton_nvidia_gpu.dot_wait
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: tt.store
  tt.func public @simple_gemm(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.max_divisibility = 8 : i32}, %arg11: i32 {tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = triton_gpu.alloc_tensor : tensor<3x128x64xf16, #shared>
    %1 = triton_gpu.alloc_tensor : tensor<3x64x128xf16, #shared1>
    %2 = triton_nvidia_gpu.create_token {num = 3 : i32} : tensor<3x!triton_nvidia_gpu.token>
    %c0_i32 = arith.constant 0 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.addi %arg6, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.addi %arg5, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.muli %5, %c8_i32 : i32
    %9 = arith.divsi %3, %8 : i32
    %10 = arith.muli %9, %c8_i32 : i32
    %11 = arith.subi %7, %10 : i32
    %12 = arith.minsi %11, %c8_i32 : i32
    %13 = arith.remsi %3, %12 : i32
    %14 = arith.addi %10, %13 : i32
    %15 = arith.remsi %3, %8 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = arith.muli %16, %c128_i32 : i32
    %19 = arith.extsi %arg5 : i32 to i64
    %20 = arith.extsi %arg7 : i32 to i64
    %21 = arith.extsi %arg8 : i32 to i64
    %22 = tt.make_tensor_ptr %arg0, [%19, %20], [%21, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf16, #blocked>, 1>
    %23 = arith.extsi %arg6 : i32 to i64
    %24 = arith.extsi %arg9 : i32 to i64
    %25 = tt.make_tensor_ptr %arg1, [%20, %23], [%c1_i64, %24], [%c0_i32, %18] {order = array<i32: 0, 1>} : <tensor<64x128xf16, #blocked1>, 1>
    %26 = arith.extsi %arg11 : i32 to i64
    %27 = tt.make_tensor_ptr %arg4, [%19, %23], [%26, %c1_i64], [%17, %18] {order = array<i32: 1, 0>} : <tensor<128x128xf32, #blocked>, 1>
    %28 = triton_nvidia_gpu.get_agent_id : i32
    %c0_i32_0 = arith.constant 0 : i32
    %29 = arith.cmpi eq, %28, %c0_i32_0 : i32
    scf.if %29 {
      %c64_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 64 : i32
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %false = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} false
      %31:4 = scf.for %arg12 = %c0_i32 to %arg7 step %c64_i32 iter_args(%arg13 = %22, %arg14 = %25, %arg15 = %false, %arg16 = %c0_i32_1) -> (!tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>, i1, i32)  : i32 {
        triton_nvidia_gpu.producer_acquire %2, %arg16 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        %32 = triton_gpu.insert_slice %arg13, %0, %arg16 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xf16, #blocked>, 1> -> tensor<3x128x64xf16, #shared>
        %33 = triton_gpu.insert_slice %arg14, %1, %arg16 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x128xf16, #blocked1>, 1> -> tensor<3x64x128xf16, #shared1>
        triton_nvidia_gpu.producer_commit %2, %arg16 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        %34 = tt.advance %arg13, [%c0_i32, %c64_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<128x64xf16, #blocked>, 1>
        %35 = tt.advance %arg14, [%c64_i32, %c0_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<64x128xf16, #blocked1>, 1>
        %c1_i32_2 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
        %c0_i32_3 = arith.constant {async_agent = dense<0> : vector<1xi32>} 0 : i32
        %true = arith.constant {async_agent = dense<0> : vector<1xi32>} true
        %36 = arith.addi %arg16, %c1_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
        %37 = arith.cmpi uge, %36, %c3_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %38 = arith.cmpi ult, %36, %c3_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %39 = arith.subi %36, %c3_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %40 = arith.select %37, %39, %36 {async_agent = dense<0> : vector<1xi32>} : i32
        %41 = arith.xori %arg15, %true {async_agent = dense<0> : vector<1xi32>} : i1
        %42 = arith.andi %37, %41 {async_agent = dense<0> : vector<1xi32>} : i1
        %43 = arith.andi %38, %arg15 {async_agent = dense<0> : vector<1xi32>} : i1
        %44 = arith.ori %42, %43 {async_agent = dense<0> : vector<1xi32>} : i1
        scf.yield {async_agent = dense<0> : vector<1xi32>} %34, %35, %44, %40 : !tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>, i1, i32
      } {async_agent = dense<0> : vector<1xi32>}
    } {async_agent = dense<0> : vector<1xi32>}
    %c1_i32 = arith.constant 1 : i32
    %30 = arith.cmpi eq, %28, %c1_i32 : i32
    scf.if %30 {
      %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
      %c64_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 64 : i32
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %false = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} false
      %31:3 = scf.for %arg12 = %c0_i32 to %arg7 step %c64_i32 iter_args(%arg13 = %cst, %arg14 = %false, %arg15 = %c0_i32_1) -> (tensor<128x128xf32, #mma>, i1, i32)  : i32 {
        triton_nvidia_gpu.consumer_wait %2, %arg15 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        %37 = triton_gpu.extract_slice %0[%arg15, 0, 0] [1, 128, 64] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x128x64xf16, #shared> to tensor<128x64xf16, #shared>
        %38 = triton_gpu.convert_layout %37 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x64xf16, #shared>) -> tensor<128x64xf16, #shared>
        %39 = triton_gpu.extract_slice %1[%arg15, 0, 0] [1, 64, 128] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x64x128xf16, #shared1> to tensor<64x128xf16, #shared1>
        %40 = triton_gpu.convert_layout %39 {async_agent = dense<1> : vector<1xi32>} : (tensor<64x128xf16, #shared1>) -> tensor<64x128xf16, #shared1>
        %41 = triton_nvidia_gpu.dot_async %38, %40, %arg13 {allowTF32 = true, async_agent = dense<1> : vector<1xi32>, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #shared> * tensor<64x128xf16, #shared1> -> tensor<128x128xf32, #mma>
        %42 = arith.cmpi sgt, %arg12, %c0_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        scf.if %42 {
          %c0_i32_6 = arith.constant {async_agent = dense<1> : vector<1xi32>} 0 : i32
          %c1_i32_7 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
          %c2_i32_8 = arith.constant {async_agent = dense<1> : vector<1xi32>} 2 : i32
          %52 = arith.subi %arg15, %c1_i32_7 {async_agent = dense<1> : vector<1xi32>} : i32
          %53 = arith.cmpi eq, %arg15, %c0_i32_6 {async_agent = dense<1> : vector<1xi32>} : i32
          %54 = arith.select %53, %c2_i32_8, %52 {async_agent = dense<1> : vector<1xi32>} : i32
          triton_nvidia_gpu.consumer_release %2, %54 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        } {async_agent = dense<1> : vector<1xi32>}
        %c1_i32_4 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
        %c0_i32_5 = arith.constant {async_agent = dense<1> : vector<1xi32>} 0 : i32
        %true = arith.constant {async_agent = dense<1> : vector<1xi32>} true
        %43 = arith.addi %arg15, %c1_i32_4 {async_agent = dense<1> : vector<1xi32>} : i32
        %44 = arith.cmpi uge, %43, %c3_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %45 = arith.cmpi ult, %43, %c3_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %46 = arith.subi %43, %c3_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        %47 = arith.select %44, %46, %43 {async_agent = dense<1> : vector<1xi32>} : i32
        %48 = arith.xori %arg14, %true {async_agent = dense<1> : vector<1xi32>} : i1
        %49 = arith.andi %44, %48 {async_agent = dense<1> : vector<1xi32>} : i1
        %50 = arith.andi %45, %arg14 {async_agent = dense<1> : vector<1xi32>} : i1
        %51 = arith.ori %49, %50 {async_agent = dense<1> : vector<1xi32>} : i1
        scf.yield {async_agent = dense<1> : vector<1xi32>} %41, %51, %47 : tensor<128x128xf32, #mma>, i1, i32
      } {async_agent = dense<1> : vector<1xi32>}
      %32 = triton_nvidia_gpu.dot_wait %31#0 {async_agent = dense<1> : vector<1xi32>, pendings = 0 : i32} : tensor<128x128xf32, #mma>
      %c0_i32_2 = arith.constant {async_agent = dense<1> : vector<1xi32>} 0 : i32
      %c1_i32_3 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
      %c2_i32 = arith.constant {async_agent = dense<1> : vector<1xi32>} 2 : i32
      %33 = arith.subi %31#2, %c1_i32_3 {async_agent = dense<1> : vector<1xi32>} : i32
      %34 = arith.cmpi eq, %31#2, %c0_i32_2 {async_agent = dense<1> : vector<1xi32>} : i32
      %35 = arith.select %34, %c2_i32, %33 {async_agent = dense<1> : vector<1xi32>} : i32
      triton_nvidia_gpu.consumer_release %2, %35 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
      %36 = triton_gpu.convert_layout %32 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x128xf32, #mma>) -> tensor<128x128xf32, #blocked2>
      tt.store %27, %36 {async_agent = dense<1> : vector<1xi32>, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x128xf32, #blocked>, 1>, tensor<128x128xf32, #blocked2>
    } {async_agent = dense<1> : vector<1xi32>}
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"async.num-agents" = 2 : i32, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @matmal_from_wsmutex
  // CHECK: triton_nvidia_gpu.alloc_mbarrier
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_nvidia_gpu.insert_slice_async_v2
  // CHECK: triton_nvidia_gpu.insert_slice_async_v2
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: scf.if
  // CHECK: triton_nvidia_gpu.bar_wait
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_gpu.extract_slice
  // CHECK: triton_gpu.extract_slice
  // CHECK: triton_nvidia_gpu.dot_async
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: triton_nvidia_gpu.bar_arrive
  // CHECK: triton_nvidia_gpu.dot_wait
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: triton_nvidia_gpu.bar_wait
  // CHECK: tt.store
  // CHECK: triton_nvidia_gpu.bar_arrive
  tt.func public @matmal_from_wsmutex(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %0 = triton_gpu.alloc_tensor : tensor<3x64x16xf16, #shared>
    %1 = triton_gpu.alloc_tensor : tensor<3x16x64xf16, #shared1>
    %2 = triton_nvidia_gpu.create_token {num = 3 : i32} : tensor<3x!triton_nvidia_gpu.token>
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.addi %arg6, %c63_i32 : i32
    %5 = arith.divsi %4, %c64_i32 : i32
    %6 = arith.addi %arg5, %c63_i32 : i32
    %7 = arith.divsi %6, %c64_i32 : i32
    %8 = arith.muli %5, %c8_i32 : i32
    %9 = arith.divsi %3, %8 : i32
    %10 = arith.muli %9, %c8_i32 : i32
    %11 = arith.subi %7, %10 : i32
    %12 = arith.minsi %11, %c8_i32 : i32
    %13 = arith.remsi %3, %8 : i32
    %14 = arith.remsi %13, %12 : i32
    %15 = arith.addi %10, %14 : i32
    %16 = arith.divsi %13, %12 : i32
    %17 = arith.muli %15, %c64_i32 : i32
    %18 = arith.muli %16, %c64_i32 : i32
    %19 = arith.extsi %arg5 : i32 to i64
    %20 = arith.extsi %arg7 : i32 to i64
    %21 = arith.extsi %arg8 : i32 to i64
    %22 = tt.make_tensor_ptr %arg0, [%19, %20], [%21, %c1_i64], [%17, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %23 = arith.extsi %arg6 : i32 to i64
    %24 = arith.extsi %arg9 : i32 to i64
    %25 = tt.make_tensor_ptr %arg1, [%20, %23], [%c1_i64, %24], [%c0_i32, %18] {order = array<i32: 0, 1>} : <tensor<16x64xf16, #blocked1>, 1>
    %26 = arith.extsi %arg10 : i32 to i64
    %27 = tt.make_tensor_ptr %arg4, [%19, %23], [%26, %c1_i64], [%17, %18] {order = array<i32: 1, 0>} : <tensor<64x64xf16, #blocked>, 1>
    %28 = triton_nvidia_gpu.get_agent_id : i32
    %c0_i32_0 = arith.constant 0 : i32
    %29 = arith.cmpi eq, %28, %c0_i32_0 : i32
    scf.if %29 {
      %c132_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 132 : i32
      %c15_i32 = arith.constant {async_agent = dense<0> : vector<1xi32>} 15 : i32
      %c16_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 16 : i32
      %31 = arith.muli %7, %5 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %32 = arith.addi %arg7, %c15_i32 {async_agent = dense<0> : vector<1xi32>} : i32
      %33 = arith.divsi %32, %c16_i32 {async_agent = dense<0> : vector<1xi32>} : i32
      %34 = arith.subi %c0_i32, %33 {async_agent = dense<0> : vector<1xi32>} : i32
      %35 = arith.muli %34, %c16_i32 {async_agent = dense<0> : vector<1xi32>} : i32
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %36:5 = scf.for %arg11 = %3 to %31 step %c132_i32 iter_args(%arg12 = %22, %arg13 = %25, %arg14 = %15, %arg15 = %16, %arg16 = %c0_i32_1) -> (!tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i32, i32, i32)  : i32 {
        %37 = arith.divsi %arg11, %8 {async_agent = dense<0> : vector<1xi32>} : i32
        %38 = arith.muli %37, %c8_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %39 = arith.subi %7, %38 {async_agent = dense<0> : vector<1xi32>} : i32
        %40 = arith.minsi %39, %c8_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %41 = arith.remsi %arg11, %8 {async_agent = dense<0> : vector<1xi32>} : i32
        %42 = arith.remsi %41, %40 {async_agent = dense<0> : vector<1xi32>} : i32
        %43 = arith.addi %38, %42 {async_agent = dense<0> : vector<1xi32>} : i32
        %44 = arith.divsi %41, %40 {async_agent = dense<0> : vector<1xi32>} : i32
        %45 = arith.subi %43, %arg14 {async_agent = dense<0> : vector<1xi32>} : i32
        %46 = arith.muli %45, %c64_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %47 = tt.advance %arg12, [%46, %c0_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<64x16xf16, #blocked>, 1>
        %48 = arith.subi %44, %arg15 {async_agent = dense<0> : vector<1xi32>} : i32
        %49 = arith.muli %48, %c64_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %50 = tt.advance %arg13, [%c0_i32, %49] {async_agent = dense<0> : vector<1xi32>} : <tensor<16x64xf16, #blocked1>, 1>
        %c3_i32_2 = arith.constant {async_agent = dense<0> : vector<1xi32>} 3 : i32
        %c0_i32_3 = arith.constant {async_agent = dense<0> : vector<1xi32>} 0 : i32
        %51 = arith.subi %arg7, %c0_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %52 = arith.addi %51, %c16_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %c1_i32_4 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
        %c2_i32 = arith.constant {async_agent = dense<0> : vector<1xi32>} 2 : i32
        %53 = arith.subi %52, %c1_i32_4 {async_agent = dense<0> : vector<1xi32>} : i32
        %54 = arith.divui %53, %c16_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        %55 = arith.muli %arg16, %54 {async_agent = dense<0> : vector<1xi32>} : i32
        %56 = arith.divui %55, %c3_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
        %57 = arith.muli %56, %c3_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
        %58 = arith.subi %55, %57 {async_agent = dense<0> : vector<1xi32>} : i32
        %59 = arith.andi %56, %c1_i32_4 {async_agent = dense<0> : vector<1xi32>} : i32
        %60 = arith.trunci %59 {async_agent = dense<0> : vector<1xi32>} : i32 to i1
        %61:4 = scf.for %arg17 = %c0_i32 to %arg7 step %c16_i32 iter_args(%arg18 = %47, %arg19 = %50, %arg20 = %60, %arg21 = %58) -> (!tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i1, i32)  : i32 {
          triton_nvidia_gpu.producer_acquire %2, %arg21 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %65 = triton_gpu.insert_slice %arg18, %0, %arg21 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<3x64x16xf16, #shared>
          %66 = triton_gpu.insert_slice %arg19, %1, %arg21 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<3x16x64xf16, #shared1>
          triton_nvidia_gpu.producer_commit %2, %arg21 {async_agent = dense<0> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %67 = tt.advance %arg18, [%c0_i32, %c16_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<64x16xf16, #blocked>, 1>
          %68 = tt.advance %arg19, [%c16_i32, %c0_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<16x64xf16, #blocked1>, 1>
          %c1_i32_6 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
          %c0_i32_7 = arith.constant {async_agent = dense<0> : vector<1xi32>} 0 : i32
          %true = arith.constant {async_agent = dense<0> : vector<1xi32>} true
          %69 = arith.addi %arg21, %c1_i32_6 {async_agent = dense<0> : vector<1xi32>} : i32
          %70 = arith.cmpi uge, %69, %c3_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
          %71 = arith.cmpi ult, %69, %c3_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
          %72 = arith.subi %69, %c3_i32_2 {async_agent = dense<0> : vector<1xi32>} : i32
          %73 = arith.select %70, %72, %69 {async_agent = dense<0> : vector<1xi32>} : i32
          %74 = arith.xori %arg20, %true {async_agent = dense<0> : vector<1xi32>} : i1
          %75 = arith.andi %70, %74 {async_agent = dense<0> : vector<1xi32>} : i1
          %76 = arith.andi %71, %arg20 {async_agent = dense<0> : vector<1xi32>} : i1
          %77 = arith.ori %75, %76 {async_agent = dense<0> : vector<1xi32>} : i1
          scf.yield {async_agent = dense<0> : vector<1xi32>} %67, %68, %77, %73 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i1, i32
        } {async_agent = dense<0> : vector<1xi32>}
        %62 = tt.advance %61#0, [%c0_i32, %35] {async_agent = dense<0> : vector<1xi32>} : <tensor<64x16xf16, #blocked>, 1>
        %63 = tt.advance %61#1, [%35, %c0_i32] {async_agent = dense<0> : vector<1xi32>} : <tensor<16x64xf16, #blocked1>, 1>
        %c1_i32_5 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
        %64 = arith.addi %arg16, %c1_i32_5 {async_agent = dense<0> : vector<1xi32>} : i32
        scf.yield {async_agent = dense<0> : vector<1xi32>} %62, %63, %43, %44, %64 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i32, i32, i32
      } {async_agent = dense<0> : vector<1xi32>}
    } {async_agent = dense<0> : vector<1xi32>}
    %c1_i32 = arith.constant 1 : i32
    %30 = arith.cmpi eq, %28, %c1_i32 : i32
    scf.if %30 {
      %c0_i32_1 = arith.constant 0 : i32
      %31 = triton_nvidia_gpu.get_mutex_role_id {async_agent = dense<1> : vector<1xi32>, num = 2 : i32} : i32
      %32 = arith.cmpi ne, %31, %c0_i32_1 : i32
      %33 = triton_nvidia_gpu.create_mutex {async_agent = dense<1> : vector<1xi32>} : !triton_nvidia_gpu.mutex
      %34 = triton_nvidia_gpu.create_mutex {async_agent = dense<1> : vector<1xi32>} : !triton_nvidia_gpu.mutex
      %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<64x64xf32, #mma>
      %c132_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 132 : i32
      %c16_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 16 : i32
      %35 = arith.muli %7, %5 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_2 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %36 = arith.muli %c132_i32, %31 {async_agent = dense<1> : vector<1xi32>} : i32
      %37 = arith.addi %3, %36 {async_agent = dense<1> : vector<1xi32>} : i32
      %c2_i32 = arith.constant {async_agent = dense<1> : vector<1xi32>} 2 : i32
      %38 = arith.muli %c132_i32, %c2_i32 {async_agent = dense<1> : vector<1xi32>} : i32
      %39 = arith.addi %c0_i32_2, %31 {async_agent = dense<1> : vector<1xi32>} : i32
      %40:4 = scf.for %arg11 = %37 to %35 step %38 iter_args(%arg12 = %27, %arg13 = %15, %arg14 = %16, %arg15 = %39) -> (!tt.ptr<tensor<64x64xf16, #blocked>, 1>, i32, i32, i32)  : i32 {
        %41 = arith.cmpi ne, %arg11, %3 : i32
        %42 = arith.ori %41, %32 : i1
        scf.if %42 {
          triton_nvidia_gpu.lock %33 {agent.mutex_role = 0 : i32} : !triton_nvidia_gpu.mutex
        } {agent.mutex_role = 0 : i32}
        %43 = arith.divsi %arg11, %8 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %44 = arith.muli %43, %c8_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %45 = arith.subi %7, %44 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %46 = arith.minsi %45, %c8_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %47 = arith.remsi %arg11, %8 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %48 = arith.remsi %47, %46 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %49 = arith.addi %44, %48 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %50 = arith.divsi %47, %46 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %51 = arith.subi %49, %arg13 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %52 = arith.muli %51, %c64_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %53 = arith.subi %50, %arg14 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %54 = arith.muli %53, %c64_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %c3_i32_3 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 3 : i32
        %c0_i32_4 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 0 : i32
        %55 = arith.subi %arg7, %c0_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %56 = arith.addi %55, %c16_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %c1_i32_5 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
        %c2_i32_6 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 2 : i32
        %57 = arith.subi %56, %c1_i32_5 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %58 = arith.divui %57, %c16_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %59 = arith.muli %arg15, %58 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %60 = arith.divui %59, %c3_i32_3 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %61 = arith.muli %60, %c3_i32_3 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %62 = arith.subi %59, %61 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %63 = arith.andi %60, %c1_i32_5 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %64 = arith.trunci %63 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32 to i1
        %65:3 = scf.for %arg16 = %c0_i32 to %arg7 step %c16_i32 iter_args(%arg17 = %cst, %arg18 = %64, %arg19 = %62) -> (tensor<64x64xf32, #mma>, i1, i32)  : i32 {
          triton_nvidia_gpu.consumer_wait %2, %arg19 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %74 = triton_gpu.extract_slice %0[%arg19, 0, 0] [1, 64, 16] [1, 1, 1] {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x64x16xf16, #shared> to tensor<64x16xf16, #shared>
          %75 = triton_gpu.convert_layout %74 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<64x16xf16, #shared>) -> tensor<64x16xf16, #shared>
          %76 = triton_gpu.extract_slice %1[%arg19, 0, 0] [1, 16, 64] [1, 1, 1] {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x16x64xf16, #shared1> to tensor<16x64xf16, #shared1>
          %77 = triton_gpu.convert_layout %76 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<16x64xf16, #shared1>) -> tensor<16x64xf16, #shared1>
          %78 = triton_nvidia_gpu.dot_async %75, %77, %arg17 {agent.mutex_role = 0 : i32, allowTF32 = true, async_agent = dense<1> : vector<1xi32>, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
          %79 = arith.cmpi sgt, %arg16, %c0_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          scf.if %79 {
            %c0_i32_13 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 0 : i32
            %c1_i32_14 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
            %c2_i32_15 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 2 : i32
            %89 = arith.subi %arg19, %c1_i32_14 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
            %90 = arith.cmpi eq, %arg19, %c0_i32_13 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
            %91 = arith.select %90, %c2_i32_15, %89 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
            triton_nvidia_gpu.consumer_release %2, %91 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          } {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>}
          %c1_i32_11 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
          %c0_i32_12 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 0 : i32
          %true = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} true
          %80 = arith.addi %arg19, %c1_i32_11 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          %81 = arith.cmpi uge, %80, %c3_i32_3 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          %82 = arith.cmpi ult, %80, %c3_i32_3 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          %83 = arith.subi %80, %c3_i32_3 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          %84 = arith.select %81, %83, %80 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          %85 = arith.xori %arg18, %true {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i1
          %86 = arith.andi %81, %85 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i1
          %87 = arith.andi %82, %arg18 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i1
          %88 = arith.ori %86, %87 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i1
          scf.yield {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} %78, %88, %84 : tensor<64x64xf32, #mma>, i1, i32
        } {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>}
        triton_nvidia_gpu.unlock %33 : !triton_nvidia_gpu.mutex
        %66 = triton_nvidia_gpu.dot_wait %65#0 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>, pendings = 0 : i32} : tensor<64x64xf32, #mma>
        %c0_i32_7 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 0 : i32
        %c1_i32_8 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
        %c2_i32_9 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 2 : i32
        %67 = arith.subi %65#2, %c1_i32_8 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %68 = arith.cmpi eq, %65#2, %c0_i32_7 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %69 = arith.select %68, %c2_i32_9, %67 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        triton_nvidia_gpu.consumer_release %2, %69 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        scf.if %42 {
          triton_nvidia_gpu.lock %34 {agent.mutex_role = 1 : i32} : !triton_nvidia_gpu.mutex
        } {agent.mutex_role = 1 : i32}
        %70 = arith.truncf %66 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
        %71 = tt.advance %arg12, [%52, %54] {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : <tensor<64x64xf16, #blocked>, 1>
        %72 = triton_gpu.convert_layout %70 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<64x64xf16, #mma>) -> tensor<64x64xf16, #blocked2>
        tt.store %71, %72 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>, boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<64x64xf16, #blocked2>
        triton_nvidia_gpu.unlock %34 : !triton_nvidia_gpu.mutex
        %c1_i32_10 = arith.constant {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
        %73 = arith.addi %arg15, %c2_i32 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        scf.yield {async_agent = dense<1> : vector<1xi32>} %71, %49, %50, %73 : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, i32, i32, i32
      } {async_agent = dense<1> : vector<1xi32>}
    } {"agent.num-roles" = 2 : i32, async_agent = dense<1> : vector<1xi32>}
    tt.return
  }
}
