// RUN: triton-opt -split-input-file -triton-nvidia-gpu-ws-materialization='compute-capability=90' %s | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32} {
  // CHECK-LABEL: @simple_gemm
  // CHECK: triton_nvidia_gpu.alloc_mbarrier
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_gpu.insert_slice
  // CHECK: triton_gpu.insert_slice
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
  // CHECK: tt.dot
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: triton_nvidia_gpu.bar_arrive
  // CHECK: triton_nvidia_gpu.bar_wait
  // CHECK: tt.store
  // CHECK: triton_nvidia_gpu.bar_arrive
  tt.func public @simple_gemm(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %0 = triton_gpu.alloc_tensor : tensor<3x32x128xf16, #shared>
    %1 = triton_gpu.alloc_tensor : tensor<3x128x32xf16, #shared1>
    %2 = triton_nvidia_gpu.create_token {num = 3 : i32} : tensor<3x!triton_nvidia_gpu.token>
    %3 = triton_nvidia_gpu.create_mutex : !triton_nvidia_gpu.mutex
    %4 = triton_nvidia_gpu.create_mutex : !triton_nvidia_gpu.mutex
    %5 = triton_nvidia_gpu.get_agent_id : i32
    %c0_i32 = arith.constant 0 : i32
    %6 = arith.cmpi eq, %5, %c0_i32 : i32
    scf.if %6 {
      %cst = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<32x128xi32, #blocked>
      %cst_1 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<128x32xi32, #blocked1>
      %c31_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 31 : i32
      %c127_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 127 : i32
      %c1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 1 : index
      %c0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : index
      %c32_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 32 : i32
      %c128_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 128 : i32
      %c8_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 8 : i32
      %8 = tt.get_program_id x {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = tt.get_program_id y {async_agent = dense<0> : vector<1xi32>} : i32
      %10 = arith.addi %arg3, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.divsi %10, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = arith.addi %arg4, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %13 = arith.divsi %12, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %14 = arith.muli %13, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %15 = arith.divsi %8, %14 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %16 = arith.muli %15, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %17 = arith.subi %11, %16 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %18 = arith.cmpi slt, %17, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %19 = arith.select %18, %17, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %20 = arith.remsi %8, %19 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %21 = arith.addi %16, %20 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %22 = arith.remsi %8, %14 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %23 = arith.divsi %22, %19 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %24 = arith.muli %21, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %25 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %26 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %27 = tt.splat %24 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %28 = arith.addi %27, %25 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %29 = arith.muli %23, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %30 = tt.splat %29 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %31 = arith.addi %30, %26 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %32 = tt.splat %arg3 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %33 = arith.remsi %28, %32 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %34 = tt.splat %arg4 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %35 = arith.remsi %31, %34 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %36 = arith.muli %9, %c32_i32 {async_agent = dense<0> : vector<1xi32>} : i32
      %37 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %38 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %39 = tt.splat %36 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %40 = tt.splat %36 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %41 = arith.addi %39, %37 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
      %42 = arith.addi %40, %38 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %43 = tt.expand_dims %33 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
      %44 = tt.splat %arg6 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked1>
      %45 = arith.muli %43, %44 {async_agent = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
      %46 = tt.expand_dims %41 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
      %47 = tt.broadcast %45 {async_agent = dense<0> : vector<1xi32>} : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
      %48 = tt.broadcast %46 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
      %49 = arith.addi %47, %48 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32xi32, #blocked1>
      %50 = tt.splat %arg0 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      %51 = tt.addptr %50, %49 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
      %52 = tt.expand_dims %42 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
      %53 = tt.expand_dims %35 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
      %54 = tt.splat %arg7 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<1x128xi32, #blocked>
      %55 = arith.muli %53, %54 {async_agent = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked>
      %56 = tt.broadcast %52 {async_agent = dense<0> : vector<1xi32>} : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
      %57 = tt.broadcast %55 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
      %58 = arith.addi %56, %57 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128xi32, #blocked>
      %59 = tt.splat %arg1 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
      %60 = tt.addptr %59, %58 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
      %61 = arith.addi %arg5, %c31_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %62 = arith.divsi %61, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %63 = arith.index_cast %62 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32 to index
      %c0_i32_2 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %64:3 = scf.for %arg9 = %c0 to %63 step %c1 iter_args(%arg10 = %51, %arg11 = %60, %arg12 = %c0_i32_2) -> (tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>, i32) {
        triton_nvidia_gpu.producer_acquire %2, %arg12 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        %65 = triton_gpu.insert_slice %arg10, %1, %arg12 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1> -> tensor<3x128x32xf16, #shared1>
        %66 = triton_gpu.insert_slice %arg11, %0, %arg12 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128x!tt.ptr<f16, 1>, #blocked> -> tensor<3x32x128xf16, #shared>
        %67 = tt.addptr %arg10, %cst_1 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
        %68 = tt.addptr %arg11, %cst {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
        %c1_i32_3 = arith.constant {async_agent = dense<0> : vector<1xi32>} 1 : i32
        %c3_i32 = arith.constant {async_agent = dense<0> : vector<1xi32>} 3 : i32
        %69 = arith.addi %arg12, %c1_i32_3 {async_agent = dense<0> : vector<1xi32>} : i32
        %70 = arith.remsi %69, %c3_i32 {async_agent = dense<0> : vector<1xi32>} : i32
        triton_nvidia_gpu.producer_commit %2, %arg12 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        scf.yield %67, %68, %70 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>, i32
      } {async_agent = dense<0> : vector<1xi32>}
    }
    %c1_i32 = arith.constant 1 : i32
    %c1_i32_0 = arith.constant 1 : i32
    %7 = arith.cmpi sge, %5, %c1_i32_0 : i32
    scf.if %7 {
      %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
      %c31_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 31 : i32
      %c127_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 127 : i32
      %c1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 1 : index
      %c0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : index
      %c32_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 32 : i32
      %c128_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 128 : i32
      %c8_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 8 : i32
      %8 = tt.get_program_id x {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %9 = arith.addi %arg3, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %10 = arith.divsi %9, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %11 = arith.addi %arg4, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = arith.divsi %11, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %13 = arith.muli %12, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %14 = arith.divsi %8, %13 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %15 = arith.muli %14, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %16 = arith.subi %10, %15 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %17 = arith.cmpi slt, %16, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %18 = arith.select %17, %16, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %19 = arith.remsi %8, %18 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %20 = arith.addi %15, %19 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %21 = arith.remsi %8, %13 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %22 = arith.divsi %21, %18 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %23 = arith.muli %20, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %24 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %25 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %26 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %27 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %28 = tt.splat %23 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %29 = tt.splat %23 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %30 = arith.addi %28, %24 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %31 = arith.addi %29, %26 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %32 = arith.muli %22, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %33 = tt.splat %32 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %34 = tt.splat %32 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %35 = arith.addi %33, %25 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %36 = arith.addi %34, %27 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %37 = tt.splat %arg3 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %38 = tt.splat %arg4 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %39 = arith.addi %arg5, %c31_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %40 = arith.divsi %39, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %41 = arith.index_cast %40 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32 to index
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      triton_nvidia_gpu.lock %3 {mutex.barId = dense<1> : vector<1xi32>, mutex.numThreads = dense<256> : vector<1xi32>}  : !triton_nvidia_gpu.mutex
      %42:2 = scf.for %arg9 = %c0 to %41 step %c1 iter_args(%arg10 = %cst, %arg11 = %c0_i32_1) -> (tensor<128x128xf32, #mma>, i32) {
        triton_nvidia_gpu.consumer_wait %2, %arg11 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        %62 = triton_gpu.extract_slice %1[%arg11, 0, 0] [1, 128, 32] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x128x32xf16, #shared1> to tensor<128x32xf16, #shared1>
        %63 = triton_gpu.extract_slice %0[%arg11, 0, 0] [1, 32, 128] [1, 1, 1] {async_agent = dense<1> : vector<1xi32>} : tensor<3x32x128xf16, #shared> to tensor<32x128xf16, #shared>
        %64 = triton_gpu.convert_layout %62 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x32xf16, #shared1>) -> tensor<128x32xf16, #shared1>
        %65 = triton_gpu.convert_layout %63 {async_agent = dense<1> : vector<1xi32>} : (tensor<32x128xf16, #shared>) -> tensor<32x128xf16, #shared>
        %66 = tt.dot %64, %65, %arg10 {allowTF32 = true, async_agent = dense<1> : vector<1xi32>} : tensor<128x32xf16, #shared1> * tensor<32x128xf16, #shared> -> tensor<128x128xf32, #mma>
        %c1_i32_2 = arith.constant {async_agent = dense<1> : vector<1xi32>} 1 : i32
        %c3_i32 = arith.constant {async_agent = dense<1> : vector<1xi32>} 3 : i32
        %67 = arith.addi %arg11, %c1_i32_2 {async_agent = dense<1> : vector<1xi32>} : i32
        %68 = arith.remsi %67, %c3_i32 {async_agent = dense<1> : vector<1xi32>} : i32
        triton_nvidia_gpu.consumer_release %2, %arg11 {async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
        scf.yield %66, %68 : tensor<128x128xf32, #mma>, i32
      } {async_agent = dense<1> : vector<1xi32>}
      triton_nvidia_gpu.unlock %3 {mutex.barId = dense<2> : vector<1xi32>, mutex.numThreads = dense<256> : vector<1xi32>}  : !triton_nvidia_gpu.mutex
      triton_nvidia_gpu.lock %4 {mutex.barId = dense<3> : vector<1xi32>, mutex.numThreads = dense<256> : vector<1xi32>} : !triton_nvidia_gpu.mutex
      %43 = arith.truncf %42#0 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
      %44 = tt.expand_dims %30 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
      %45 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked2>
      %46 = arith.muli %44, %45 {async_agent = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
      %47 = tt.expand_dims %35 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
      %48 = tt.broadcast %46 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
      %49 = tt.broadcast %47 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
      %50 = arith.addi %48, %49 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi32, #blocked2>
      %51 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
      %52 = tt.addptr %51, %50 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
      %53 = "triton_gpu.cmpi"(%31, %37) {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %54 = tt.expand_dims %53 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
      %55 = "triton_gpu.cmpi"(%36, %38) {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %56 = tt.expand_dims %55 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
      %57 = tt.broadcast %54 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
      %58 = tt.broadcast %56 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
      %59 = arith.andi %57, %58 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked2>
      %60 = triton_gpu.convert_layout %43 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
      tt.store %52, %60, %59 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
      triton_nvidia_gpu.unlock %4 {mutex.barId = dense<4> : vector<1xi32>, mutex.numThreads = dense<256> : vector<1xi32>} : !triton_nvidia_gpu.mutex
    }
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"async.num-agents" = 2 : i32, "triton_gpu.compute-capability" = 90 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @matmal_from_wsmutex
  // CHECK: triton_nvidia_gpu.alloc_mbarrier
  // CHECK: scf.if
  // CHECK: scf.for
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_wait
  // CHECK: triton_gpu.insert_slice
  // CHECK: triton_gpu.insert_slice
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
  // CHECK: tt.dot
  // CHECK: triton_nvidia_gpu.extract_mbarrier
  // CHECK: triton_nvidia_gpu.mbarrier_arrive
  // CHECK: scf.yield
  // CHECK: triton_nvidia_gpu.bar_arrive
  // CHECK: triton_nvidia_gpu.bar_wait
  // CHECK: tt.store
  // CHECK: triton_nvidia_gpu.bar_arrive
  tt.func public @matmal_from_wsmutex(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
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
      %c0_i32_0 = arith.constant 0 : i32
      %6 = triton_nvidia_gpu.get_mutex_role_id {async_agent = dense<1> : vector<1xi32>, num = 2 : i32} : i32
      %7 = arith.cmpi ne, %6, %c0_i32_0 : i32
      %8 = triton_nvidia_gpu.create_mutex {async_agent = dense<1> : vector<1xi32>} : !triton_nvidia_gpu.mutex
      %9 = triton_nvidia_gpu.create_mutex {async_agent = dense<1> : vector<1xi32>} : !triton_nvidia_gpu.mutex
      %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<64x64xf32, #mma>
      %c63_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 63 : i32
      %c114_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 114 : i32
      %c16_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 16 : i32
      %c0_i32_1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %c64_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 64 : i32
      %10 = tt.get_program_id x {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
      %11 = arith.addi %arg3, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %12 = arith.divsi %11, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %13 = arith.addi %arg4, %c63_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %14 = arith.divsi %13, %c64_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %15 = arith.muli %12, %14 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
      %16 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %17 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %18 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64x1xi32, #blocked2>
      %19 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked2>
      %c3_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 3 : i32
      %c0_i32_2 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : i32
      %20 = arith.muli %c114_i32, %6 {async_agent = dense<1> : vector<1xi32>} : i32
      %21 = arith.addi %10, %20 {async_agent = dense<1> : vector<1xi32>} : i32
      %c2_i32 = arith.constant {async_agent = dense<1> : vector<1xi32>} 2 : i32
      %22 = arith.muli %c114_i32, %c2_i32 {async_agent = dense<1> : vector<1xi32>} : i32
      %23 = arith.addi %c0_i32_2, %6 {async_agent = dense<1> : vector<1xi32>} : i32
      %24 = scf.for %arg9 = %21 to %15 step %22 iter_args(%arg10 = %23) -> (i32)  : i32 {
        %25 = arith.cmpi ne, %arg9, %10 : i32
        %26 = arith.ori %25, %7 {agent.mutex_role = 0 : i32} : i1
        scf.if %26 {
          triton_nvidia_gpu.lock %8 {agent.mutex_role = 0 : i32} : !triton_nvidia_gpu.mutex
        } {agent.mutex_role = 0 : i32}
        %27 = arith.divsi %arg9, %14 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %28 = arith.remsi %arg9, %14 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %29 = arith.muli %27, %c64_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %30 = tt.splat %29 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
        %31 = arith.addi %30, %17 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
        %32 = arith.muli %28, %c64_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %33 = tt.splat %32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
        %34 = arith.addi %33, %16 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
        %c3_i32_3 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 3 : i32
        %35 = arith.subi %arg5, %c0_i32_1 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %36 = arith.divui %35, %c16_i32 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %37 = arith.muli %arg10, %36 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        %c3_i32_4 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 3 : i32
        %38:2 = scf.for %arg11 = %c0_i32_1 to %arg5 step %c16_i32 iter_args(%arg12 = %cst, %arg13 = %37) -> (tensor<64x64xf32, #mma>, i32)  : i32 {
          %48 = arith.remsi %arg13, %c3_i32_4 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          triton_nvidia_gpu.consumer_wait %2, %48 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %49 = triton_gpu.extract_slice %0[%48, 0, 0] [1, 64, 16] [1, 1, 1] {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x64x16xf16, #shared> to tensor<64x16xf16, #shared>
          %50 = triton_gpu.convert_layout %49 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<64x16xf16, #shared>) -> tensor<64x16xf16, #shared>
          %51 = triton_gpu.extract_slice %1[%48, 0, 0] [1, 16, 64] [1, 1, 1] {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x16x64xf16, #shared1> to tensor<16x64xf16, #shared1>
          %52 = triton_gpu.convert_layout %51 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<16x64xf16, #shared1>) -> tensor<16x64xf16, #shared1>
          %53 = tt.dot %50, %52, %arg12 {agent.mutex_role = 0 : i32, allowTF32 = true, async_agent = dense<1> : vector<1xi32>} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
          triton_nvidia_gpu.consumer_release %2, %48 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<3x!triton_nvidia_gpu.token>, i32
          %c1_i32_6 = arith.constant {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
          %54 = arith.addi %arg13, %c1_i32_6 {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} : i32
          scf.yield {agent.mutex_role = 0 : i32, async_agent = dense<1> : vector<1xi32>} %53, %54 : tensor<64x64xf32, #mma>, i32
        } {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>}
        triton_nvidia_gpu.unlock %8 : !triton_nvidia_gpu.mutex
        scf.if %26 {
          triton_nvidia_gpu.lock %9 {agent.mutex_role = 1 : i32} : !triton_nvidia_gpu.mutex
        } {agent.mutex_role = 1 : i32}
        %39 = tt.expand_dims %31 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
        %40 = arith.muli %39, %18 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64x1xi32, #blocked2>
        %41 = tt.addptr %19, %40 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64x1x!tt.ptr<f32, 1>, #blocked2>, tensor<64x1xi32, #blocked2>
        %42 = tt.expand_dims %34 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2>
        %43 = tt.broadcast %41 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<64x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked2>
        %44 = tt.broadcast %42 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
        %45 = tt.addptr %43, %44 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<64x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x64xi32, #blocked2>
        %46 = triton_gpu.convert_layout %38#0 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked2>
        tt.store %45, %46 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked2>
        triton_nvidia_gpu.unlock %9 : !triton_nvidia_gpu.mutex
        %c1_i32_5 = arith.constant {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} 1 : i32
        %47 = arith.addi %arg10, %c2_i32 {agent.mutex_role = 1 : i32, async_agent = dense<1> : vector<1xi32>} : i32
        scf.yield {async_agent = dense<1> : vector<1xi32>} %47 : i32
      } {async_agent = dense<1> : vector<1xi32>}
    } {"agent.num-roles" = 2 : i32, async_agent = dense<1> : vector<1xi32>}
    tt.return
  }
}
