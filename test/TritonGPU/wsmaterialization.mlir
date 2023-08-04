// RUN: triton-opt -triton-nvidia-gpu-ws-materialization='compute-capability=90' %s | FileCheck %s

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

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32} {
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
      triton_nvidia_gpu.lock %3 : !triton_nvidia_gpu.mutex
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
      triton_nvidia_gpu.unlock %3 : !triton_nvidia_gpu.mutex
      triton_nvidia_gpu.lock %4 : !triton_nvidia_gpu.mutex
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
      triton_nvidia_gpu.unlock %4 : !triton_nvidia_gpu.mutex
    }
    tt.return
  }
}
