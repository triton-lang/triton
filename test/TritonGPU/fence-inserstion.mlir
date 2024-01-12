// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_like_fence_1(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %4 = arith.extsi %arg4 : i32 to i64
    %5 = arith.extsi %arg7 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%1, %4], [%5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %7 = arith.extsi %arg8 : i32 to i64
    %8 = tt.make_tensor_ptr %arg2, [%0, %4], [%7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %9 = triton_nvidia_gpu.alloc_mbarrier {count = 1 : i32} : tensor<3xi64, #shared>
    %10 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %11 = triton_gpu.alloc_tensor : tensor<3x128x128xf16, #shared1>
    %12 = tt.splat %10 : (i1) -> tensor<128x128xi1, #blocked1>
    %13 = triton_nvidia_gpu.extract_mbarrier %9[%c0_i32] : tensor<3xi64, #shared>, i32 -> <i64, 3>
    %14 = triton_nvidia_gpu.get_thread_id : i32
    %15 = arith.cmpi eq, %14, %c0_i32 : i32
    %16 = arith.andi %15, %10 : i1
    triton_nvidia_gpu.mbarrier_arrive %13, %16 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
    %17 = triton_nvidia_gpu.insert_slice_tma %3, %11, %c0_i32, %13, %12 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %18 = triton_gpu.alloc_tensor : tensor<3x128x128xf16, #shared1>
    %19 = triton_nvidia_gpu.insert_slice_tma %6, %18, %c0_i32, %13, %12 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %20 = tt.advance %3, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
    %21 = tt.advance %6, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
    %22 = arith.cmpi sgt, %arg5, %c128_i32 : i32
    %23 = tt.splat %22 : (i1) -> tensor<128x128xi1, #blocked1>
    %24 = triton_nvidia_gpu.extract_mbarrier %9[%c1_i32] : tensor<3xi64, #shared>, i32 -> <i64, 3>
    %25 = arith.andi %15, %22 : i1
    triton_nvidia_gpu.mbarrier_arrive %24, %25 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
    %26 = triton_nvidia_gpu.insert_slice_tma %20, %17, %c1_i32, %24, %23 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %27 = triton_nvidia_gpu.insert_slice_tma %21, %19, %c1_i32, %24, %23 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %28 = triton_gpu.extract_slice %26[0, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
    %29 = triton_gpu.extract_slice %27[0, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
    %30:15 = scf.for %arg9 = %c0_i32 to %arg5 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %3, %arg12 = %6, %arg13 = %26, %arg14 = %27, %arg15 = %28, %arg16 = %29, %arg17 = %20, %arg18 = %21, %arg19 = %c128_i32, %arg20 = %c2_i32, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %false, %arg24 = %true) -> (tensor<128x128xf32, #mma>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, tensor<3x128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, i32, i32, i32, i32, i1, i1)  : i32 {
      %33 = triton_nvidia_gpu.extract_mbarrier %9[%arg21] : tensor<3xi64, #shared>, i32 -> <i64, 3>
      triton_nvidia_gpu.mbarrier_wait %33, %arg23 : <i64, 3>
      // CHECK: triton_nvidia_gpu.fence_async_shared
      %34 = triton_nvidia_gpu.dot_async %arg15, %arg16, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #shared1> * tensor<128x128xf16, #shared1> -> tensor<128x128xf32, #mma>
      %35 = tt.advance %arg11, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
      %36 = tt.advance %arg12, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
      %37 = arith.addi %arg19, %c128_i32 : i32
      %38 = arith.cmpi slt, %37, %arg5 : i32
      %39 = arith.addi %arg21, %c1_i32 : i32
      %40 = arith.cmpi uge, %39, %c3_i32 : i32
      %41 = arith.select %40, %c0_i32, %39 : i32
      %42 = tt.advance %arg17, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
      %43 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
      %44 = tt.splat %38 : (i1) -> tensor<128x128xi1, #blocked1>
      %45 = triton_nvidia_gpu.extract_mbarrier %9[%arg20] : tensor<3xi64, #shared>, i32 -> <i64, 3>
      %46 = arith.andi %15, %38 : i1
      triton_nvidia_gpu.mbarrier_arrive %45, %46 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
      %47 = triton_nvidia_gpu.insert_slice_tma %42, %arg13, %arg20, %45, %44 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
      %48 = triton_gpu.extract_slice %47[%41, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
      %49 = triton_nvidia_gpu.insert_slice_tma %43, %arg14, %arg20, %45, %44 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
      %50 = triton_gpu.extract_slice %49[%41, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
      %b_48 = triton_gpu.convert_layout %48 : (tensor<128x128xf16, #shared1>) -> tensor<128x128xf16, #blocked1>
      %s_48 = triton_gpu.convert_layout %b_48 : (tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #shared1>
      %51 = arith.addi %arg20, %c1_i32 : i32
      %52 = arith.cmpi uge, %51, %c3_i32 : i32
      %53 = arith.select %52, %c0_i32, %51 : i32
      %54 = arith.addi %arg22, %c1_i32 : i32
      %55 = arith.xori %arg23, %true : i1
      %56 = arith.cmpi ult, %39, %c3_i32 : i32
      %57 = arith.andi %40, %55 : i1
      %58 = arith.andi %56, %arg23 : i1
      %59 = arith.ori %57, %58 : i1
      %60 = arith.xori %arg24, %true : i1
      %61 = arith.cmpi ult, %51, %c3_i32 : i32
      %62 = arith.andi %52, %60 : i1
      %63 = arith.andi %61, %arg24 : i1
      %64 = arith.ori %62, %63 : i1
      scf.yield %34, %35, %36, %47, %49, %s_48, %50, %42, %43, %37, %53, %41, %54, %59, %64 : tensor<128x128xf32, #mma>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, tensor<3x128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, i32, i32, i32, i32, i1, i1
    }
    %w = triton_nvidia_gpu.dot_wait %30#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
    %31 = arith.truncf %w : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %32 = triton_gpu.convert_layout %31 : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #shared1>
    triton_nvidia_gpu.store_async_tma %8, %32 : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<128x128xf16, #shared1>
    triton_gpu.async_bulk_commit_group
    triton_gpu.async_bulk_wait {num = 0 : i32}
    tt.return
  }
}


// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_like_fence_2(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %4 = arith.extsi %arg4 : i32 to i64
    %5 = arith.extsi %arg7 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%1, %4], [%5, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %7 = arith.extsi %arg8 : i32 to i64
    %8 = tt.make_tensor_ptr %arg2, [%0, %4], [%7, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %9 = triton_nvidia_gpu.alloc_mbarrier {count = 1 : i32} : tensor<3xi64, #shared>
    %10 = arith.cmpi sgt, %arg5, %c0_i32 : i32
    %11 = triton_gpu.alloc_tensor : tensor<3x128x128xf16, #shared1>
    %12 = tt.splat %10 : (i1) -> tensor<128x128xi1, #blocked1>
    %13 = triton_nvidia_gpu.extract_mbarrier %9[%c0_i32] : tensor<3xi64, #shared>, i32 -> <i64, 3>
    %14 = triton_nvidia_gpu.get_thread_id : i32
    %15 = arith.cmpi eq, %14, %c0_i32 : i32
    %16 = arith.andi %15, %10 : i1
    triton_nvidia_gpu.mbarrier_arrive %13, %16 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
    %17 = triton_nvidia_gpu.insert_slice_tma %3, %11, %c0_i32, %13, %12 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %18 = triton_gpu.alloc_tensor : tensor<3x128x128xf16, #shared1>
    %19 = triton_nvidia_gpu.insert_slice_tma %6, %18, %c0_i32, %13, %12 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %20 = tt.advance %3, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
    %21 = tt.advance %6, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
    %22 = arith.cmpi sgt, %arg5, %c128_i32 : i32
    %23 = tt.splat %22 : (i1) -> tensor<128x128xi1, #blocked1>
    %24 = triton_nvidia_gpu.extract_mbarrier %9[%c1_i32] : tensor<3xi64, #shared>, i32 -> <i64, 3>
    %25 = arith.andi %15, %22 : i1
    triton_nvidia_gpu.mbarrier_arrive %24, %25 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
    %26 = triton_nvidia_gpu.insert_slice_tma %20, %17, %c1_i32, %24, %23 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %27 = triton_nvidia_gpu.insert_slice_tma %21, %19, %c1_i32, %24, %23 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
    %28 = triton_gpu.extract_slice %26[0, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
    %29 = triton_gpu.extract_slice %27[0, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
    %b_29 = triton_gpu.convert_layout %29 : (tensor<128x128xf16, #shared1>) -> tensor<128x128xf16, #blocked1>
    %s_29 = triton_gpu.convert_layout %b_29 : (tensor<128x128xf16, #blocked1>) -> tensor<128x128xf16, #shared1>
    %30:15 = scf.for %arg9 = %c0_i32 to %arg5 step %c128_i32 iter_args(%arg10 = %cst, %arg11 = %3, %arg12 = %6, %arg13 = %26, %arg14 = %27, %arg15 = %28, %arg16 = %s_29, %arg17 = %20, %arg18 = %21, %arg19 = %c128_i32, %arg20 = %c2_i32, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %false, %arg24 = %true) -> (tensor<128x128xf32, #mma>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, tensor<3x128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, i32, i32, i32, i32, i1, i1)  : i32 {
      %33 = triton_nvidia_gpu.extract_mbarrier %9[%arg21] : tensor<3xi64, #shared>, i32 -> <i64, 3>
      triton_nvidia_gpu.mbarrier_wait %33, %arg23 : <i64, 3>
      // CHECK: triton_nvidia_gpu.fence_async_shared
      %34 = triton_nvidia_gpu.dot_async %arg15, %arg16, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #shared1> * tensor<128x128xf16, #shared1> -> tensor<128x128xf32, #mma>
      %35 = tt.advance %arg11, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
      %36 = tt.advance %arg12, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
      %37 = arith.addi %arg19, %c128_i32 : i32
      %38 = arith.cmpi slt, %37, %arg5 : i32
      %39 = arith.addi %arg21, %c1_i32 : i32
      %40 = arith.cmpi uge, %39, %c3_i32 : i32
      %41 = arith.select %40, %c0_i32, %39 : i32
      %42 = tt.advance %arg17, [%c0_i32, %c128_i32] : <tensor<128x128xf16, #blocked>, 1>
      %43 = tt.advance %arg18, [%c128_i32, %c0_i32] : <tensor<128x128xf16, #blocked>, 1>
      %44 = tt.splat %38 : (i1) -> tensor<128x128xi1, #blocked1>
      %45 = triton_nvidia_gpu.extract_mbarrier %9[%arg20] : tensor<3xi64, #shared>, i32 -> <i64, 3>
      %46 = arith.andi %15, %38 : i1
      triton_nvidia_gpu.mbarrier_arrive %45, %46 {operandSegmentSizes = array<i32: 1, 1, 0>, trackAsyncOp = false, txCount = 65536 : i32} : !tt.ptr<i64, 3>, i1
      %47 = triton_nvidia_gpu.insert_slice_tma %42, %arg13, %arg20, %45, %44 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
      %48 = triton_gpu.extract_slice %47[%41, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
      %49 = triton_nvidia_gpu.insert_slice_tma %43, %arg14, %arg20, %45, %44 {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0>} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, i32, !tt.ptr<i64, 3>, tensor<128x128xi1, #blocked1> -> tensor<3x128x128xf16, #shared1>
      %50 = triton_gpu.extract_slice %49[%41, 0, 0] [1, 128, 128] [1, 1, 1] : tensor<3x128x128xf16, #shared1> to tensor<128x128xf16, #shared1>
      %51 = arith.addi %arg20, %c1_i32 : i32
      %52 = arith.cmpi uge, %51, %c3_i32 : i32
      %53 = arith.select %52, %c0_i32, %51 : i32
      %54 = arith.addi %arg22, %c1_i32 : i32
      %55 = arith.xori %arg23, %true : i1
      %56 = arith.cmpi ult, %39, %c3_i32 : i32
      %57 = arith.andi %40, %55 : i1
      %58 = arith.andi %56, %arg23 : i1
      %59 = arith.ori %57, %58 : i1
      %60 = arith.xori %arg24, %true : i1
      %61 = arith.cmpi ult, %51, %c3_i32 : i32
      %62 = arith.andi %52, %60 : i1
      %63 = arith.andi %61, %arg24 : i1
      %64 = arith.ori %62, %63 : i1
      scf.yield %34, %35, %36, %47, %49, %48, %50, %42, %43, %37, %53, %41, %54, %59, %64 : tensor<128x128xf32, #mma>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<3x128x128xf16, #shared1>, tensor<3x128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, tensor<128x128xf16, #shared1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, !tt.ptr<tensor<128x128xf16, #blocked>, 1>, i32, i32, i32, i32, i1, i1
    }
    %w = triton_nvidia_gpu.dot_wait %30#0 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
    %31 = arith.truncf %w : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %32 = triton_gpu.convert_layout %31 : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #shared1>
    triton_nvidia_gpu.store_async_tma %8, %32 : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<128x128xf16, #shared1>
    triton_gpu.async_bulk_commit_group
    triton_gpu.async_bulk_wait {num = 0 : i32}
    tt.return
  }
}
