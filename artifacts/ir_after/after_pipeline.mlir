#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @gemm_scatter_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %c2 = arith.constant 2 : index
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #blocked1>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    %13 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %14 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %15 = ttg.async_copy_global_to_local %4, %13 mask %14 other %cst_4 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
    %16 = ttg.async_commit_group tokens %15
    %17 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %18 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %19 = ttg.async_copy_global_to_local %9, %17 mask %18 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %20 = ttg.async_commit_group tokens %19
    %21 = arith.addi %arg0, %arg2 : index
    %22 = arith.cmpi slt, %21, %arg1 : index
    %23 = tt.addptr %4, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %24 = tt.addptr %9, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %25 = ttg.memdesc_index %10[%c1_i32] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %26 = tt.splat %22 : i1 -> tensor<128x32xi1, #blocked1>
    %27 = ttg.async_copy_global_to_local %23, %25 mask %26 other %cst_4 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
    %28 = ttg.async_commit_group tokens %27
    %29 = ttg.memdesc_index %11[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    %30 = tt.splat %22 : i1 -> tensor<32x128xi1, #blocked>
    %31 = ttg.async_copy_global_to_local %24, %29 mask %30 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
    %32 = ttg.async_commit_group tokens %31
    %33:9 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %23, %arg8 = %24, %arg9 = %cst_2, %arg10 = %c1_i32, %arg11 = %c-1_i32, %arg12 = %16, %arg13 = %28, %arg14 = %20, %arg15 = %32) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token) {
      %37 = arith.muli %arg2, %c2 : index
      %38 = arith.subi %arg1, %37 : index
      %39 = arith.cmpi slt, %arg6, %38 : index
      %40 = arith.addi %arg11, %c1_i32 : i32
      %41 = arith.cmpi sge, %40, %c2_i32 : i32
      %42 = arith.select %41, %c0_i32, %40 : i32
      %43 = ttg.async_wait %arg12, %arg14 {num = 2 : i32}
      %44 = ttg.memdesc_index %10[%42] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %45 = ttg.local_load %44 token %43 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %46 = ttg.memdesc_index %11[%42] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %47 = ttg.local_load %46 token %43 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %48 = arith.mulf %47, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %49 = tt.dot %45, %48, %arg9 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %50 = tt.addptr %arg7, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %51 = tt.addptr %arg8, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      %52 = arith.addi %arg10, %c1_i32 : i32
      %53 = arith.cmpi sge, %52, %c2_i32 : i32
      %54 = arith.select %53, %c0_i32, %52 : i32
      %55 = ttg.memdesc_index %10[%54] : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %56 = tt.splat %39 : i1 -> tensor<128x32xi1, #blocked1>
      %57 = ttg.async_copy_global_to_local %50, %55 mask %56 other %cst_4 {contiguity = 4 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1> -> <128x32xf16, #shared, #smem, mutable>
      %58 = ttg.async_commit_group tokens %57
      %59 = ttg.memdesc_index %11[%54] : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      %60 = tt.splat %39 : i1 -> tensor<32x128xi1, #blocked>
      %61 = ttg.async_copy_global_to_local %51, %59 mask %60 other %cst_3 {contiguity = 4 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked> -> <32x128xf16, #shared1, #smem, mutable>
      %62 = ttg.async_commit_group tokens %61
      scf.yield %50, %51, %49, %54, %42, %arg13, %58, %arg15, %62 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    %34 = ttg.async_wait {num = 0 : i32}
    ttg.local_dealloc %11 : !ttg.memdesc<2x32x128xf16, #shared1, #smem, mutable>
    ttg.local_dealloc %10 : !ttg.memdesc<2x128x32xf16, #shared, #smem, mutable>
    %35 = ttg.convert_layout %33#2 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
    %36 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.store %36, %35 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return %33#2 : tensor<128x128xf32, #mma>
  }
}

