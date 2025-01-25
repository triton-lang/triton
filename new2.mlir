#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i64 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i64 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %range_1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %range_2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %range_3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %range_4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %splat_1 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %splat_2 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>

    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    %9 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = arith.subi %7, %0 : i32
    %12 = arith.ceildivsi %11, %c132_i32 : i32
    %13 = arith.addi %6, %c0_i32 : i32
    %14 = arith.maxsi %13, %c1_i64 : i32
    %15 = arith.addi %12, %c0_i32 : i32
    %16 = arith.muli %15, %14 : i32
    %17 = arith.subi %0, %c132_i32 : i32
    %18 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    %19 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    %20 = arith.cmpi sgt, %16, %c0_i64 : i32
    %21 = arith.constant 0 : i32
    %22 = arith.cmpi eq, %21, %c0_i64 : i32
    %23 = arith.select %22, %0, %17 : i32
    %24:4 = scf.if %22 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      %105 = arith.divsi %0, %8 : i32
      %106 = arith.muli %105, %c8_i32 : i32
      %107 = arith.subi %2, %106 : i32
      %108 = arith.minsi %107, %c8_i32 : i32
      %109 = arith.remsi %0, %108 : i32
      %110 = arith.addi %106, %109 : i32
      %111 = arith.remsi %0, %8 : i32
      %112 = arith.divsi %111, %108 : i32
      %113 = arith.muli %110, %c128_i32 : i32
      %114 = arith.muli %112, %c256_i32 : i32
      %115 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %116 = tt.splat %113 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %117 = arith.addi %116, %115 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %118 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %119 = tt.splat %114 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %120 = arith.addi %119, %118 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %121 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %122 = arith.cmpi slt, %117, %121 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %123 = arith.select %122, %117, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %124 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %125 = arith.cmpi slt, %120, %124 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %126 = arith.select %125, %120, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %113, %114, %123, %126 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } else {
      scf.yield %c0_i32, %c0_i32, %cst_0, %cst : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    }
    %25 = tt.expand_dims %24#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %26 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %27 = arith.muli %25, %26 : tensor<128x1xi32, #blocked1>
    %28 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %29 = tt.broadcast %27 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %30 = tt.broadcast %28 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %31 = arith.addi %29, %30 : tensor<128x64xi32, #blocked1>
    %32 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %33 = tt.addptr %32, %31 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %34 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %35 = tt.expand_dims %24#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %36 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %37 = arith.muli %35, %36 : tensor<1x256xi32, #blocked>
    %38 = tt.broadcast %34 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %39 = tt.broadcast %37 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %40 = arith.addi %38, %39 : tensor<64x256xi32, #blocked>
    %41 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %42 = tt.addptr %41, %40 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %43 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1>
    %44 = arith.cmpi slt, %28, %43 : tensor<1x64xi32, #blocked1>
    %45 = tt.broadcast %44 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %46 = ttg.memdesc_subview %18[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %47 = tt.splat %20 : i1 -> tensor<128x64xi1, #blocked1>
    %48 = arith.andi %47, %45 : tensor<128x64xi1, #blocked1>
    %49 = ttg.async_copy_global_to_local %33, %46 mask %48 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %50 = ttg.async_commit_group %49
    %51 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked>
    %52 = arith.cmpi slt, %34, %51 : tensor<64x1xi32, #blocked>
    %53 = tt.broadcast %52 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %54 = ttg.memdesc_subview %19[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %55 = tt.splat %20 : i1 -> tensor<64x256xi1, #blocked>
    %56 = arith.andi %55, %53 : tensor<64x256xi1, #blocked>
    %57 = ttg.async_copy_global_to_local %42, %54 mask %56 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %58 = ttg.async_commit_group %57
    %59 = arith.cmpi sgt, %16, %c1_i64 : i32
    %60 = arith.addi %21, %c1_i64 : i32
    %61 = arith.remsi %60, %14 : i32
    %62 = arith.cmpi eq, %61, %c0_i64 : i32
    %63 = arith.cmpi ne, %61, %c0_i64 : i32
    %64 = arith.extui %63 : i1 to i32
    %65:5 = scf.if %62 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
      %105 = arith.addi %23, %c132_i32 : i32
      %106 = arith.divsi %105, %8 : i32
      %107 = arith.muli %106, %c8_i32 : i32
      %108 = arith.subi %2, %107 : i32
      %109 = arith.minsi %108, %c8_i32 : i32
      %110 = arith.remsi %105, %109 : i32
      %111 = arith.addi %107, %110 : i32
      %112 = arith.remsi %105, %8 : i32
      %113 = arith.divsi %112, %109 : i32
      %114 = arith.muli %111, %c128_i32 : i32
      %115 = arith.muli %113, %c256_i32 : i32
      %116 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %117 = tt.splat %114 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %118 = arith.addi %117, %116 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %119 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %120 = tt.splat %115 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %121 = arith.addi %120, %119 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %122 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %123 = arith.cmpi slt, %118, %122 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %124 = arith.select %123, %118, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %125 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %126 = arith.cmpi slt, %121, %125 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %127 = arith.select %126, %121, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      scf.yield %114, %115, %124, %127, %105 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    } else {
      scf.yield %24#0, %24#1, %24#2, %24#3, %23 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
    }
    %66 = arith.muli %64, %c64_i32 : i32
    %67 = tt.splat %66 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %68 = tt.splat %66 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %69 = arith.addi %67, %9 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %70 = arith.addi %68, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %71 = tt.expand_dims %65#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %72 = arith.muli %71, %26 : tensor<128x1xi32, #blocked1>
    %73 = tt.expand_dims %69 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %74 = tt.broadcast %72 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %75 = tt.broadcast %73 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
    %76 = arith.addi %74, %75 : tensor<128x64xi32, #blocked1>
    %77 = tt.addptr %32, %76 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
    %78 = tt.expand_dims %70 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %79 = tt.expand_dims %65#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %80 = arith.muli %79, %36 : tensor<1x256xi32, #blocked>
    %81 = tt.broadcast %78 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %82 = tt.broadcast %80 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
    %83 = arith.addi %81, %82 : tensor<64x256xi32, #blocked>
    %84 = tt.addptr %41, %83 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
    %85 = arith.subi %arg5, %66 : i32
    %86 = tt.splat %85 : i32 -> tensor<1x64xi32, #blocked1>
    %87 = arith.cmpi slt, %28, %86 : tensor<1x64xi32, #blocked1>
    %88 = tt.broadcast %87 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
    %89 = ttg.memdesc_subview %18[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %90 = tt.splat %59 : i1 -> tensor<128x64xi1, #blocked1>
    %91 = arith.andi %90, %88 : tensor<128x64xi1, #blocked1>
    %92 = ttg.async_copy_global_to_local %77, %89 mask %91 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
    %93 = ttg.async_commit_group %92
    %94 = tt.splat %85 : i32 -> tensor<64x1xi32, #blocked>
    %95 = arith.cmpi slt, %34, %94 : tensor<64x1xi32, #blocked>
    %96 = tt.broadcast %95 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
    %97 = ttg.memdesc_subview %19[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %98 = tt.splat %59 : i1 -> tensor<64x256xi1, #blocked>
    %99 = arith.andi %98, %96 : tensor<64x256xi1, #blocked>
    %100 = ttg.async_copy_global_to_local %84, %97 mask %99 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
    %101 = ttg.async_commit_group %100
    %102:18 = scf.for %arg9 = %c0_i64 to %16 step %c1_i64 iter_args(%arg10 = %61, %arg11 = %65#4, %arg12 = %cst_3, %arg13 = %65#0, %arg14 = %65#1, %arg15 = %65#2, %arg16 = %65#3, %arg17 = %c1_i32, %arg18 = %c-1_i32, %arg19 = %64, %arg20 = %21, %arg21 = %61, %arg22 = %58, %arg23 = %101, %arg24 = %24#0, %arg25 = %65#0, %arg26 = %24#1, %arg27 = %65#1) -> (i32, i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32)  : i32 {
      %105 = arith.subi %16, %c2_i64 : i32
      %106 = arith.cmpi slt, %arg9, %105 : i32
      %107 = arith.addi %arg19, %c1_i32 : i32
      %108 = arith.addi %107, %c0_i32 : i32
      %109 = arith.remsi %108, %14 : i32
      %110 = arith.cmpi eq, %109, %c0_i64 : i32
      %111 = arith.select %110, %c0_i32, %107 : i32
      %112:5 = scf.if %110 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
        %163 = arith.addi %arg11, %c132_i32 : i32
        %164 = arith.divsi %163, %8 : i32
        %165 = arith.muli %164, %c8_i32 : i32
        %166 = arith.subi %2, %165 : i32
        %167 = arith.minsi %166, %c8_i32 : i32
        %168 = arith.remsi %163, %167 : i32
        %169 = arith.addi %165, %168 : i32
        %170 = arith.remsi %163, %8 : i32
        %171 = arith.divsi %170, %167 : i32
        %172 = arith.muli %169, %c128_i32 : i32
        %173 = arith.muli %171, %c256_i32 : i32
        %175 = tt.splat %172 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %176 = arith.addi %175, %range_3 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %178 = tt.splat %173 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %179 = arith.addi %178, %range_4 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %181 = arith.cmpi slt, %176, %splat_1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %182 = arith.select %181, %176, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %184 = arith.cmpi slt, %179, %splat_2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %185 = arith.select %184, %179, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        scf.yield %172, %173, %182, %185, %163 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      } else {
        scf.yield %arg13, %arg14, %arg15, %arg16, %arg11 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32
      }
      %113 = arith.addi %arg18, %c1_i32 : i32
      %114 = arith.cmpi slt, %113, %c3_i32 : i32
      %115 = arith.select %114, %113, %c0_i32 : i32
      %116 = arith.cmpi ne, %arg20, %c0_i64 : i32
      %117 = ttg.memdesc_subview %18[%115, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %118 = ttg.async_wait %arg22 {num = 2 : i32}
      %119 = ttg.memdesc_subview %19[%115, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %120 = ttng.warp_group_dot %117, %119, %arg12, %116 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf32, #mma>
      %121:3 = ttng.warp_group_dot_wait %120, %117, %119 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %122 = arith.addi %arg17, %c1_i32 : i32
      %123 = arith.cmpi slt, %122, %c3_i32 : i32
      %124 = arith.select %123, %122, %c0_i32 : i32
      %125 = arith.muli %111, %c64_i32 : i32
      %126 = tt.splat %125 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %127 = tt.splat %125 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %128 = arith.addi %126, %9 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %129 = arith.addi %127, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %130 = tt.expand_dims %112#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %131 = arith.muli %130, %26 : tensor<128x1xi32, #blocked1>
      %132 = tt.expand_dims %128 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
      %133 = tt.broadcast %131 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %134 = tt.broadcast %132 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %135 = arith.addi %133, %134 : tensor<128x64xi32, #blocked1>
      %136 = tt.addptr %32, %135 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
      %137 = tt.expand_dims %129 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %138 = tt.expand_dims %112#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %139 = arith.muli %138, %36 : tensor<1x256xi32, #blocked>
      %140 = tt.broadcast %137 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %141 = tt.broadcast %139 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %142 = arith.addi %140, %141 : tensor<64x256xi32, #blocked>
      %143 = tt.addptr %41, %142 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
      %144 = arith.subi %arg5, %125 : i32
      %145 = tt.splat %144 : i32 -> tensor<1x64xi32, #blocked1>
      %146 = arith.cmpi slt, %28, %145 : tensor<1x64xi32, #blocked1>
      %147 = tt.broadcast %146 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
      %148 = ttg.memdesc_subview %18[%124, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
      %149 = tt.splat %106 : i1 -> tensor<128x64xi1, #blocked1>
      %150 = arith.andi %149, %147 : tensor<128x64xi1, #blocked1>
      %151 = ttg.async_copy_global_to_local %136, %148 mask %150 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable>
      %152 = ttg.async_commit_group %151
      %153 = tt.splat %144 : i32 -> tensor<64x1xi32, #blocked>
      %154 = arith.cmpi slt, %34, %153 : tensor<64x1xi32, #blocked>
      %155 = tt.broadcast %154 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
      %156 = ttg.memdesc_subview %19[%124, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      %157 = tt.splat %106 : i1 -> tensor<64x256xi1, #blocked>
      %158 = arith.andi %157, %155 : tensor<64x256xi1, #blocked>
      %159 = ttg.async_copy_global_to_local %143, %156 mask %158 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable>
      %160 = ttg.async_commit_group %159
      %161 = arith.subi %14, %c1_i64 : i32
      %162 = arith.cmpi eq, %arg20, %161 : i32
      scf.if %162 {
        %163:3 = ttng.warp_group_dot_wait %121#0, %117, %119 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        %165 = tt.splat %arg24 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %166 = arith.addi %165, %range_1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %168 = tt.splat %arg26 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %169 = arith.addi %168, %range_2 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %170 = tt.expand_dims %166 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %171 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
        %172 = arith.muli %171, %170 : tensor<128x1xi32, #blocked2>
        %173 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
        %174 = tt.addptr %173, %172 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %175 = tt.expand_dims %169 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %176 = tt.broadcast %174 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %177 = tt.broadcast %175 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %178 = tt.addptr %176, %177 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %179 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
        %180 = arith.cmpi slt, %170, %179 : tensor<128x1xi32, #blocked2>
        %181 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
        %182 = arith.cmpi slt, %175, %181 : tensor<1x256xi32, #blocked2>
        %183 = tt.broadcast %180 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %184 = tt.broadcast %182 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %185 = arith.andi %183, %184 : tensor<128x256xi1, #blocked2>
        %186 = arith.truncf %163#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %187 = ttg.convert_layout %186 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %178, %187, %185 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
      scf.yield %109, %112#4, %121#0, %112#0, %112#1, %112#2, %112#3, %124, %115, %111, %arg21, %109, %arg23, %160, %arg25, %112#0, %arg27, %112#1 : i32, i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, i32, i32, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32
    }
    %103 = ttng.warp_group_dot_wait %102#2 {pendings = 0 : i32} : tensor<128x256xf32, #mma>
    %104 = ttg.async_wait  {num = 0 : i32}
    ttg.local_dealloc %18 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %19 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}

