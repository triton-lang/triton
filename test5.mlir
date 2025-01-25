#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c2_i64 = arith.constant 2 : i64
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #blocked>
    %0 = ub.poison : tensor<64x256xi32, #blocked1>
    %1 = ub.poison : tensor<128x64xi32, #blocked2>
    %2 = ub.poison : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = ub.poison : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = ub.poison : tensor<128x256xf32, #mma>
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked2>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked1>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %5 = tt.get_program_id x : i32
    %6 = arith.addi %arg3, %c127_i32 : i32
    %7 = arith.divsi %6, %c128_i32 : i32
    %8 = arith.addi %arg4, %c255_i32 : i32
    %9 = arith.divsi %8, %c256_i32 : i32
    %10 = arith.addi %arg5, %c63_i32 : i32
    %11 = arith.divsi %10, %c64_i32 : i32
    %12 = arith.muli %7, %9 : i32
    %13 = arith.muli %9, %c8_i32 : i32
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %20 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %21 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2>
    %23 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
    %24 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked1>
    %25 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %26 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %27 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %28 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %29 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %30 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
    %31 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked>
    %32 = arith.cmpi eq, %11, %c0_i32 : i32
    scf.if %32 {
      scf.for %arg9 = %5 to %12 step %c132_i32  : i32 {
        %33 = arith.divsi %arg9, %13 : i32
        %34 = arith.muli %33, %c8_i32 : i32
        %35 = arith.subi %7, %34 : i32
        %36 = arith.minsi %35, %c8_i32 : i32
        %37 = arith.remsi %arg9, %36 : i32
        %38 = arith.addi %34, %37 : i32
        %39 = arith.remsi %arg9, %13 : i32
        %40 = arith.divsi %39, %36 : i32
        %41 = arith.muli %38, %c128_i32 : i32
        %42 = arith.muli %40, %c256_i32 : i32
        %43 = tt.splat %41 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %44 = arith.addi %43, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %45 = tt.splat %42 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %46 = arith.addi %45, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %47 = tt.expand_dims %44 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
        %48 = arith.muli %28, %47 : tensor<128x1xi32, #blocked>
        %49 = tt.addptr %29, %48 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
        %50 = tt.expand_dims %46 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
        %51 = tt.broadcast %49 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
        %52 = tt.broadcast %50 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
        %53 = tt.addptr %51, %52 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
        %54 = arith.cmpi slt, %47, %30 : tensor<128x1xi32, #blocked>
        %55 = arith.cmpi slt, %50, %31 : tensor<1x256xi32, #blocked>
        %56 = tt.broadcast %54 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %57 = tt.broadcast %55 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %58 = arith.andi %56, %57 : tensor<128x256xi1, #blocked>
        tt.store %53, %cst, %58 : tensor<128x256x!tt.ptr<f16>, #blocked>
      }
    } else {
      %33 = arith.subi %12, %5 : i32
      %34 = arith.ceildivsi %33, %c132_i32 : i32
      %35 = arith.extsi %11 : i32 to i64
      %36 = arith.maxsi %35, %c1_i64 : i64
      %37 = arith.extsi %34 : i32 to i64
      %38 = arith.muli %37, %36 : i64
      %39 = arith.subi %5, %c132_i32 : i32
      %40 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
      %41 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
      %42 = arith.cmpi sgt, %38, %c0_i64 : i64
      %43 = arith.remsi %c0_i64, %36 : i64
      %44 = arith.cmpi eq, %43, %c0_i64 : i64
      %45:5 = scf.if %44 -> (tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32) {
        %108 = arith.divsi %5, %13 : i32
        %109 = arith.muli %108, %c8_i32 : i32
        %110 = arith.subi %7, %109 : i32
        %111 = arith.minsi %110, %c8_i32 : i32
        %112 = arith.remsi %5, %111 : i32
        %113 = arith.addi %109, %112 : i32
        %114 = arith.remsi %5, %13 : i32
        %115 = arith.divsi %114, %111 : i32
        %116 = arith.muli %113, %c128_i32 : i32
        %117 = arith.muli %115, %c256_i32 : i32
        %118 = tt.splat %116 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %119 = tt.splat %116 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %120 = arith.addi %118, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %121 = arith.addi %119, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %122 = tt.splat %117 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %123 = tt.splat %117 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %124 = arith.addi %122, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %125 = arith.addi %123, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %126 = arith.cmpi slt, %120, %20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %127 = arith.select %126, %120, %cst_1 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %128 = arith.cmpi slt, %124, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %129 = arith.select %128, %124, %cst_0 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %130 = tt.expand_dims %127 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %131 = arith.muli %130, %22 : tensor<128x1xi32, #blocked2>
        %132 = tt.broadcast %131 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
        %133 = tt.expand_dims %129 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
        %134 = arith.muli %133, %24 : tensor<1x256xi32, #blocked1>
        %135 = tt.broadcast %134 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
        scf.yield %121, %125, %132, %135, %5 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
      } else {
        scf.yield %3, %2, %1, %0, %39 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
      }
      %46 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
      %47 = tt.broadcast %46 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
      %48 = arith.addi %45#2, %47 : tensor<128x64xi32, #blocked2>
      %49 = tt.addptr %23, %48 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
      %50 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %51 = tt.broadcast %50 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
      %52 = arith.addi %51, %45#3 : tensor<64x256xi32, #blocked1>
      %53 = tt.addptr %25, %52 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %54 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked2>
      %55 = arith.cmpi slt, %26, %54 : tensor<1x64xi32, #blocked2>
      %56 = tt.broadcast %55 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
      %57 = ttg.memdesc_subview %40[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      %58 = tt.splat %42 : i1 -> tensor<128x64xi1, #blocked2>
      %59 = arith.andi %58, %56 : tensor<128x64xi1, #blocked2>
      %60 = ttg.async_copy_global_to_local %49, %57 mask %59 other %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> <128x64xf16, #shared, #smem, mutable, 3x128x64>
      %61 = ttg.async_commit_group %60
      %62 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked1>
      %63 = arith.cmpi slt, %27, %62 : tensor<64x1xi32, #blocked1>
      %64 = tt.broadcast %63 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
      %65 = ttg.memdesc_subview %41[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %66 = tt.splat %42 : i1 -> tensor<64x256xi1, #blocked1>
      %67 = arith.andi %66, %64 : tensor<64x256xi1, #blocked1>
      %68 = ttg.async_copy_global_to_local %53, %65 mask %67 other %cst_3 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %69 = ttg.async_commit_group %68
      %70 = arith.cmpi sgt, %38, %c1_i64 : i64
      %71 = arith.addi %43, %c1_i64 : i64
      %72 = arith.remsi %71, %36 : i64
      %73 = arith.cmpi eq, %72, %c0_i64 : i64
      %74:5 = scf.if %73 -> (tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32) {
        %108 = arith.addi %45#4, %c132_i32 : i32
        %109 = arith.divsi %108, %13 : i32
        %110 = arith.muli %109, %c8_i32 : i32
        %111 = arith.subi %7, %110 : i32
        %112 = arith.minsi %111, %c8_i32 : i32
        %113 = arith.remsi %108, %112 : i32
        %114 = arith.addi %110, %113 : i32
        %115 = arith.remsi %108, %13 : i32
        %116 = arith.divsi %115, %112 : i32
        %117 = arith.muli %114, %c128_i32 : i32
        %118 = arith.muli %116, %c256_i32 : i32
        %119 = tt.splat %117 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %120 = tt.splat %117 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %121 = arith.addi %119, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %122 = arith.addi %120, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %123 = tt.splat %118 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %124 = tt.splat %118 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %125 = arith.addi %123, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %126 = arith.addi %124, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %127 = arith.cmpi slt, %121, %20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %128 = arith.select %127, %121, %cst_1 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %129 = arith.cmpi slt, %125, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %130 = arith.select %129, %125, %cst_0 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %131 = tt.expand_dims %128 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %132 = arith.muli %131, %22 : tensor<128x1xi32, #blocked2>
        %133 = tt.broadcast %132 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
        %134 = tt.expand_dims %130 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
        %135 = arith.muli %134, %24 : tensor<1x256xi32, #blocked1>
        %136 = tt.broadcast %135 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
        scf.yield %122, %126, %133, %136, %108 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
      } else {
        scf.yield %45#0, %45#1, %45#2, %45#3, %45#4 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
      }
      %75 = arith.select %73, %c0_i32, %c1_i32 : i32
      %76 = arith.muli %75, %c64_i32 : i32
      %77 = tt.splat %76 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %78 = tt.splat %76 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %79 = arith.addi %77, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %80 = arith.addi %78, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %81 = tt.expand_dims %79 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
      %82 = tt.broadcast %81 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
      %83 = arith.addi %74#2, %82 : tensor<128x64xi32, #blocked2>
      %84 = tt.addptr %23, %83 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
      %85 = tt.expand_dims %80 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %86 = tt.broadcast %85 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
      %87 = arith.addi %86, %74#3 : tensor<64x256xi32, #blocked1>
      %88 = tt.addptr %25, %87 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %89 = arith.subi %arg5, %76 : i32
      %90 = tt.splat %89 : i32 -> tensor<1x64xi32, #blocked2>
      %91 = arith.cmpi slt, %26, %90 : tensor<1x64xi32, #blocked2>
      %92 = tt.broadcast %91 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
      %93 = ttg.memdesc_subview %40[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
      %94 = tt.splat %70 : i1 -> tensor<128x64xi1, #blocked2>
      %95 = arith.andi %94, %92 : tensor<128x64xi1, #blocked2>
      %96 = ttg.async_copy_global_to_local %84, %93 mask %95 other %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> <128x64xf16, #shared, #smem, mutable, 3x128x64>
      %97 = ttg.async_commit_group %96
      %98 = tt.splat %89 : i32 -> tensor<64x1xi32, #blocked1>
      %99 = arith.cmpi slt, %27, %98 : tensor<64x1xi32, #blocked1>
      %100 = tt.broadcast %99 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
      %101 = ttg.memdesc_subview %41[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %102 = tt.splat %70 : i1 -> tensor<64x256xi1, #blocked1>
      %103 = arith.andi %102, %100 : tensor<64x256xi1, #blocked1>
      %104 = ttg.async_copy_global_to_local %88, %101 mask %103 other %cst_3 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256>
      %105 = ttg.async_commit_group %104
      %106:23 = scf.for %arg9 = %c0_i64 to %38 step %c1_i64 iter_args(%arg10 = %72, %arg11 = %74#4, %arg12 = %c1_i32, %arg13 = %4, %arg14 = %74#0, %arg15 = %74#1, %arg16 = %74#2, %arg17 = %74#3, %arg18 = %c1_i32, %arg19 = %c-1_i32, %arg20 = %44, %arg21 = %73, %arg22 = %61, %arg23 = %97, %arg24 = %69, %arg25 = %105, %arg26 = %75, %arg27 = %43, %arg28 = %72, %arg29 = %45#0, %arg30 = %74#0, %arg31 = %45#1, %arg32 = %74#1) -> (i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32, i32, i1, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, i32, i64, i64, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>)  : i64 {
        %108 = arith.subi %38, %c2_i64 : i64
        %109 = arith.cmpi slt, %arg9, %108 : i64
        %110 = arith.addi %arg10, %c1_i64 : i64
        %111 = arith.remsi %110, %36 : i64
        %112 = arith.cmpi eq, %111, %c0_i64 : i64
        %113:5 = scf.if %112 -> (tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32) {
          %160 = arith.addi %arg11, %c132_i32 : i32
          %161 = arith.divsi %160, %13 : i32
          %162 = arith.muli %161, %c8_i32 : i32
          %163 = arith.subi %7, %162 : i32
          %164 = arith.minsi %163, %c8_i32 : i32
          %165 = arith.remsi %160, %164 : i32
          %166 = arith.addi %162, %165 : i32
          %167 = arith.remsi %160, %13 : i32
          %168 = arith.divsi %167, %164 : i32
          %169 = arith.muli %166, %c128_i32 : i32
          %170 = arith.muli %168, %c256_i32 : i32
          %171 = tt.splat %169 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %172 = tt.splat %169 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %173 = arith.addi %171, %16 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %174 = arith.addi %172, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %175 = tt.splat %170 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %176 = tt.splat %170 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %177 = arith.addi %175, %18 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %178 = arith.addi %176, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %179 = arith.cmpi slt, %173, %20 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %180 = arith.select %179, %173, %cst_1 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %181 = arith.cmpi slt, %177, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %182 = arith.select %181, %177, %cst_0 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %183 = tt.expand_dims %180 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
          %184 = arith.muli %183, %22 : tensor<128x1xi32, #blocked2>
          %185 = tt.broadcast %184 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
          %186 = tt.expand_dims %182 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
          %187 = arith.muli %186, %24 : tensor<1x256xi32, #blocked1>
          %188 = tt.broadcast %187 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
          scf.yield %174, %178, %185, %188, %160 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17, %arg11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32
        }
        %114 = arith.addi %arg19, %c1_i32 : i32
        %115 = arith.cmpi slt, %114, %c3_i32 : i32
        %116 = arith.select %115, %114, %c0_i32 : i32
        %117 = arith.select %arg20, %cst_4, %arg13 : tensor<128x256xf32, #mma>
        %118 = ttg.memdesc_subview %40[%116, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
        %119 = ttg.async_wait %arg24 {num = 2 : i32}
        %120 = ttg.memdesc_subview %41[%116, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
        %121 = ttng.warp_group_dot %118, %120, %117 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> -> tensor<128x256xf32, #mma>
        %122:3 = ttng.warp_group_dot_wait %121, %118, %120 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
        %123 = arith.addi %arg26, %c1_i32 : i32
        %124 = arith.addi %arg18, %c1_i32 : i32
        %125 = arith.cmpi slt, %124, %c3_i32 : i32
        %126 = arith.select %125, %124, %c0_i32 : i32
        %127 = arith.select %112, %c0_i32, %123 : i32
        %128 = arith.muli %127, %c64_i32 : i32
        %129 = tt.splat %128 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %130 = tt.splat %128 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %131 = arith.addi %129, %14 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %132 = arith.addi %130, %15 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %133 = tt.expand_dims %131 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
        %134 = tt.broadcast %133 : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
        %135 = arith.addi %113#2, %134 : tensor<128x64xi32, #blocked2>
        %136 = tt.addptr %23, %135 : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
        %137 = tt.expand_dims %132 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
        %138 = tt.broadcast %137 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
        %139 = arith.addi %138, %113#3 : tensor<64x256xi32, #blocked1>
        %140 = tt.addptr %25, %139 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
        %141 = arith.subi %arg5, %128 : i32
        %142 = tt.splat %141 : i32 -> tensor<1x64xi32, #blocked2>
        %143 = arith.cmpi slt, %26, %142 : tensor<1x64xi32, #blocked2>
        %144 = tt.broadcast %143 : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
        %145 = ttg.memdesc_subview %40[%126, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>
        %146 = tt.splat %109 : i1 -> tensor<128x64xi1, #blocked2>
        %147 = arith.andi %146, %144 : tensor<128x64xi1, #blocked2>
        %148 = ttg.async_copy_global_to_local %136, %145 mask %147 other %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked2> -> <128x64xf16, #shared, #smem, mutable, 3x128x64>
        %149 = ttg.async_commit_group %148
        %150 = tt.splat %141 : i32 -> tensor<64x1xi32, #blocked1>
        %151 = arith.cmpi slt, %27, %150 : tensor<64x1xi32, #blocked1>
        %152 = tt.broadcast %151 : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
        %153 = ttg.memdesc_subview %41[%126, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256>
        %154 = tt.splat %109 : i1 -> tensor<64x256xi1, #blocked1>
        %155 = arith.andi %154, %152 : tensor<64x256xi1, #blocked1>
        %156 = ttg.async_copy_global_to_local %140, %153 mask %155 other %cst_3 : tensor<64x256x!tt.ptr<f16>, #blocked1> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256>
        %157 = ttg.async_commit_group %156
        %158 = arith.subi %36, %c1_i64 : i64
        %159 = arith.cmpi eq, %arg27, %158 : i64
        scf.if %159 {
          %160 = tt.expand_dims %arg29 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
          %161 = arith.muli %28, %160 : tensor<128x1xi32, #blocked>
          %162 = tt.addptr %29, %161 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
          %163 = tt.expand_dims %arg31 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
          %164 = tt.broadcast %162 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
          %165 = tt.broadcast %163 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
          %166 = tt.addptr %164, %165 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
          %167 = arith.cmpi slt, %160, %30 : tensor<128x1xi32, #blocked>
          %168 = arith.cmpi slt, %163, %31 : tensor<1x256xi32, #blocked>
          %169 = tt.broadcast %167 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
          %170 = tt.broadcast %168 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
          %171 = arith.andi %169, %170 : tensor<128x256xi1, #blocked>
          %172 = arith.truncf %122#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
          %173 = ttg.convert_layout %172 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
          tt.store %166, %173, %171 : tensor<128x256x!tt.ptr<f16>, #blocked>
        }
        scf.yield %111, %113#4, %123, %122#0, %113#0, %113#1, %113#2, %113#3, %126, %116, %arg21, %112, %arg23, %149, %arg25, %157, %127, %arg28, %111, %arg30, %113#0, %arg32, %113#1 : i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32, i32, i1, i1, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, i32, i64, i64, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      }
      %107 = ttg.async_wait  {num = 0 : i32}
      ttg.local_dealloc %40 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
      ttg.local_dealloc %41 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable>
    }
    tt.return
  }
}

