#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #blocked>
    %0 = ub.poison : tensor<64x256xi32, #blocked1>
    %1 = ub.poison : tensor<128x64xi32, #blocked2>
    %2 = ub.poison : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = ub.poison : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %4 = ub.poison : tensor<128x256xf32, #mma>
    %5 = ub.poison : i32
    %c-1_i64 = arith.constant -1 : i64
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
    %true = arith.constant true
    %false = arith.constant false
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.addi %arg5, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %21 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %22 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %23 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked2>
    %24 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked2>
    %25 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked1>
    %26 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %27 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %28 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %29 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %30 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %31 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
    %32 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked>
    %33 = arith.cmpi eq, %12, %c0_i32 : i32
    scf.if %33 {
      scf.for %arg9 = %6 to %13 step %c132_i32  : i32 {
        %34 = arith.divsi %arg9, %14 : i32
        %35 = arith.muli %34, %c8_i32 : i32
        %36 = arith.subi %8, %35 : i32
        %37 = arith.minsi %36, %c8_i32 : i32
        %38 = arith.remsi %arg9, %37 : i32
        %39 = arith.addi %35, %38 : i32
        %40 = arith.remsi %arg9, %14 : i32
        %41 = arith.divsi %40, %37 : i32
        %42 = arith.muli %39, %c128_i32 : i32
        %43 = arith.muli %41, %c256_i32 : i32
        %44 = tt.splat %42 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %45 = arith.addi %44, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %46 = tt.splat %43 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %47 = arith.addi %46, %20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %48 = tt.expand_dims %45 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
        %49 = arith.muli %29, %48 : tensor<128x1xi32, #blocked>
        %50 = tt.addptr %30, %49 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
        %51 = tt.expand_dims %47 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
        %52 = tt.broadcast %50 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
        %53 = tt.broadcast %51 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
        %54 = tt.addptr %52, %53 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
        %55 = arith.cmpi slt, %48, %31 : tensor<128x1xi32, #blocked>
        %56 = arith.cmpi slt, %51, %32 : tensor<1x256xi32, #blocked>
        %57 = tt.broadcast %55 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %58 = tt.broadcast %56 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
        %59 = arith.andi %57, %58 : tensor<128x256xi1, #blocked>
        tt.store %54, %cst, %59 : tensor<128x256x!tt.ptr<f16>, #blocked>
      }
    } else {
      %34 = arith.subi %13, %6 : i32
      %35 = arith.ceildivsi %34, %c132_i32 : i32
      %36 = arith.extsi %12 : i32 to i64
      %37 = arith.maxsi %36, %c1_i64 : i64
      %38 = arith.extsi %35 : i32 to i64
      %39 = arith.muli %38, %37 : i64
      %40 = arith.subi %6, %c132_i32 : i32
      %41:9 = scf.for %arg9 = %c0_i64 to %39 step %c1_i64 iter_args(%arg10 = %c-1_i64, %arg11 = %40, %arg12 = %5, %arg13 = %4, %arg14 = %3, %arg15 = %2, %arg16 = %1, %arg17 = %0, %arg18 = %true) -> (i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i1)  : i64 {
        %42 = arith.addi %arg10, %c1_i64 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %43 = arith.remsi %42, %37 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %44 = arith.cmpi eq, %43, %c0_i64 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i64
        %45:7 = scf.if %44 -> (tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32, i32, i1) {
          %74 = arith.addi %arg11, %c132_i32 : i32
          %75 = arith.divsi %74, %14 : i32
          %76 = arith.muli %75, %c8_i32 : i32
          %77 = arith.subi %8, %76 : i32
          %78 = arith.minsi %77, %c8_i32 : i32
          %79 = arith.remsi %74, %78 : i32
          %80 = arith.addi %76, %79 : i32
          %81 = arith.remsi %74, %14 : i32
          %82 = arith.divsi %81, %78 : i32
          %83 = arith.muli %80, %c128_i32 : i32
          %84 = arith.muli %82, %c256_i32 : i32
          %85 = tt.splat %83 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %86 = tt.splat %83 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %87 = arith.addi %85, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %88 = arith.addi %86, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
          %89 = tt.splat %84 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %90 = tt.splat %84 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %91 = arith.addi %89, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %92 = arith.addi %90, %20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %93 = arith.cmpi slt, %87, %21 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %94 = arith.select %93, %87, %cst_1 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %95 = arith.cmpi slt, %91, %22 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %96 = arith.select %95, %91, %cst_0 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %97 = tt.expand_dims %94 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
          %98 = arith.muli %97, %23 : tensor<128x1xi32, #blocked2>
          %99 = tt.broadcast %98 : tensor<128x1xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
          %100 = tt.expand_dims %96 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
          %101 = arith.muli %100, %25 : tensor<1x256xi32, #blocked1>
          %102 = tt.broadcast %101 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
          scf.yield %88, %92, %99, %102, %74, %c0_i32, %false : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32, i32, i1
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17, %arg11, %arg12, %arg18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i32, i32, i1
        } {loop.cluster = 1 : i32, loop.stage = 0 : i32}
        %46 = arith.muli %45#5, %c64_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
        %47 = tt.splat %46 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %48 = tt.splat %46 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %49 = arith.addi %47, %15 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %50 = arith.addi %48, %16 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %51 = tt.expand_dims %49 {axis = 0 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
        %52 = tt.broadcast %51 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x64xi32, #blocked2> -> tensor<128x64xi32, #blocked2>
        %53 = arith.addi %45#2, %52 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64xi32, #blocked2>
        %54 = tt.addptr %24, %53 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked2>, tensor<128x64xi32, #blocked2>
        %55 = tt.expand_dims %50 {axis = 1 : i32, loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
        %56 = tt.broadcast %55 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
        %57 = arith.addi %56, %45#3 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x256xi32, #blocked1>
        %58 = tt.addptr %26, %57 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
        %59 = arith.subi %arg5, %46 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
        %60 = tt.splat %59 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<1x64xi32, #blocked2>
        %61 = arith.cmpi slt, %27, %60 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x64xi32, #blocked2>
        %62 = tt.broadcast %61 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<1x64xi1, #blocked2> -> tensor<128x64xi1, #blocked2>
        %63 = tt.load %54, %62, %cst_2 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x64x!tt.ptr<f16>, #blocked2>
        %64 = ttg.local_alloc %63 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %65 = tt.splat %59 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32 -> tensor<64x1xi32, #blocked1>
        %66 = arith.cmpi slt, %28, %65 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x1xi32, #blocked1>
        %67 = tt.broadcast %66 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x1xi1, #blocked1> -> tensor<64x256xi1, #blocked1>
        %68 = tt.load %58, %67, %cst_3 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        %69 = ttg.local_alloc %68 {loop.cluster = 2 : i32, loop.stage = 2 : i32} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared1, #smem>
        %70 = ttng.warp_group_dot %64, %69, %arg13, %45#6 {inputPrecision = 0 : i32, loop.cluster = 2 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<128x256xf32, #mma>
        %71 = arith.addi %45#5, %c1_i32 {loop.cluster = 0 : i32, loop.stage = 1 : i32} : i32
        %72 = arith.subi %37, %c1_i64 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : i64
        %73 = arith.cmpi eq, %43, %72 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : i64
        scf.if %73 {
          %74 = tt.expand_dims %45#0 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
          %75 = arith.muli %29, %74 : tensor<128x1xi32, #blocked>
          %76 = tt.addptr %30, %75 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
          %77 = tt.expand_dims %45#1 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
          %78 = tt.broadcast %76 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
          %79 = tt.broadcast %77 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
          %80 = tt.addptr %78, %79 : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
          %81 = arith.cmpi slt, %74, %31 : tensor<128x1xi32, #blocked>
          %82 = arith.cmpi slt, %77, %32 : tensor<1x256xi32, #blocked>
          %83 = tt.broadcast %81 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
          %84 = tt.broadcast %82 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
          %85 = arith.andi %83, %84 : tensor<128x256xi1, #blocked>
          %86 = arith.truncf %70 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
          %87 = ttg.convert_layout %86 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked>
          tt.store %80, %87, %85 : tensor<128x256x!tt.ptr<f16>, #blocked>
        } {loop.cluster = 5 : i32, loop.stage = 2 : i32}
        scf.yield %43, %45#4, %71, %70, %45#0, %45#1, %45#2, %45#3, %true : i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<128x64xi32, #blocked2>, tensor<64x256xi32, #blocked1>, i1
      }
    }
    tt.return
  }
}

