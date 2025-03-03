#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x32xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c31_i32 : i32
    %2 = arith.divsi %1, %c32_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c32_i32 : i32
    %15 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %17 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %18 = tt.splat %14 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %17, %15 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %18, %16 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %21 = tt.splat %arg3 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.remsi %19, %21 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %23 = arith.muli %13, %c64_i32 : i32
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.splat %23 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %26 = arith.addi %25, %24 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27 = tt.splat %arg4 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %28 = arith.remsi %26, %27 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %29 = tt.expand_dims %22 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %30 = tt.splat %arg6 : i32 -> tensor<32x1xi32, #blocked>
    %31 = arith.muli %29, %30 : tensor<32x1xi32, #blocked>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %34 = tt.broadcast %31 : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %35 = tt.broadcast %33 : tensor<1x32xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %36 = arith.addi %34, %35 : tensor<32x32xi32, #blocked>
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    %38 = tt.addptr %37, %36 : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
    %39 = tt.expand_dims %16 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %40 = tt.splat %arg7 : i32 -> tensor<32x1xi32, #blocked1>
    %41 = arith.muli %39, %40 : tensor<32x1xi32, #blocked1>
    %42 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %43 = tt.broadcast %41 : tensor<32x1xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %44 = tt.broadcast %42 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %45 = arith.addi %43, %44 : tensor<32x64xi32, #blocked1>
    %46 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %47 = tt.addptr %46, %45 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %48 = arith.addi %arg5, %c31_i32 : i32
    %49 = arith.divsi %48, %c32_i32 : i32
    %50 = arith.muli %arg7, %c32_i32 : i32
    %51 = tt.splat %50 : i32 -> tensor<32x64xi32, #blocked1>
    %52:3 = scf.for %arg9 = %c0_i32 to %49 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %38, %arg12 = %47) -> (tensor<32x64xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)  : i32 {
      %71 = arith.muli %arg9, %c32_i32 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %72 = arith.subi %arg5, %71 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32
      %73 = tt.splat %72 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32 -> tensor<1x32xi32, #blocked>
      %74 = arith.cmpi slt, %33, %73 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<1x32xi32, #blocked>
      %75 = tt.broadcast %74 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<1x32xi1, #blocked> -> tensor<32x32xi1, #blocked>
      %76 = tt.load %arg11, %75, %cst_0 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x32x!tt.ptr<f16>, #blocked>
      %77 = tt.splat %72 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : i32 -> tensor<32x1xi32, #blocked1>
      %78 = arith.cmpi slt, %39, %77 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x1xi32, #blocked1>
      %79 = tt.broadcast %78 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
      %80 = tt.load %arg12, %79, %cst_1 {loop.cluster = 3 : i32, loop.stage = 0 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked1>
      %81 = ttg.convert_layout %76 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %82 = ttg.convert_layout %80 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x64xf16, #blocked1> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %83 = tt.dot %81, %82, %arg10, inputPrecision = tf32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x64xf32, #mma>
      %84 = tt.addptr %arg11, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x32xi32, #blocked>
      %85 = tt.addptr %arg12, %51 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
      scf.yield %83, %84, %85 : tensor<32x64xf32, #mma>, tensor<32x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>
    }
    %53 = arith.truncf %52#0 : tensor<32x64xf32, #mma> to tensor<32x64xf16, #mma>
    %54 = tt.expand_dims %20 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %55 = tt.splat %arg8 : i32 -> tensor<32x1xi32, #blocked1>
    %56 = arith.muli %55, %54 : tensor<32x1xi32, #blocked1>
    %57 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #blocked1>
    %58 = tt.addptr %57, %56 : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi32, #blocked1>
    %59 = tt.expand_dims %26 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %60 = tt.broadcast %58 : tensor<32x1x!tt.ptr<f16>, #blocked1> -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %61 = tt.broadcast %59 : tensor<1x64xi32, #blocked1> -> tensor<32x64xi32, #blocked1>
    %62 = tt.addptr %60, %61 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %63 = tt.splat %arg3 : i32 -> tensor<32x1xi32, #blocked1>
    %64 = arith.cmpi slt, %54, %63 : tensor<32x1xi32, #blocked1>
    %65 = tt.splat %arg4 : i32 -> tensor<1x64xi32, #blocked1>
    %66 = arith.cmpi slt, %59, %65 : tensor<1x64xi32, #blocked1>
    %67 = tt.broadcast %64 : tensor<32x1xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %68 = tt.broadcast %66 : tensor<1x64xi1, #blocked1> -> tensor<32x64xi1, #blocked1>
    %69 = arith.andi %67, %68 : tensor<32x64xi1, #blocked1>
    %70 = ttg.convert_layout %53 : tensor<32x64xf16, #mma> -> tensor<32x64xf16, #blocked1>
    tt.store %62, %70, %69 : tensor<32x64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

