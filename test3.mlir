#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c-1_i64 = arith.constant -1 : i64
    %0 = ub.poison : i32
    %1 = ub.poison : tensor<128x256xf32, #mma>
    %2 = ub.poison : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %3 = ub.poison : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = ub.poison : tensor<128x64xi32, #blocked1>
    %5 = ub.poison : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #blocked2>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.addi %arg5, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %23 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %25 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %26 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %27 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %28 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %29 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %30 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
    %31 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked1>
    %32 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %33 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %34 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
    %35 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked1>
    %36 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
    %37 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked1>
    %38 = arith.cmpi eq, %12, %c0_i32 : i32
    scf.if %38 {
      scf.for %arg9 = %6 to %13 step %c132_i32  : i32 {
        %39 = arith.divsi %arg9, %14 : i32
        %40 = arith.muli %39, %c8_i32 : i32
        %41 = arith.subi %8, %40 : i32
        %42 = arith.minsi %41, %c8_i32 : i32
        %43 = arith.remsi %arg9, %42 : i32
        %44 = arith.addi %40, %43 : i32
        %45 = arith.remsi %arg9, %14 : i32
        %46 = arith.divsi %45, %42 : i32
        %47 = arith.muli %44, %c128_i32 : i32
        %48 = arith.muli %46, %c256_i32 : i32
        %49 = tt.splat %47 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %50 = arith.addi %49, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %51 = tt.splat %48 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %52 = arith.addi %51, %19 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %53 = tt.expand_dims %50 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %54 = arith.muli %30, %53 : tensor<128x1xi32, #blocked2>
        %55 = tt.addptr %32, %54 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %56 = tt.expand_dims %52 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %57 = tt.broadcast %55 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %58 = tt.broadcast %56 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %59 = tt.addptr %57, %58 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %60 = arith.cmpi slt, %53, %34 : tensor<128x1xi32, #blocked2>
        %61 = arith.cmpi slt, %56, %36 : tensor<1x256xi32, #blocked2>
        %62 = tt.broadcast %60 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %63 = tt.broadcast %61 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %64 = arith.andi %62, %63 : tensor<128x256xi1, #blocked2>
        tt.store %59, %cst_1, %64 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      }
    } else {
      %39 = arith.subi %13, %6 : i32
      %40 = arith.ceildivsi %39, %c132_i32 : i32
      %41 = arith.extsi %12 : i32 to i64
      %42 = arith.maxsi %41, %c1_i64 : i64
      %43 = arith.extsi %40 : i32 to i64
      %44 = arith.muli %43, %42 : i64
      %45 = arith.subi %42, %c1_i64 : i64
      %46:9 = scf.for %arg9 = %c0_i64 to %44 step %c1_i64 iter_args(%arg10 = %c-1_i64, %arg11 = %6, %arg12 = %0, %arg13 = %1, %arg14 = %4, %arg15 = %5, %arg16 = %3, %arg17 = %2, %arg18 = %true) -> (i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, i1)  : i64 {
        %47 = arith.addi %arg10, %c1_i64 : i64
        %48 = arith.remsi %47, %42 : i64
        %49 = arith.cmpi eq, %48, %c0_i64 : i64
        %50 = arith.select %49, %c0_i32, %arg12 : i32
        %51 = arith.select %49, %false, %arg18 : i1
        %52:4 = scf.if %49 -> (tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>) {
          %81 = arith.divsi %arg11, %14 : i32
          %82 = arith.muli %81, %c8_i32 : i32
          %83 = arith.subi %8, %82 : i32
          %84 = arith.minsi %83, %c8_i32 : i32
          %85 = arith.remsi %arg11, %84 : i32
          %86 = arith.addi %82, %85 : i32
          %87 = arith.remsi %arg11, %14 : i32
          %88 = arith.divsi %87, %84 : i32
          %89 = arith.muli %86, %c128_i32 : i32
          %90 = arith.muli %88, %c256_i32 : i32
          %91 = tt.splat %89 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %92 = arith.addi %91, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %93 = tt.splat %90 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %94 = tt.splat %90 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %95 = arith.addi %93, %20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %96 = arith.addi %94, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
          %97 = arith.cmpi slt, %92, %22 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %98 = arith.select %97, %92, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %99 = arith.cmpi slt, %95, %23 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %100 = arith.select %99, %95, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %101 = tt.expand_dims %98 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
          %102 = arith.muli %101, %24 : tensor<128x1xi32, #blocked1>
          %103 = tt.broadcast %102 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
          %104 = tt.expand_dims %100 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
          %105 = arith.muli %104, %26 : tensor<1x256xi32, #blocked>
          %106 = tt.broadcast %105 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
          scf.yield %103, %106, %96, %92 : tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17 : tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        }
        %53 = arith.muli %50, %c64_i32 : i32
        %54 = tt.splat %53 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %55 = tt.splat %53 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %56 = arith.addi %54, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %57 = arith.addi %55, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %58 = tt.expand_dims %56 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %59 = tt.broadcast %58 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %60 = arith.addi %52#0, %59 : tensor<128x64xi32, #blocked1>
        %61 = tt.addptr %25, %60 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %62 = tt.expand_dims %57 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %63 = tt.broadcast %62 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
        %64 = arith.addi %63, %52#1 : tensor<64x256xi32, #blocked>
        %65 = tt.addptr %27, %64 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
        %66 = arith.subi %arg5, %53 : i32
        %67 = tt.splat %66 : i32 -> tensor<1x64xi32, #blocked1>
        %68 = arith.cmpi slt, %28, %67 : tensor<1x64xi32, #blocked1>
        %69 = tt.broadcast %68 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %70 = tt.load %61, %69, %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %71 = ttg.local_alloc %70 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %72 = tt.splat %66 : i32 -> tensor<64x1xi32, #blocked>
        %73 = arith.cmpi slt, %29, %72 : tensor<64x1xi32, #blocked>
        %74 = tt.broadcast %73 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
        %75 = tt.load %65, %74, %cst_3 : tensor<64x256x!tt.ptr<f16>, #blocked>
        %76 = ttg.local_alloc %75 : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %77 = ttng.warp_group_dot %71, %76, %arg13, %51 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %78 = arith.addi %50, %c1_i32 : i32
        %79 = arith.cmpi eq, %48, %45 : i64
        %80 = scf.if %79 -> (i32) {
          %81 = tt.expand_dims %52#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
          %82 = arith.muli %31, %81 : tensor<128x1xi32, #blocked1>
          %83 = tt.addptr %33, %82 : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
          %84 = tt.expand_dims %52#2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
          %85 = tt.broadcast %83 : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
          %86 = tt.broadcast %84 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
          %87 = tt.addptr %85, %86 : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
          %88 = arith.cmpi slt, %81, %35 : tensor<128x1xi32, #blocked1>
          %89 = arith.cmpi slt, %84, %37 : tensor<1x256xi32, #blocked1>
          %90 = tt.broadcast %88 : tensor<128x1xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
          %91 = tt.broadcast %89 : tensor<1x256xi1, #blocked1> -> tensor<128x256xi1, #blocked1>
          %92 = arith.andi %90, %91 : tensor<128x256xi1, #blocked1>
          %93 = arith.truncf %77 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
          %94 = ttg.convert_layout %93 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
          tt.store %87, %94, %92 : tensor<128x256x!tt.ptr<f16>, #blocked1>
          %95 = arith.addi %arg11, %c132_i32 : i32
          scf.yield %95 : i32
        } else {
          scf.yield %arg11 : i32
        }
        scf.yield %48, %80, %78, %77, %52#0, %52#1, %52#2, %52#3, %true : i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, i1
      }
    }
    tt.return
  }
}

