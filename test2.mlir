#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %2 = ub.poison : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %3 = ub.poison : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %4 = ub.poison : tensor<128x64xi32, #blocked1>
    %5 = ub.poison : tensor<64x256xi32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xf16, #blocked3>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c132_i32 = arith.constant 132 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
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
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %20 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %26 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %27 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %28 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %29 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %30 = tt.expand_dims %16 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %31 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked3>
    %32 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
    %33 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked3>
    %34 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %35 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked3>
    %36 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
    %37 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked3>
    %38 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
    %39 = arith.cmpi eq, %12, %c0_i32 : i32
    scf.if %39 {
      scf.for %arg9 = %6 to %13 step %c132_i32  : i32 {
        %40 = arith.divsi %arg9, %14 : i32
        %41 = arith.muli %40, %c8_i32 : i32
        %42 = arith.subi %8, %41 : i32
        %43 = arith.minsi %42, %c8_i32 : i32
        %44 = arith.remsi %arg9, %43 : i32
        %45 = arith.addi %41, %44 : i32
        %46 = arith.remsi %arg9, %14 : i32
        %47 = arith.divsi %46, %43 : i32
        %48 = arith.muli %45, %c128_i32 : i32
        %49 = arith.muli %47, %c256_i32 : i32
        %50 = tt.splat %48 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %51 = arith.addi %50, %17 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
        %52 = tt.splat %49 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %53 = arith.addi %52, %20 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
        %54 = tt.expand_dims %51 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<128x1xi32, #blocked3>
        %55 = arith.muli %31, %54 : tensor<128x1xi32, #blocked3>
        %56 = tt.addptr %33, %55 : tensor<128x1x!tt.ptr<f16>, #blocked3>, tensor<128x1xi32, #blocked3>
        %57 = tt.expand_dims %53 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
        %58 = tt.broadcast %56 : tensor<128x1x!tt.ptr<f16>, #blocked3> -> tensor<128x256x!tt.ptr<f16>, #blocked3>
        %59 = tt.broadcast %57 : tensor<1x256xi32, #blocked3> -> tensor<128x256xi32, #blocked3>
        %60 = tt.addptr %58, %59 : tensor<128x256x!tt.ptr<f16>, #blocked3>, tensor<128x256xi32, #blocked3>
        %61 = arith.cmpi slt, %54, %35 : tensor<128x1xi32, #blocked3>
        %62 = arith.cmpi slt, %57, %37 : tensor<1x256xi32, #blocked3>
        %63 = tt.broadcast %61 : tensor<128x1xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
        %64 = tt.broadcast %62 : tensor<1x256xi1, #blocked3> -> tensor<128x256xi1, #blocked3>
        %65 = arith.andi %63, %64 : tensor<128x256xi1, #blocked3>
        tt.store %60, %cst_1, %65 : tensor<128x256x!tt.ptr<f16>, #blocked3>
      }
    } else {
      %40 = arith.subi %13, %6 : i32
      %41 = arith.ceildivsi %40, %c132_i32 : i32
      %42 = arith.extsi %12 : i32 to i64
      %43 = arith.maxsi %42, %c1_i64 : i64
      %44 = arith.extsi %41 : i32 to i64
      %45 = arith.muli %44, %43 : i64
      %46 = arith.subi %43, %c1_i64 : i64
      %true = arith.constant true
      %false = arith.constant false
      %47:9 = scf.for %arg9 = %c0_i64 to %45 step %c1_i64 iter_args(%arg10 = %c-1_i64, %arg11 = %6, %arg12 = %0, %arg13 = %1, %arg14 = %4, %arg15 = %5, %arg16 = %3, %arg17 = %2, %arg18 = %true) -> (i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>, i1)  : i64 {
        %48 = arith.addi %arg10, %c1_i64 : i64
        %49 = arith.remsi %48, %43 : i64
        %50 = arith.cmpi eq, %49, %c0_i64 : i64
        %51 = arith.select %50, %c0_i32, %arg12 : i32
        %52 = arith.select %50, %false, %arg18 : i1
        %53:4 = scf.if %50 -> (tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>) {
          %82 = arith.divsi %arg11, %14 : i32
          %83 = arith.muli %82, %c8_i32 : i32
          %84 = arith.subi %8, %83 : i32
          %85 = arith.minsi %84, %c8_i32 : i32
          %86 = arith.remsi %arg11, %85 : i32
          %87 = arith.addi %83, %86 : i32
          %88 = arith.remsi %arg11, %14 : i32
          %89 = arith.divsi %88, %85 : i32
          %90 = arith.muli %87, %c128_i32 : i32
          %91 = arith.muli %89, %c256_i32 : i32
          %92 = tt.splat %90 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %93 = tt.splat %90 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %94 = arith.addi %92, %19 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %95 = arith.addi %93, %18 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
          %96 = tt.splat %91 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %97 = tt.splat %91 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
          %98 = arith.addi %96, %21 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %99 = arith.addi %97, %22 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
          %100 = arith.cmpi slt, %94, %23 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %101 = arith.select %100, %94, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
          %102 = arith.cmpi slt, %98, %24 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %103 = arith.select %102, %98, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
          %104 = tt.expand_dims %101 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
          %105 = arith.muli %104, %25 : tensor<128x1xi32, #blocked1>
          %106 = tt.broadcast %105 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
          %107 = tt.expand_dims %103 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
          %108 = arith.muli %107, %27 : tensor<1x256xi32, #blocked>
          %109 = tt.broadcast %108 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
          scf.yield %106, %109, %99, %95 : tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17 : tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        }
        %54 = arith.muli %51, %c64_i32 : i32
        %55 = tt.splat %54 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %56 = tt.splat %54 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %57 = arith.addi %55, %15 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %58 = arith.addi %56, %16 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %59 = tt.expand_dims %57 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %60 = tt.broadcast %59 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %61 = arith.addi %53#0, %60 : tensor<128x64xi32, #blocked1>
        %62 = tt.addptr %26, %61 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %63 = tt.expand_dims %58 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %64 = tt.broadcast %63 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
        %65 = arith.addi %64, %53#1 : tensor<64x256xi32, #blocked>
        %66 = tt.addptr %28, %65 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
        %67 = arith.subi %arg5, %54 : i32
        %68 = tt.splat %67 : i32 -> tensor<1x64xi32, #blocked1>
        %69 = arith.cmpi slt, %29, %68 : tensor<1x64xi32, #blocked1>
        %70 = tt.broadcast %69 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %71 = tt.load %62, %70, %cst_2 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %72 = ttg.local_alloc %71 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %73 = tt.splat %67 : i32 -> tensor<64x1xi32, #blocked>
        %74 = arith.cmpi slt, %30, %73 : tensor<64x1xi32, #blocked>
        %75 = tt.broadcast %74 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
        %76 = tt.load %66, %75, %cst_3 : tensor<64x256x!tt.ptr<f16>, #blocked>
        %77 = ttg.local_alloc %76 : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared1, #smem>
        %78 = ttng.warp_group_dot %72, %77, %arg13, %52 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<128x256xf32, #mma>
        %79 = arith.addi %51, %c1_i32 : i32
        %80 = arith.cmpi eq, %49, %46 : i64
        %81 = scf.if %80 -> (i32) {
          %82 = tt.expand_dims %53#3 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
          %83 = arith.muli %32, %82 : tensor<128x1xi32, #blocked2>
          %84 = tt.addptr %34, %83 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
          %85 = tt.expand_dims %53#2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
          %86 = tt.broadcast %84 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
          %87 = tt.broadcast %85 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
          %88 = tt.addptr %86, %87 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
          %89 = arith.cmpi slt, %82, %36 : tensor<128x1xi32, #blocked2>
          %90 = arith.cmpi slt, %85, %38 : tensor<1x256xi32, #blocked2>
          %91 = tt.broadcast %89 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
          %92 = tt.broadcast %90 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
          %93 = arith.andi %91, %92 : tensor<128x256xi1, #blocked2>
          %94 = arith.truncf %78 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
          %95 = ttg.convert_layout %94 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
          tt.store %88, %95, %93 : tensor<128x256x!tt.ptr<f16>, #blocked2>
          %96 = arith.addi %arg11, %c132_i32 : i32
          scf.yield %96 : i32
        } else {
          scf.yield %arg11 : i32
        }
        scf.yield %49, %81, %79, %78, %53#0, %53#1, %53#2, %53#3, %true : i64, i32, i32, tensor<128x256xf32, #mma>, tensor<128x64xi32, #blocked1>, tensor<64x256xi32, #blocked>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>, i1
      }
    }
    tt.return
  }
}

