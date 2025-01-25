#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %14 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %17 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1>
    %18 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1>
    %19 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked>
    %20 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked>
    %21 = tt.expand_dims %9 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %22 = tt.expand_dims %10 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %23 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2>
    %24 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %25 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2>
    %26 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2>
    %27 = arith.subi %7, %0 : i32
    %28 = arith.ceildivsi %27, %c132_i32 : i32
    %29 = arith.subi %6, %c0_i32 : i32
    %30 = arith.ceildivsi %29, %c1_i32 : i32
    %c0_i64 = arith.constant 0 : i64
    %31 = arith.extsi %30 : i32 to i64
    %c1_i64 = arith.constant 1 : i64
    %32 = arith.maxsi %c1_i64, %31 : i64
    %33 = arith.addi %c0_i64, %32 : i64
    %c0_i64_4 = arith.constant 0 : i64
    %34 = arith.subi %33, %c0_i64_4 : i64
    %35 = arith.extsi %28 : i32 to i64
    %36 = arith.muli %35, %34 : i64
    %c-1_i64 = arith.constant -1 : i64
    %37 = arith.subi %0, %c132_i32 : i32
    %38 = ub.poison : i32
    %39 = ub.poison : tensor<128x256xf32, #mma>
    %40 = ub.poison : i32
    %41 = ub.poison : i32
    %c0_i64_5 = arith.constant 0 : i64
    %c1_i64_6 = arith.constant 1 : i64
    %42:6 = scf.for %arg9 = %c0_i64_5 to %36 step %c1_i64_6 iter_args(%arg10 = %c-1_i64, %arg11 = %37, %arg12 = %38, %arg13 = %39, %arg14 = %40, %arg15 = %41) -> (i64, i32, i32, tensor<128x256xf32, #mma>, i32, i32)  : i64 {
      %c1_i64_7 = arith.constant 1 : i64
      %43 = arith.addi %arg10, %c1_i64_7 : i64
      %44 = arith.remsi %43, %34 : i64
      %c0_i64_8 = arith.constant 0 : i64
      %45 = arith.subi %c0_i64, %c0_i64_8 : i64
      %46 = arith.cmpi eq, %44, %45 : i64
      %47:5 = scf.if %46 -> (i32, i32, i32, tensor<128x256xf32, #mma>, i32) {
        %56 = arith.addi %arg11, %c132_i32 : i32
        %57 = arith.divsi %56, %8 : i32
        %58 = arith.muli %57, %c8_i32 : i32
        %59 = arith.subi %2, %58 : i32
        %60 = arith.minsi %59, %c8_i32 : i32
        %61 = arith.remsi %56, %60 : i32
        %62 = arith.addi %58, %61 : i32
        %63 = arith.remsi %56, %8 : i32
        %64 = arith.divsi %63, %60 : i32
        %65 = arith.muli %62, %c128_i32 : i32
        %66 = arith.muli %64, %c256_i32 : i32
        scf.yield %c0_i32, %65, %66, %cst_3, %56 : i32, i32, i32, tensor<128x256xf32, #mma>, i32
      } else {
        scf.yield %arg12, %arg14, %arg15, %arg13, %arg11 : i32, i32, i32, tensor<128x256xf32, #mma>, i32
      }
      %48 = arith.extsi %30 : i32 to i64
      %49 = arith.addi %45, %48 : i64
      %50 = arith.cmpi sge, %44, %45 : i64
      %51 = arith.cmpi slt, %44, %49 : i64
      %52 = arith.andi %50, %51 : i1
      %true = arith.constant true
      %53:2 = scf.if %true -> (i32, tensor<128x256xf32, #mma>) {
        %56 = tt.splat %47#1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %57 = arith.addi %56, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %58 = tt.splat %47#2 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %59 = arith.addi %58, %13 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %60 = arith.cmpi slt, %57, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %61 = arith.select %60, %57, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
        %62 = arith.cmpi slt, %59, %16 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %63 = arith.select %62, %59, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
        %64 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
        %65 = arith.muli %64, %17 : tensor<128x1xi32, #blocked1>
        %66 = tt.broadcast %65 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %67 = tt.expand_dims %63 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
        %68 = arith.muli %67, %19 : tensor<1x256xi32, #blocked>
        %69 = tt.broadcast %68 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
        %70 = arith.muli %47#0, %c64_i32 : i32
        %71 = tt.splat %70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %72 = tt.splat %70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %73 = arith.addi %71, %9 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %74 = arith.addi %72, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %75 = tt.expand_dims %73 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %76 = tt.broadcast %75 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %77 = arith.addi %66, %76 : tensor<128x64xi32, #blocked1>
        %78 = tt.addptr %18, %77 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %79 = tt.expand_dims %74 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %80 = tt.broadcast %79 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
        %81 = arith.addi %80, %69 : tensor<64x256xi32, #blocked>
        %82 = tt.addptr %20, %81 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
        %83 = arith.subi %arg5, %70 : i32
        %84 = tt.splat %83 : i32 -> tensor<1x64xi32, #blocked1>
        %85 = arith.cmpi slt, %21, %84 : tensor<1x64xi32, #blocked1>
        %86 = tt.broadcast %85 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1>
        %87 = tt.load %78, %86, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1>
        %88 = ttg.local_alloc %87 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %89 = tt.splat %83 : i32 -> tensor<64x1xi32, #blocked>
        %90 = arith.cmpi slt, %22, %89 : tensor<64x1xi32, #blocked>
        %91 = tt.broadcast %90 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked>
        %92 = tt.load %82, %91, %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked>
        %93 = ttg.local_alloc %92 : (tensor<64x256xf16, #blocked>) -> !ttg.memdesc<64x256xf16, #shared1, #smem>
        %94 = ttng.warp_group_dot %88, %93, %47#3 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<128x256xf32, #mma>
        %95 = arith.addi %47#0, %c1_i32 : i32
        scf.yield %95, %94 : i32, tensor<128x256xf32, #mma>
      } else {
        scf.yield %47#0, %arg13 : i32, tensor<128x256xf32, #mma>
      }
      %c1_i64_9 = arith.constant 1 : i64
      %54 = arith.subi %34, %c1_i64_9 : i64
      %55 = arith.cmpi eq, %44, %54 : i64
      scf.if %55 {
        %56 = tt.splat %47#1 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %57 = arith.addi %56, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %58 = tt.splat %47#2 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %59 = arith.addi %58, %14 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
        %60 = tt.expand_dims %57 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
        %61 = arith.muli %23, %60 : tensor<128x1xi32, #blocked2>
        %62 = tt.addptr %24, %61 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
        %63 = tt.expand_dims %59 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
        %64 = tt.broadcast %62 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
        %65 = tt.broadcast %63 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
        %66 = tt.addptr %64, %65 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
        %67 = arith.cmpi slt, %60, %25 : tensor<128x1xi32, #blocked2>
        %68 = arith.cmpi slt, %63, %26 : tensor<1x256xi32, #blocked2>
        %69 = tt.broadcast %67 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %70 = tt.broadcast %68 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
        %71 = arith.andi %69, %70 : tensor<128x256xi1, #blocked2>
        %72 = arith.truncf %53#1 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
        %73 = ttg.convert_layout %72 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
        tt.store %66, %73, %71 : tensor<128x256x!tt.ptr<f16>, #blocked2>
      } else {
      }
      scf.yield %44, %47#4, %53#0, %53#1, %47#1, %47#2 : i64, i32, i32, tensor<128x256xf32, #mma>, i32, i32
    }
    tt.return
  }
}

