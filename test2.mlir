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
    scf.for %arg9 = %0 to %7 step %c132_i32  : i32 {
      %27 = arith.divsi %arg9, %8 : i32
      %28 = arith.muli %27, %c8_i32 : i32
      %29 = arith.subi %2, %28 : i32
      %30 = arith.minsi %29, %c8_i32 : i32
      %31 = arith.remsi %arg9, %30 : i32
      %32 = arith.addi %28, %31 : i32
      %33 = arith.remsi %arg9, %8 : i32
      %34 = arith.divsi %33, %30 : i32
      %35 = arith.muli %32, %c128_i32 : i32
      %36 = arith.muli %34, %c256_i32 : i32
      %37 = tt.splat %35 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %38 = tt.splat %35 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %39 = arith.addi %37, %11 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %40 = arith.addi %38, %12 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
      %41 = tt.splat %36 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %42 = tt.splat %36 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %43 = arith.addi %41, %13 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %44 = arith.addi %42, %14 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %45 = arith.cmpi slt, %39, %15 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %46 = arith.select %45, %39, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %47 = arith.cmpi slt, %43, %16 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %48 = arith.select %47, %43, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %49 = tt.expand_dims %46 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
      %50 = arith.muli %49, %17 : tensor<128x1xi32, #blocked1>
      %51 = tt.broadcast %50 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
      %52 = tt.expand_dims %48 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
      %53 = arith.muli %52, %19 : tensor<1x256xi32, #blocked>
      %54 = tt.broadcast %53 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked>
      %55 = scf.for %arg10 = %c0_i32 to %6 step %c1_i32 iter_args(%arg11 = %cst_3) -> (tensor<128x256xf32, #mma>)  : i32 {
        %70 = arith.muli %arg10, %c64_i32 : i32
        %71 = tt.splat %70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %72 = tt.splat %70 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %73 = arith.addi %71, %9 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
        %74 = arith.addi %72, %10 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %75 = tt.expand_dims %73 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
        %76 = tt.broadcast %75 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1>
        %77 = arith.addi %51, %76 : tensor<128x64xi32, #blocked1>
        %78 = tt.addptr %18, %77 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1>
        %79 = tt.expand_dims %74 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
        %80 = tt.broadcast %79 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked>
        %81 = arith.addi %80, %54 : tensor<64x256xi32, #blocked>
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
        %94 = ttng.warp_group_dot %88, %93, %arg11 {inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared1, #smem> -> tensor<128x256xf32, #mma>
        scf.yield %94 : tensor<128x256xf32, #mma>
      }
      %56 = tt.expand_dims %40 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
      %57 = arith.muli %23, %56 : tensor<128x1xi32, #blocked2>
      %58 = tt.addptr %24, %57 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
      %59 = tt.expand_dims %44 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
      %60 = tt.broadcast %58 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
      %61 = tt.broadcast %59 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
      %62 = tt.addptr %60, %61 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
      %63 = arith.cmpi slt, %56, %25 : tensor<128x1xi32, #blocked2>
      %64 = arith.cmpi slt, %59, %26 : tensor<1x256xi32, #blocked2>
      %65 = tt.broadcast %63 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %66 = tt.broadcast %64 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %67 = arith.andi %65, %66 : tensor<128x256xi1, #blocked2>
      %68 = arith.truncf %55 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %69 = ttg.convert_layout %68 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2>
      tt.store %62, %69, %67 : tensor<128x256x!tt.ptr<f16>, #blocked2>
    }
    tt.return
  }
}

