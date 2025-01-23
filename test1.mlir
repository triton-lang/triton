module {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = ub.poison : tensor<64x256xi32>
    %1 = ub.poison : tensor<128x64xi32>
    %2 = ub.poison : tensor<256xi32>
    %3 = ub.poison : tensor<128xi32>
    %4 = ub.poison : tensor<128x256xf32>
    %5 = ub.poison : i32
    %c-1_i64 = arith.constant -1 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x256xf16>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c132_i32 = arith.constant 132 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant dense<0> : tensor<256xi32>
    %cst_3 = arith.constant dense<0> : tensor<128xi32>
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %6 = tt.get_program_id x : i32
    %7 = arith.addi %arg3, %c127_i32 : i32
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = arith.addi %arg4, %c255_i32 : i32
    %10 = arith.divsi %9, %c256_i32 : i32
    %11 = arith.addi %arg5, %c63_i32 : i32
    %12 = arith.divsi %11, %c64_i32 : i32
    %13 = arith.muli %8, %10 : i32
    %14 = arith.muli %10, %c8_i32 : i32
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %17 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %18 = tt.splat %arg3 : i32 -> tensor<128xi32>
    %19 = tt.splat %arg4 : i32 -> tensor<256xi32>
    %20 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %21 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>>
    %22 = tt.splat %arg7 : i32 -> tensor<1x256xi32>
    %23 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>>
    %24 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %25 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %26 = tt.splat %arg8 : i32 -> tensor<128x1xi32>
    %27 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %28 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %29 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %30 = arith.subi %13, %6 : i32
    %31 = arith.ceildivsi %30, %c132_i32 : i32
    %32 = arith.extsi %12 : i32 to i64
    %33 = arith.maxsi %32, %c1_i64 : i64
    %34 = arith.extsi %31 : i32 to i64
    %35 = arith.muli %34, %33 : i64
    %36 = arith.subi %33, %c1_i64 : i64
    %37:8 = scf.for %arg9 = %c0_i64 to %35 step %c1_i64 iter_args(%arg10 = %c-1_i64, %arg11 = %6, %arg12 = %5, %arg13 = %4, %arg14 = %3, %arg15 = %2, %arg16 = %1, %arg17 = %0) -> (i64, i32, i32, tensor<128x256xf32>, tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>)  : i64 {
      %38 = arith.addi %arg10, %c1_i64 : i64
      %39 = arith.remsi %38, %33 : i64
      %40 = arith.cmpi eq, %39, %c0_i64 : i64
      %41 = arith.select %40, %c0_i32, %arg12 : i32
      %42 = arith.select %40, %cst, %arg13 : tensor<128x256xf32>
      %43:4 = scf.if %40 -> (tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>) {
        %50 = arith.divsi %arg11, %14 : i32
        %51 = arith.muli %50, %c8_i32 : i32
        %52 = arith.subi %8, %51 : i32
        %53 = arith.minsi %52, %c8_i32 : i32
        %54 = arith.remsi %arg11, %53 : i32
        %55 = arith.addi %51, %54 : i32
        %56 = arith.remsi %arg11, %14 : i32
        %57 = arith.divsi %56, %53 : i32
        %58 = arith.muli %55, %c128_i32 : i32
        %59 = arith.muli %57, %c256_i32 : i32
        %60 = tt.splat %58 : i32 -> tensor<128xi32>
        %61 = arith.addi %60, %16 : tensor<128xi32>
        %62 = tt.splat %59 : i32 -> tensor<256xi32>
        %63 = arith.addi %62, %17 : tensor<256xi32>
        %64 = arith.cmpi slt, %61, %18 : tensor<128xi32>
        %65 = arith.select %64, %61, %cst_3 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
        %66 = arith.cmpi slt, %63, %19 : tensor<256xi32>
        %67 = arith.select %66, %63, %cst_2 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1>, tensor<256xi32>
        %68 = tt.expand_dims %65 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %69 = arith.muli %68, %20 : tensor<128x1xi32>
        %70 = tt.broadcast %69 : tensor<128x1xi32> -> tensor<128x64xi32>
        %71 = tt.expand_dims %67 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
        %72 = arith.muli %71, %22 : tensor<1x256xi32>
        %73 = tt.broadcast %72 : tensor<1x256xi32> -> tensor<64x256xi32>
        scf.yield %61, %63, %70, %73 : tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
      } else {
        scf.yield %arg14, %arg15, %arg16, %arg17 : tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
      }
      %44 = arith.cmpi sge, %39, %c0_i64 : i64
      %45 = arith.cmpi slt, %39, %32 : i64
      %46 = arith.andi %44, %45 : i1
      %47:2 = scf.if %46 -> (i32, tensor<128x256xf32>) {
        %50 = arith.muli %41, %c64_i32 : i32
        %51 = tt.splat %50 : i32 -> tensor<64xi32>
        %52 = arith.addi %51, %15 : tensor<64xi32>
        %53 = tt.expand_dims %52 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %54 = tt.broadcast %53 : tensor<1x64xi32> -> tensor<128x64xi32>
        %55 = arith.addi %43#2, %54 : tensor<128x64xi32>
        %56 = tt.addptr %21, %55 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %57 = tt.expand_dims %52 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %58 = tt.broadcast %57 : tensor<64x1xi32> -> tensor<64x256xi32>
        %59 = arith.addi %58, %43#3 : tensor<64x256xi32>
        %60 = tt.addptr %23, %59 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
        %61 = arith.subi %arg5, %50 : i32
        %62 = tt.splat %61 : i32 -> tensor<1x64xi32>
        %63 = arith.cmpi slt, %24, %62 : tensor<1x64xi32>
        %64 = tt.broadcast %63 : tensor<1x64xi1> -> tensor<128x64xi1>
        %65 = tt.load %56, %64, %cst_1 : tensor<128x64x!tt.ptr<f16>>
        %66 = tt.splat %61 : i32 -> tensor<64x1xi32>
        %67 = arith.cmpi slt, %25, %66 : tensor<64x1xi32>
        %68 = tt.broadcast %67 : tensor<64x1xi1> -> tensor<64x256xi1>
        %69 = tt.load %60, %68, %cst_0 : tensor<64x256x!tt.ptr<f16>>
        %70 = tt.dot %65, %69, %42, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
        %71 = arith.addi %41, %c1_i32 : i32
        scf.yield %71, %70 : i32, tensor<128x256xf32>
      } else {
        scf.yield %41, %arg13 : i32, tensor<128x256xf32>
      }
      %48 = arith.cmpi eq, %39, %36 : i64
      %49 = scf.if %48 -> (i32) {
        %50 = tt.expand_dims %43#0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %51 = arith.muli %26, %50 : tensor<128x1xi32>
        %52 = tt.addptr %27, %51 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %53 = tt.expand_dims %43#1 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
        %54 = tt.broadcast %52 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x256x!tt.ptr<f16>>
        %55 = tt.broadcast %53 : tensor<1x256xi32> -> tensor<128x256xi32>
        %56 = tt.addptr %54, %55 : tensor<128x256x!tt.ptr<f16>>, tensor<128x256xi32>
        %57 = arith.cmpi slt, %50, %28 : tensor<128x1xi32>
        %58 = arith.cmpi slt, %53, %29 : tensor<1x256xi32>
        %59 = tt.broadcast %57 : tensor<128x1xi1> -> tensor<128x256xi1>
        %60 = tt.broadcast %58 : tensor<1x256xi1> -> tensor<128x256xi1>
        %61 = arith.andi %59, %60 : tensor<128x256xi1>
        %62 = arith.truncf %47#1 : tensor<128x256xf32> to tensor<128x256xf16>
        tt.store %56, %62, %61 : tensor<128x256x!tt.ptr<f16>>
        %63 = arith.addi %arg11, %c132_i32 : i32
        scf.yield %63 : i32
      } else {
        scf.yield %arg11 : i32
      }
      scf.yield %39, %49, %47#0, %47#1, %43#0, %43#1, %43#2, %43#3 : i64, i32, i32, tensor<128x256xf32>, tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
    }
    tt.return
  }
}

