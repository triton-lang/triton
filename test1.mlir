module {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf16>
    %0 = ub.poison : tensor<64x256xi32>
    %1 = ub.poison : tensor<128x64xi32>
    %2 = ub.poison : tensor<256xi32>
    %3 = ub.poison : tensor<128xi32>
    %4 = ub.poison : tensor<128x256xf32>
    %5 = ub.poison : i32
    %c-1_i64 = arith.constant -1 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x256xf16>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x64xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c132_i32 = arith.constant 132 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_3 = arith.constant dense<0> : tensor<256xi32>
    %cst_4 = arith.constant dense<0> : tensor<128xi32>
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
    %30 = arith.cmpi eq, %12, %c0_i32 : i32
    scf.if %30 {
      scf.for %arg9 = %6 to %13 step %c132_i32  : i32 {
        %31 = arith.divsi %arg9, %14 : i32
        %32 = arith.muli %31, %c8_i32 : i32
        %33 = arith.subi %8, %32 : i32
        %34 = arith.minsi %33, %c8_i32 : i32
        %35 = arith.remsi %arg9, %34 : i32
        %36 = arith.addi %32, %35 : i32
        %37 = arith.remsi %arg9, %14 : i32
        %38 = arith.divsi %37, %34 : i32
        %39 = arith.muli %36, %c128_i32 : i32
        %40 = arith.muli %38, %c256_i32 : i32
        %41 = tt.splat %39 : i32 -> tensor<128xi32>
        %42 = arith.addi %41, %16 : tensor<128xi32>
        %43 = tt.splat %40 : i32 -> tensor<256xi32>
        %44 = arith.addi %43, %17 : tensor<256xi32>
        %45 = tt.expand_dims %42 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %46 = arith.muli %26, %45 : tensor<128x1xi32>
        %47 = tt.addptr %27, %46 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
        %48 = tt.expand_dims %44 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
        %49 = tt.broadcast %47 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x256x!tt.ptr<f16>>
        %50 = tt.broadcast %48 : tensor<1x256xi32> -> tensor<128x256xi32>
        %51 = tt.addptr %49, %50 : tensor<128x256x!tt.ptr<f16>>, tensor<128x256xi32>
        %52 = arith.cmpi slt, %45, %28 : tensor<128x1xi32>
        %53 = arith.cmpi slt, %48, %29 : tensor<1x256xi32>
        %54 = tt.broadcast %52 : tensor<128x1xi1> -> tensor<128x256xi1>
        %55 = tt.broadcast %53 : tensor<1x256xi1> -> tensor<128x256xi1>
        %56 = arith.andi %54, %55 : tensor<128x256xi1>
        tt.store %51, %cst, %56 : tensor<128x256x!tt.ptr<f16>>
      }
    } else {
      %31 = arith.subi %13, %6 : i32
      %32 = arith.ceildivsi %31, %c132_i32 : i32
      %33 = arith.extsi %12 : i32 to i64
      %34 = arith.maxsi %33, %c1_i64 : i64
      %35 = arith.extsi %32 : i32 to i64
      %36 = arith.muli %35, %34 : i64
      %37 = arith.subi %34, %c1_i64 : i64
      %38:8 = scf.for %arg9 = %c0_i64 to %36 step %c1_i64 iter_args(%arg10 = %c-1_i64, %arg11 = %6, %arg12 = %5, %arg13 = %4, %arg14 = %3, %arg15 = %2, %arg16 = %1, %arg17 = %0) -> (i64, i32, i32, tensor<128x256xf32>, tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>)  : i64 {
        %39 = arith.addi %arg10, %c1_i64 : i64
        %40 = arith.remsi %39, %34 : i64
        %41 = arith.cmpi eq, %40, %c0_i64 : i64
        %42 = arith.select %41, %c0_i32, %arg12 : i32
        %43 = arith.select %41, %cst_0, %arg13 : tensor<128x256xf32>
        %44:4 = scf.if %41 -> (tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>) {
          %69 = arith.divsi %arg11, %14 : i32
          %70 = arith.muli %69, %c8_i32 : i32
          %71 = arith.subi %8, %70 : i32
          %72 = arith.minsi %71, %c8_i32 : i32
          %73 = arith.remsi %arg11, %72 : i32
          %74 = arith.addi %70, %73 : i32
          %75 = arith.remsi %arg11, %14 : i32
          %76 = arith.divsi %75, %72 : i32
          %77 = arith.muli %74, %c128_i32 : i32
          %78 = arith.muli %76, %c256_i32 : i32
          %79 = tt.splat %77 : i32 -> tensor<128xi32>
          %80 = arith.addi %79, %16 : tensor<128xi32>
          %81 = tt.splat %78 : i32 -> tensor<256xi32>
          %82 = arith.addi %81, %17 : tensor<256xi32>
          %83 = arith.cmpi slt, %80, %18 : tensor<128xi32>
          %84 = arith.select %83, %80, %cst_4 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1>, tensor<128xi32>
          %85 = arith.cmpi slt, %82, %19 : tensor<256xi32>
          %86 = arith.select %85, %82, %cst_3 {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1>, tensor<256xi32>
          %87 = tt.expand_dims %84 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %88 = arith.muli %87, %20 : tensor<128x1xi32>
          %89 = tt.broadcast %88 : tensor<128x1xi32> -> tensor<128x64xi32>
          %90 = tt.expand_dims %86 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
          %91 = arith.muli %90, %22 : tensor<1x256xi32>
          %92 = tt.broadcast %91 : tensor<1x256xi32> -> tensor<64x256xi32>
          scf.yield %80, %82, %89, %92 : tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
        } else {
          scf.yield %arg14, %arg15, %arg16, %arg17 : tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
        }
        %45 = arith.muli %42, %c64_i32 : i32
        %46 = tt.splat %45 : i32 -> tensor<64xi32>
        %47 = arith.addi %46, %15 : tensor<64xi32>
        %48 = tt.expand_dims %47 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
        %49 = tt.broadcast %48 : tensor<1x64xi32> -> tensor<128x64xi32>
        %50 = arith.addi %44#2, %49 : tensor<128x64xi32>
        %51 = tt.addptr %21, %50 : tensor<128x64x!tt.ptr<f16>>, tensor<128x64xi32>
        %52 = tt.expand_dims %47 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
        %53 = tt.broadcast %52 : tensor<64x1xi32> -> tensor<64x256xi32>
        %54 = arith.addi %53, %44#3 : tensor<64x256xi32>
        %55 = tt.addptr %23, %54 : tensor<64x256x!tt.ptr<f16>>, tensor<64x256xi32>
        %56 = arith.subi %arg5, %45 : i32
        %57 = tt.splat %56 : i32 -> tensor<1x64xi32>
        %58 = arith.cmpi slt, %24, %57 : tensor<1x64xi32>
        %59 = tt.broadcast %58 : tensor<1x64xi1> -> tensor<128x64xi1>
        %60 = tt.load %51, %59, %cst_2 : tensor<128x64x!tt.ptr<f16>>
        %61 = tt.splat %56 : i32 -> tensor<64x1xi32>
        %62 = arith.cmpi slt, %25, %61 : tensor<64x1xi32>
        %63 = tt.broadcast %62 : tensor<64x1xi1> -> tensor<64x256xi1>
        %64 = tt.load %55, %63, %cst_1 : tensor<64x256x!tt.ptr<f16>>
        %65 = tt.dot %60, %64, %43, inputPrecision = tf32 : tensor<128x64xf16> * tensor<64x256xf16> -> tensor<128x256xf32>
        %66 = arith.addi %42, %c1_i32 : i32
        %67 = arith.cmpi eq, %40, %37 : i64
        %68 = scf.if %67 -> (i32) {
          %69 = tt.expand_dims %44#0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
          %70 = arith.muli %26, %69 : tensor<128x1xi32>
          %71 = tt.addptr %27, %70 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
          %72 = tt.expand_dims %44#1 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
          %73 = tt.broadcast %71 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x256x!tt.ptr<f16>>
          %74 = tt.broadcast %72 : tensor<1x256xi32> -> tensor<128x256xi32>
          %75 = tt.addptr %73, %74 : tensor<128x256x!tt.ptr<f16>>, tensor<128x256xi32>
          %76 = arith.cmpi slt, %69, %28 : tensor<128x1xi32>
          %77 = arith.cmpi slt, %72, %29 : tensor<1x256xi32>
          %78 = tt.broadcast %76 : tensor<128x1xi1> -> tensor<128x256xi1>
          %79 = tt.broadcast %77 : tensor<1x256xi1> -> tensor<128x256xi1>
          %80 = arith.andi %78, %79 : tensor<128x256xi1>
          %81 = arith.truncf %65 : tensor<128x256xf32> to tensor<128x256xf16>
          tt.store %75, %81, %80 : tensor<128x256x!tt.ptr<f16>>
          %82 = arith.addi %arg11, %c132_i32 : i32
          scf.yield %82 : i32
        } else {
          scf.yield %arg11 : i32
        }
        scf.yield %40, %68, %66, %65, %44#0, %44#1, %44#2, %44#3 : i64, i32, i32, tensor<128x256xf32>, tensor<128xi32>, tensor<256xi32>, tensor<128x64xi32>, tensor<64x256xi32>
      }
    }
    tt.return
  }
}

