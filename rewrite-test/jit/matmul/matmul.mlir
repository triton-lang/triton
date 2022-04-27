module {
  func @matmul_kernel__Pfp16_Pfp16_Pfp16_i32_i32_i32_i32_i32_i32__7c1_9c1_11c1_12c64_13c64_14c32_15c8(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = call @"cdiv__i32__1cconstexpr[64]"(%arg3) : (i32) -> i32
    %2 = call @"cdiv__i32__1cconstexpr[64]"(%arg4) : (i32) -> i32
    %c8_i32 = arith.constant 8 : i32
    %3 = arith.muli %2, %c8_i32 : i32
    %4 = arith.divsi %0, %3 : i32
    %c8_i32_0 = arith.constant 8 : i32
    %5 = arith.muli %4, %c8_i32_0 : i32
    %6 = arith.subi %1, %5 : i32
    %7 = call @"minimum__i32__1cconstexpr[8]"(%6) : (i32) -> i32
    %8 = arith.remsi %0, %7 : i32
    %9 = arith.addi %5, %8 : i32
    %10 = arith.remsi %0, %3 : i32
    %11 = arith.divsi %10, %7 : i32
    %c64_i32 = arith.constant 64 : i32
    %12 = arith.muli %9, %c64_i32 : i32
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %14 = tt.broadcast %12 : (i32) -> tensor<64xi32>
    %15 = arith.addi %14, %13 : tensor<64xi32>
    %c64_i32_1 = arith.constant 64 : i32
    %16 = arith.muli %11, %c64_i32_1 : i32
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %18 = tt.broadcast %16 : (i32) -> tensor<64xi32>
    %19 = arith.addi %18, %17 : tensor<64xi32>
    %20 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %21 = tt.reshape %15 : (tensor<64xi32>) -> tensor<64x1xi32>
    %22 = tt.broadcast %arg6 : (i32) -> tensor<64x1xi32>
    %23 = arith.muli %21, %22 : tensor<64x1xi32>
    %24 = tt.reshape %20 : (tensor<32xi32>) -> tensor<1x32xi32>
    %c1_i32 = arith.constant 1 : i32
    %25 = tt.broadcast %c1_i32 : (i32) -> tensor<1x32xi32>
    %26 = arith.muli %24, %25 : tensor<1x32xi32>
    %27 = tt.broadcast %23 : (tensor<64x1xi32>) -> tensor<64x32xi32>
    %28 = tt.broadcast %26 : (tensor<1x32xi32>) -> tensor<64x32xi32>
    %29 = arith.addi %27, %28 : tensor<64x32xi32>
    %30 = tt.broadcast %arg0 : (!tt.ptr<f16>) -> tensor<64x32x!tt.ptr<f16>>
    %31 = tt.getelementptr %30, %29, : tensor<64x32x!tt.ptr<f16>>
    %32 = tt.reshape %20 : (tensor<32xi32>) -> tensor<32x1xi32>
    %33 = tt.broadcast %arg7 : (i32) -> tensor<32x1xi32>
    %34 = arith.muli %32, %33 : tensor<32x1xi32>
    %35 = tt.reshape %19 : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_2 = arith.constant 1 : i32
    %36 = tt.broadcast %c1_i32_2 : (i32) -> tensor<1x64xi32>
    %37 = arith.muli %35, %36 : tensor<1x64xi32>
    %38 = tt.broadcast %34 : (tensor<32x1xi32>) -> tensor<32x64xi32>
    %39 = tt.broadcast %37 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %40 = arith.addi %38, %39 : tensor<32x64xi32>
    %41 = tt.broadcast %arg1 : (!tt.ptr<f16>) -> tensor<32x64x!tt.ptr<f16>>
    %42 = tt.getelementptr %41, %40, : tensor<32x64x!tt.ptr<f16>>
    %cst = arith.constant 0.000000e+00 : f32
    %43 = tt.broadcast %cst : (f32) -> tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %44 = arith.index_cast %c0_i32 : i32 to index
    %45 = arith.index_cast %arg5 : i32 to index
    %46 = arith.index_cast %c32_i32 : i32 to index
    %47:3 = scf.for %arg9 = %44 to %45 step %46 iter_args(%arg10 = %43, %arg11 = %31, %arg12 = %42) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>) {
      %cst_6 = arith.constant dense<true> : tensor<64x32xi1>
      %cst_7 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
      %77 = tt.load %arg11, %cst_6, %cst_7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf16>
      %cst_8 = arith.constant dense<true> : tensor<32x64xi1>
      %cst_9 = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
      %78 = tt.load %arg12, %cst_8, %cst_9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16>
      %cst_10 = arith.constant 0.000000e+00 : f32
      %79 = tt.broadcast %cst_10 : (f32) -> tensor<64x64xf32>
      %80 = tt.dot %77, %78, %79 {allowTF32 = true} : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
      %81 = arith.addf %arg10, %80 : tensor<64x64xf32>
      %c32_i32_11 = arith.constant 32 : i32
      %82 = tt.broadcast %c32_i32_11 : (i32) -> tensor<64x32xi32>
      %83 = tt.getelementptr %arg11, %82, : tensor<64x32x!tt.ptr<f16>>
      %c32_i32_12 = arith.constant 32 : i32
      %84 = arith.muli %arg7, %c32_i32_12 : i32
      %85 = tt.broadcast %84 : (i32) -> tensor<32x64xi32>
      %86 = tt.getelementptr %arg12, %85, : tensor<32x64x!tt.ptr<f16>>
      scf.yield %81, %83, %86 : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>
    }
    %48 = arith.truncf %47#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %c64_i32_3 = arith.constant 64 : i32
    %49 = arith.muli %9, %c64_i32_3 : i32
    %50 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %51 = tt.broadcast %49 : (i32) -> tensor<64xi32>
    %52 = arith.addi %51, %50 : tensor<64xi32>
    %c64_i32_4 = arith.constant 64 : i32
    %53 = arith.muli %11, %c64_i32_4 : i32
    %54 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %55 = tt.broadcast %53 : (i32) -> tensor<64xi32>
    %56 = arith.addi %55, %54 : tensor<64xi32>
    %57 = tt.reshape %52 : (tensor<64xi32>) -> tensor<64x1xi32>
    %58 = tt.broadcast %arg8 : (i32) -> tensor<64x1xi32>
    %59 = arith.muli %58, %57 : tensor<64x1xi32>
    %60 = tt.broadcast %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>>
    %61 = tt.getelementptr %60, %59, : tensor<64x1x!tt.ptr<f16>>
    %62 = tt.reshape %56 : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_5 = arith.constant 1 : i32
    %63 = tt.broadcast %c1_i32_5 : (i32) -> tensor<1x64xi32>
    %64 = arith.muli %62, %63 : tensor<1x64xi32>
    %65 = tt.broadcast %61 : (tensor<64x1x!tt.ptr<f16>>) -> tensor<64x64x!tt.ptr<f16>>
    %66 = tt.broadcast %64 : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %67 = tt.getelementptr %65, %66, : tensor<64x64x!tt.ptr<f16>>
    %68 = tt.reshape %52 : (tensor<64xi32>) -> tensor<64x1xi32>
    %69 = tt.broadcast %arg3 : (i32) -> tensor<64x1xi32>
    %70 = arith.cmpi slt, %68, %69 : tensor<64x1xi32>
    %71 = tt.reshape %56 : (tensor<64xi32>) -> tensor<1x64xi32>
    %72 = tt.broadcast %arg4 : (i32) -> tensor<1x64xi32>
    %73 = arith.cmpi slt, %71, %72 : tensor<1x64xi32>
    %74 = tt.broadcast %70 : (tensor<64x1xi1>) -> tensor<64x64xi1>
    %75 = tt.broadcast %73 : (tensor<1x64xi1>) -> tensor<64x64xi1>
    %76 = arith.andi %74, %75 : tensor<64x64xi1>
    tt.store %67, %48, %76, : tensor<64x64xf16>
    return
  }
  func @"cdiv__i32__1cconstexpr[64]"(%arg0: i32) -> i32 {
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg0, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %2 = arith.divsi %1, %c64_i32_0 : i32
    return %2 : i32
  }
  func @"minimum__i32__1cconstexpr[8]"(%arg0: i32) -> i32 {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.cmpi slt, %arg0, %c8_i32 : i32
    %c8_i32_0 = arith.constant 8 : i32
    %1 = select %0, %arg0, %c8_i32_0 : i32
    return %1 : i32
  }
}
module {
  func @matmul_kernel__Pfp16_Pfp16_Pfp16_i32_i32_i32_i32_i32_i32__7c1_9c1_11c1_12c64_13c64_14c32_15c8(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
    %cst_0 = arith.constant dense<true> : tensor<32x64xi1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
    %cst_2 = arith.constant dense<true> : tensor<64x32xi1>
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.cmpi slt, %8, %c8_i32 : i32
    %10 = select %9, %8, %c8_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %17 = tt.broadcast %15 : (i32) -> tensor<64xi32>
    %18 = arith.addi %17, %16 : tensor<64xi32>
    %19 = arith.muli %14, %c64_i32 : i32
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %21 = tt.broadcast %19 : (i32) -> tensor<64xi32>
    %22 = arith.addi %21, %20 : tensor<64xi32>
    %23 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %24 = tt.reshape %18 : (tensor<64xi32>) -> tensor<64x1xi32>
    %25 = tt.broadcast %arg6 : (i32) -> tensor<64x1xi32>
    %26 = arith.muli %24, %25 : tensor<64x1xi32>
    %27 = tt.reshape %23 : (tensor<32xi32>) -> tensor<1x32xi32>
    %28 = tt.broadcast %c1_i32 : (i32) -> tensor<1x32xi32>
    %29 = arith.muli %27, %28 : tensor<1x32xi32>
    %30 = tt.broadcast %26 : (tensor<64x1xi32>) -> tensor<64x32xi32>
    %31 = tt.broadcast %29 : (tensor<1x32xi32>) -> tensor<64x32xi32>
    %32 = arith.addi %30, %31 : tensor<64x32xi32>
    %33 = tt.broadcast %arg0 : (!tt.ptr<f16>) -> tensor<64x32x!tt.ptr<f16>>
    %34 = tt.getelementptr %33, %32, : tensor<64x32x!tt.ptr<f16>>
    %35 = tt.reshape %23 : (tensor<32xi32>) -> tensor<32x1xi32>
    %36 = tt.broadcast %arg7 : (i32) -> tensor<32x1xi32>
    %37 = arith.muli %35, %36 : tensor<32x1xi32>
    %38 = tt.reshape %22 : (tensor<64xi32>) -> tensor<1x64xi32>
    %39 = tt.broadcast %c1_i32 : (i32) -> tensor<1x64xi32>
    %40 = arith.muli %38, %39 : tensor<1x64xi32>
    %41 = tt.broadcast %37 : (tensor<32x1xi32>) -> tensor<32x64xi32>
    %42 = tt.broadcast %40 : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %43 = arith.addi %41, %42 : tensor<32x64xi32>
    %44 = tt.broadcast %arg1 : (!tt.ptr<f16>) -> tensor<32x64x!tt.ptr<f16>>
    %45 = tt.getelementptr %44, %43, : tensor<32x64x!tt.ptr<f16>>
    %46 = tt.broadcast %cst_3 : (f32) -> tensor<64x64xf32>
    %47 = arith.index_cast %arg5 : i32 to index
    %48:3 = scf.for %arg9 = %c0 to %47 step %c32 iter_args(%arg10 = %46, %arg11 = %34, %arg12 = %45) -> (tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>) {
      %78 = tt.load %arg11, %cst_2, %cst_1 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf16>
      %79 = tt.load %arg12, %cst_0, %cst {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16>
      %80 = tt.broadcast %cst_3 : (f32) -> tensor<64x64xf32>
      %81 = tt.dot %78, %79, %80 {allowTF32 = true} : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
      %82 = arith.addf %arg10, %81 : tensor<64x64xf32>
      %83 = tt.broadcast %c32_i32 : (i32) -> tensor<64x32xi32>
      %84 = tt.getelementptr %arg11, %83, : tensor<64x32x!tt.ptr<f16>>
      %85 = arith.muli %arg7, %c32_i32 : i32
      %86 = tt.broadcast %85 : (i32) -> tensor<32x64xi32>
      %87 = tt.getelementptr %arg12, %86, : tensor<32x64x!tt.ptr<f16>>
      scf.yield %82, %84, %87 : tensor<64x64xf32>, tensor<64x32x!tt.ptr<f16>>, tensor<32x64x!tt.ptr<f16>>
    }
    %49 = arith.truncf %48#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %50 = arith.muli %12, %c64_i32 : i32
    %51 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %52 = tt.broadcast %50 : (i32) -> tensor<64xi32>
    %53 = arith.addi %52, %51 : tensor<64xi32>
    %54 = arith.muli %14, %c64_i32 : i32
    %55 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %56 = tt.broadcast %54 : (i32) -> tensor<64xi32>
    %57 = arith.addi %56, %55 : tensor<64xi32>
    %58 = tt.reshape %53 : (tensor<64xi32>) -> tensor<64x1xi32>
    %59 = tt.broadcast %arg8 : (i32) -> tensor<64x1xi32>
    %60 = arith.muli %59, %58 : tensor<64x1xi32>
    %61 = tt.broadcast %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>>
    %62 = tt.getelementptr %61, %60, : tensor<64x1x!tt.ptr<f16>>
    %63 = tt.reshape %57 : (tensor<64xi32>) -> tensor<1x64xi32>
    %64 = tt.broadcast %c1_i32 : (i32) -> tensor<1x64xi32>
    %65 = arith.muli %63, %64 : tensor<1x64xi32>
    %66 = tt.broadcast %62 : (tensor<64x1x!tt.ptr<f16>>) -> tensor<64x64x!tt.ptr<f16>>
    %67 = tt.broadcast %65 : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %68 = tt.getelementptr %66, %67, : tensor<64x64x!tt.ptr<f16>>
    %69 = tt.reshape %53 : (tensor<64xi32>) -> tensor<64x1xi32>
    %70 = tt.broadcast %arg3 : (i32) -> tensor<64x1xi32>
    %71 = arith.cmpi slt, %69, %70 : tensor<64x1xi32>
    %72 = tt.reshape %57 : (tensor<64xi32>) -> tensor<1x64xi32>
    %73 = tt.broadcast %arg4 : (i32) -> tensor<1x64xi32>
    %74 = arith.cmpi slt, %72, %73 : tensor<1x64xi32>
    %75 = tt.broadcast %71 : (tensor<64x1xi1>) -> tensor<64x64xi1>
    %76 = tt.broadcast %74 : (tensor<1x64xi1>) -> tensor<64x64xi1>
    %77 = arith.andi %75, %76 : tensor<64x64xi1>
    tt.store %68, %49, %77, : tensor<64x64xf16>
    return
  }
  func @"cdiv__i32__1cconstexpr[64]"(%arg0: i32) -> i32 {
    %c63_i32 = arith.constant 63 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = arith.addi %arg0, %c63_i32 : i32
    %1 = arith.divsi %0, %c64_i32 : i32
    return %1 : i32
  }
  func @"minimum__i32__1cconstexpr[8]"(%arg0: i32) -> i32 {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.cmpi slt, %arg0, %c8_i32 : i32
    %1 = select %0, %arg0, %c8_i32 : i32
    return %1 : i32
  }
}
