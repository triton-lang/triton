module {
  func @matmul_kernel__Pfp16_Pfp16_Pfp16_i32_i32_i32_i32_i32_i32__7c1_9c1_11c1_12c128_13c128_14c128_15c8(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = call @"cdiv__i32__1cconstexpr[128]"(%arg3) : (i32) -> i32
    %2 = call @"cdiv__i32__1cconstexpr[128]"(%arg4) : (i32) -> i32
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
    %c128_i32 = arith.constant 128 : i32
    %12 = arith.muli %9, %c128_i32 : i32
    %13 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %14 = tt.broadcast %12 : (i32) -> tensor<128xi32>
    %15 = arith.addi %14, %13 : tensor<128xi32>
    %c128_i32_1 = arith.constant 128 : i32
    %16 = arith.muli %11, %c128_i32_1 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %18 = tt.broadcast %16 : (i32) -> tensor<128xi32>
    %19 = arith.addi %18, %17 : tensor<128xi32>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %21 = tt.reshape %15 : (tensor<128xi32>) -> tensor<128x1xi32>
    %22 = tt.broadcast %arg6 : (i32) -> tensor<128x1xi32>
    %23 = arith.muli %21, %22 : tensor<128x1xi32>
    %24 = tt.reshape %20 : (tensor<128xi32>) -> tensor<1x128xi32>
    %c1_i32 = arith.constant 1 : i32
    %25 = tt.broadcast %c1_i32 : (i32) -> tensor<1x128xi32>
    %26 = arith.muli %24, %25 : tensor<1x128xi32>
    %27 = tt.broadcast %23 : (tensor<128x1xi32>) -> tensor<128x128xi32>
    %28 = tt.broadcast %26 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %29 = arith.addi %27, %28 : tensor<128x128xi32>
    %30 = tt.broadcast %arg0 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>>
    %31 = tt.getelementptr %30, %29, : tensor<128x128x!tt.ptr<f16>>
    %32 = tt.reshape %20 : (tensor<128xi32>) -> tensor<128x1xi32>
    %33 = tt.broadcast %arg7 : (i32) -> tensor<128x1xi32>
    %34 = arith.muli %32, %33 : tensor<128x1xi32>
    %35 = tt.reshape %19 : (tensor<128xi32>) -> tensor<1x128xi32>
    %c1_i32_2 = arith.constant 1 : i32
    %36 = tt.broadcast %c1_i32_2 : (i32) -> tensor<1x128xi32>
    %37 = arith.muli %35, %36 : tensor<1x128xi32>
    %38 = tt.broadcast %34 : (tensor<128x1xi32>) -> tensor<128x128xi32>
    %39 = tt.broadcast %37 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %40 = arith.addi %38, %39 : tensor<128x128xi32>
    %41 = tt.broadcast %arg1 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>>
    %42 = tt.getelementptr %41, %40, : tensor<128x128x!tt.ptr<f16>>
    %cst = arith.constant 0.000000e+00 : f32
    %43 = tt.broadcast %cst : (f32) -> tensor<128x128xf32>
    %c0_i32 = arith.constant 0 : i32
    %c128_i32_3 = arith.constant 128 : i32
    %44 = arith.index_cast %c0_i32 : i32 to index
    %45 = arith.index_cast %arg5 : i32 to index
    %46 = arith.index_cast %c128_i32_3 : i32 to index
    %47:3 = scf.for %arg9 = %44 to %45 step %46 iter_args(%arg10 = %43, %arg11 = %31, %arg12 = %42) -> (tensor<128x128xf32>, tensor<128x128x!tt.ptr<f16>>, tensor<128x128x!tt.ptr<f16>>) {
      %cst_7 = arith.constant dense<true> : tensor<128x128xi1>
      %cst_8 = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
      %77 = tt.load %arg11, %cst_7, %cst_8 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16>
      %cst_9 = arith.constant dense<true> : tensor<128x128xi1>
      %cst_10 = arith.constant dense<0.000000e+00> : tensor<128x128xf16>
      %78 = tt.load %arg12, %cst_9, %cst_10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16>
      %cst_11 = arith.constant 0.000000e+00 : f32
      %79 = tt.broadcast %cst_11 : (f32) -> tensor<128x128xf32>
      %80 = tt.dot %77, %78, %79 {allowTF32 = true} : tensor<128x128xf16> * tensor<128x128xf16> -> tensor<128x128xf32>
      %81 = arith.addf %arg10, %80 : tensor<128x128xf32>
      %c128_i32_12 = arith.constant 128 : i32
      %82 = tt.broadcast %c128_i32_12 : (i32) -> tensor<128x128xi32>
      %83 = tt.getelementptr %arg11, %82, : tensor<128x128x!tt.ptr<f16>>
      %c128_i32_13 = arith.constant 128 : i32
      %84 = arith.muli %arg7, %c128_i32_13 : i32
      %85 = tt.broadcast %84 : (i32) -> tensor<128x128xi32>
      %86 = tt.getelementptr %arg12, %85, : tensor<128x128x!tt.ptr<f16>>
      scf.yield %81, %83, %86 : tensor<128x128xf32>, tensor<128x128x!tt.ptr<f16>>, tensor<128x128x!tt.ptr<f16>>
    }
    %48 = arith.truncf %47#0 : tensor<128x128xf32> to tensor<128x128xf16>
    %c128_i32_4 = arith.constant 128 : i32
    %49 = arith.muli %9, %c128_i32_4 : i32
    %50 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %51 = tt.broadcast %49 : (i32) -> tensor<128xi32>
    %52 = arith.addi %51, %50 : tensor<128xi32>
    %c128_i32_5 = arith.constant 128 : i32
    %53 = arith.muli %11, %c128_i32_5 : i32
    %54 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %55 = tt.broadcast %53 : (i32) -> tensor<128xi32>
    %56 = arith.addi %55, %54 : tensor<128xi32>
    %57 = tt.reshape %52 : (tensor<128xi32>) -> tensor<128x1xi32>
    %58 = tt.broadcast %arg8 : (i32) -> tensor<128x1xi32>
    %59 = arith.muli %58, %57 : tensor<128x1xi32>
    %60 = tt.broadcast %arg2 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>>
    %61 = tt.getelementptr %60, %59, : tensor<128x1x!tt.ptr<f16>>
    %62 = tt.reshape %56 : (tensor<128xi32>) -> tensor<1x128xi32>
    %c1_i32_6 = arith.constant 1 : i32
    %63 = tt.broadcast %c1_i32_6 : (i32) -> tensor<1x128xi32>
    %64 = arith.muli %62, %63 : tensor<1x128xi32>
    %65 = tt.broadcast %61 : (tensor<128x1x!tt.ptr<f16>>) -> tensor<128x128x!tt.ptr<f16>>
    %66 = tt.broadcast %64 : (tensor<1x128xi32>) -> tensor<128x128xi32>
    %67 = tt.getelementptr %65, %66, : tensor<128x128x!tt.ptr<f16>>
    %68 = tt.reshape %52 : (tensor<128xi32>) -> tensor<128x1xi32>
    %69 = tt.broadcast %arg3 : (i32) -> tensor<128x1xi32>
    %70 = arith.cmpi slt, %68, %69 : tensor<128x1xi32>
    %71 = tt.reshape %56 : (tensor<128xi32>) -> tensor<1x128xi32>
    %72 = tt.broadcast %arg4 : (i32) -> tensor<1x128xi32>
    %73 = arith.cmpi slt, %71, %72 : tensor<1x128xi32>
    %74 = tt.broadcast %70 : (tensor<128x1xi1>) -> tensor<128x128xi1>
    %75 = tt.broadcast %73 : (tensor<1x128xi1>) -> tensor<128x128xi1>
    %76 = arith.andi %74, %75 : tensor<128x128xi1>
    tt.store %67, %48, %76, : tensor<128x128xf16>
    return
  }
  func @"cdiv__i32__1cconstexpr[128]"(%arg0: i32) -> i32 {
    %c128_i32 = arith.constant 128 : i32
    %0 = arith.addi %arg0, %c128_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %1 = arith.subi %0, %c1_i32 : i32
    %c128_i32_0 = arith.constant 128 : i32
    %2 = arith.divsi %1, %c128_i32_0 : i32
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
  func @matmul_kernel__Pfp16_Pfp16_Pfp16_i32_i32_i32_i32_i32_i32__7c1_9c1_11c1_12c128_13c128_14c128_15c8(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %cst_0 = arith.constant dense<true> : tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id {axis = 0 : i32} : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
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
    %15 = arith.muli %12, %c128_i32 : i32
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %17 = tt.broadcast %15 : (i32) -> tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %18 = arith.addi %17, %16 : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %19 = arith.muli %14, %c128_i32 : i32
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %21 = tt.broadcast %19 : (i32) -> tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %22 = arith.addi %21, %20 : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %24 = tt.reshape %18 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %25 = tt.broadcast %arg6 : (i32) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %26 = arith.muli %24, %25 : tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %27 = tt.reshape %23 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %28 = tt.broadcast %c1_i32 : (i32) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %29 = arith.muli %27, %28 : tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %30 = tt.broadcast %26 : (tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %31 = tt.broadcast %29 : (tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %32 = arith.addi %30, %31 : tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %33 = tt.broadcast %arg0 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %34 = tt.getelementptr %33, %32, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %35 = tt.reshape %23 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %36 = tt.broadcast %arg7 : (i32) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %37 = arith.muli %35, %36 : tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %38 = tt.reshape %22 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %39 = tt.broadcast %c1_i32 : (i32) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %40 = arith.muli %38, %39 : tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %41 = tt.broadcast %37 : (tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %42 = tt.broadcast %40 : (tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %43 = arith.addi %41, %42 : tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %44 = tt.broadcast %arg1 : (!tt.ptr<f16>) -> tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %45 = tt.getelementptr %44, %43, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %46 = tt.broadcast %cst : (f32) -> tensor<128x128xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %47 = arith.index_cast %arg5 : i32 to index
    %48 = "triton_gpu.copy_async"(%34, %cst_0, %cst_1) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>
    %49 = "triton_gpu.copy_async"(%45, %cst_0, %cst_1) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>
    %50 = tt.broadcast %c128_i32 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %51 = tt.getelementptr %34, %50, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %52 = arith.muli %arg7, %c128_i32 : i32
    %53 = tt.broadcast %52 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %54 = tt.getelementptr %45, %53, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %55:8 = scf.for %arg9 = %c0 to %47 step %c128 iter_args(%arg10 = %46, %arg11 = %34, %arg12 = %45, %arg13 = %48, %arg14 = %49, %arg15 = %51, %arg16 = %54, %arg17 = %c0) -> (tensor<128x128xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>, tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, index) {
      %85 = tt.dot %arg13, %arg14, %arg10 {allowTF32 = true} : tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">> * tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">> -> tensor<128x128xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %86 = tt.broadcast %c128_i32 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %87 = tt.getelementptr %arg11, %86, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %88 = arith.muli %arg7, %c128_i32 : i32
      %89 = tt.broadcast %88 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %90 = tt.getelementptr %arg12, %89, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %91 = arith.addi %arg17, %c128 : index
      %92 = arith.cmpi slt, %91, %47 : index
      %93 = tt.broadcast %92 : (i1) -> tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %94 = tt.broadcast %92 : (i1) -> tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %95 = arith.andi %94, %93 : tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %96 = "triton_gpu.copy_async"(%arg15, %93, %cst_1) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>
      %97 = "triton_gpu.copy_async"(%arg16, %95, %cst_1) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>
      %98 = tt.broadcast %c128_i32 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %99 = tt.getelementptr %arg15, %98, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %100 = arith.muli %arg7, %c128_i32 : i32
      %101 = tt.broadcast %100 : (i32) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      %102 = tt.getelementptr %arg16, %101, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
      scf.yield %85, %87, %90, %96, %97, %99, %102, %91 : tensor<128x128xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>, tensor<128x128xf16, #triton_gpu<"shared (memory) encoding<>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, index
    }
    %56 = arith.truncf %55#0 : tensor<128x128xf32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">> to tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %57 = arith.muli %12, %c128_i32 : i32
    %58 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %59 = tt.broadcast %57 : (i32) -> tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %60 = arith.addi %59, %58 : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %61 = arith.muli %14, %c128_i32 : i32
    %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %63 = tt.broadcast %61 : (i32) -> tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %64 = arith.addi %63, %62 : tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>
    %65 = tt.reshape %60 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %66 = tt.broadcast %arg8 : (i32) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %67 = arith.muli %66, %65 : tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %68 = tt.broadcast %arg2 : (!tt.ptr<f16>) -> tensor<128x1x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %69 = tt.getelementptr %68, %67, : tensor<128x1x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %70 = tt.reshape %64 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %71 = tt.broadcast %c1_i32 : (i32) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %72 = arith.muli %70, %71 : tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %73 = tt.broadcast %69 : (tensor<128x1x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %74 = tt.broadcast %72 : (tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>) -> tensor<128x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %75 = tt.getelementptr %73, %74, : tensor<128x128x!tt.ptr<f16>, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %76 = tt.reshape %60 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %77 = tt.broadcast %arg3 : (i32) -> tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %78 = "triton_gpu.cmpi"(%76, %77) {predicate = 2 : i64} : (tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>, tensor<128x1xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x1xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %79 = tt.reshape %64 : (tensor<128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, blockTileSize = 32, order = 0>">>) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %80 = tt.broadcast %arg4 : (i32) -> tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %81 = "triton_gpu.cmpi"(%79, %80) {predicate = 2 : i64} : (tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>, tensor<1x128xi32, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>) -> tensor<1x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>
    %82 = tt.broadcast %78 : (tensor<128x1xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>) -> tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %83 = tt.broadcast %81 : (tensor<1x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 1, 32, order = 0, 1>">>) -> tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    %84 = arith.andi %82, %83 : tensor<128x128xi1, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    tt.store %75, %56, %84, : tensor<128x128xf16, #triton_gpu<"coalesced encoding<threadTileSize = 1, 1, blockTileSize = 32, 1, order = 0, 1>">>
    return
  }
  func @"cdiv__i32__1cconstexpr[128]"(%arg0: i32) -> i32 {
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = arith.addi %arg0, %c127_i32 : i32
    %1 = arith.divsi %0, %c128_i32 : i32
    return %1 : i32
  }
  func @"minimum__i32__1cconstexpr[8]"(%arg0: i32) -> i32 {
    %c8_i32 = arith.constant 8 : i32
    %0 = arith.cmpi slt, %arg0, %c8_i32 : i32
    %1 = select %0, %arg0, %c8_i32 : i32
    return %1 : i32
  }
}
