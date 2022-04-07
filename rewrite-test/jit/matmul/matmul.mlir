module {
  func @matmul_kernel(%arg0: !triton.ptr<f16>, %arg1: !triton.ptr<f16>, %arg2: !triton.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
    %0 = triton.get_program_id {axis = 0 : i32} : i32
    %c64_i32 = arith.constant 64 : i32
    %1 = arith.addi %arg3, %c64_i32 : i32
    %c1_i32 = arith.constant 1 : i32
    %2 = arith.subi %1, %c1_i32 : i32
    %c64_i32_0 = arith.constant 64 : i32
    %3 = arith.divsi %2, %c64_i32_0 : i32
    %c64_i32_1 = arith.constant 64 : i32
    %4 = arith.addi %arg4, %c64_i32_1 : i32
    %c1_i32_2 = arith.constant 1 : i32
    %5 = arith.subi %4, %c1_i32_2 : i32
    %c64_i32_3 = arith.constant 64 : i32
    %6 = arith.divsi %5, %c64_i32_3 : i32
    %c8_i32 = arith.constant 8 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %c8_i32_4 = arith.constant 8 : i32
    %9 = arith.muli %8, %c8_i32_4 : i32
    %10 = arith.subi %3, %9 : i32
    %c8_i32_5 = arith.constant 8 : i32
    %11 = arith.cmpi slt, %10, %c8_i32_5 : i32
    %c8_i32_6 = arith.constant 8 : i32
    %12 = select %11, %10, %c8_i32_6 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %c64_i32_7 = arith.constant 64 : i32
    %17 = arith.muli %14, %c64_i32_7 : i32
    %18 = triton.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %19 = "triton.broadcast"(%17) : (i32) -> tensor<64xi32>
    %20 = arith.addi %19, %18 : tensor<64xi32>
    %c64_i32_8 = arith.constant 64 : i32
    %21 = arith.muli %16, %c64_i32_8 : i32
    %22 = triton.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %23 = "triton.broadcast"(%21) : (i32) -> tensor<64xi32>
    %24 = arith.addi %23, %22 : tensor<64xi32>
    %25 = triton.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %26 = "triton.reshape"(%20) : (tensor<64xi32>) -> tensor<64x1xi32>
    %27 = "triton.broadcast"(%arg6) : (i32) -> tensor<64x1xi32>
    %28 = arith.muli %26, %27 : tensor<64x1xi32>
    %29 = "triton.reshape"(%25) : (tensor<32xi32>) -> tensor<1x32xi32>
    %c1_i32_9 = arith.constant 1 : i32
    %30 = "triton.broadcast"(%c1_i32_9) : (i32) -> tensor<1x32xi32>
    %31 = arith.muli %29, %30 : tensor<1x32xi32>
    %32 = "triton.broadcast"(%28) : (tensor<64x1xi32>) -> tensor<64x32xi32>
    %33 = "triton.broadcast"(%31) : (tensor<1x32xi32>) -> tensor<64x32xi32>
    %34 = arith.addi %32, %33 : tensor<64x32xi32>
    %35 = "triton.broadcast"(%arg0) : (!triton.ptr<f16>) -> tensor<64x32x!triton.ptr<f16>>
    %36 = "triton.getelementptr"(%35, %34) : (tensor<64x32x!triton.ptr<f16>>, tensor<64x32xi32>) -> tensor<64x32x!triton.ptr<f16>>
    %37 = "triton.reshape"(%25) : (tensor<32xi32>) -> tensor<32x1xi32>
    %38 = "triton.broadcast"(%arg7) : (i32) -> tensor<32x1xi32>
    %39 = arith.muli %37, %38 : tensor<32x1xi32>
    %40 = "triton.reshape"(%24) : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_10 = arith.constant 1 : i32
    %41 = "triton.broadcast"(%c1_i32_10) : (i32) -> tensor<1x64xi32>
    %42 = arith.muli %40, %41 : tensor<1x64xi32>
    %43 = "triton.broadcast"(%39) : (tensor<32x1xi32>) -> tensor<32x64xi32>
    %44 = "triton.broadcast"(%42) : (tensor<1x64xi32>) -> tensor<32x64xi32>
    %45 = arith.addi %43, %44 : tensor<32x64xi32>
    %46 = "triton.broadcast"(%arg1) : (!triton.ptr<f16>) -> tensor<32x64x!triton.ptr<f16>>
    %47 = "triton.getelementptr"(%46, %45) : (tensor<32x64x!triton.ptr<f16>>, tensor<32x64xi32>) -> tensor<32x64x!triton.ptr<f16>>
    %cst = arith.constant 0.000000e+00 : f32
    %48 = "triton.broadcast"(%cst) : (f32) -> tensor<64x64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %49 = arith.index_cast %c0_i32 : i32 to index
    %50 = arith.index_cast %arg5 : i32 to index
    %51 = arith.index_cast %c32_i32 : i32 to index
    %52:3 = scf.for %arg9 = %49 to %50 step %51 iter_args(%arg10 = %48, %arg11 = %36, %arg12 = %47) -> (tensor<64x64xf32>, tensor<64x32x!triton.ptr<f16>>, tensor<32x64x!triton.ptr<f16>>) {
      %cst_14 = arith.constant dense<true> : tensor<64x32xi1>
      %cst_15 = arith.constant dense<0.000000e+00> : tensor<64x32xf16>
      %82 = "triton.load"(%arg11, %cst_14, %cst_15) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<64x32x!triton.ptr<f16>>, tensor<64x32xi1>, tensor<64x32xf16>) -> tensor<64x32xf16>
      %cst_16 = arith.constant dense<true> : tensor<32x64xi1>
      %cst_17 = arith.constant dense<0.000000e+00> : tensor<32x64xf16>
      %83 = "triton.load"(%arg12, %cst_16, %cst_17) {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : (tensor<32x64x!triton.ptr<f16>>, tensor<32x64xi1>, tensor<32x64xf16>) -> tensor<32x64xf16>
      %cst_18 = arith.constant 0.000000e+00 : f32
      %84 = "triton.broadcast"(%cst_18) : (f32) -> tensor<64x64xf32>
      %85 = "triton.dot"(%82, %83, %84) {allowTF32 = true} : (tensor<64x32xf16>, tensor<32x64xf16>, tensor<64x64xf32>) -> tensor<64x64xf32>
      %86 = arith.addf %arg10, %85 : tensor<64x64xf32>
      %c32_i32_19 = arith.constant 32 : i32
      %87 = "triton.broadcast"(%c32_i32_19) : (i32) -> tensor<64x32xi32>
      %88 = "triton.getelementptr"(%arg11, %87) : (tensor<64x32x!triton.ptr<f16>>, tensor<64x32xi32>) -> tensor<64x32x!triton.ptr<f16>>
      %c32_i32_20 = arith.constant 32 : i32
      %89 = arith.muli %arg7, %c32_i32_20 : i32
      %90 = "triton.broadcast"(%89) : (i32) -> tensor<32x64xi32>
      %91 = "triton.getelementptr"(%arg12, %90) : (tensor<32x64x!triton.ptr<f16>>, tensor<32x64xi32>) -> tensor<32x64x!triton.ptr<f16>>
      scf.yield %86, %88, %91 : tensor<64x64xf32>, tensor<64x32x!triton.ptr<f16>>, tensor<32x64x!triton.ptr<f16>>
    }
    %53 = arith.truncf %52#0 : tensor<64x64xf32> to tensor<64x64xf16>
    %c64_i32_11 = arith.constant 64 : i32
    %54 = arith.muli %14, %c64_i32_11 : i32
    %55 = triton.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %56 = "triton.broadcast"(%54) : (i32) -> tensor<64xi32>
    %57 = arith.addi %56, %55 : tensor<64xi32>
    %c64_i32_12 = arith.constant 64 : i32
    %58 = arith.muli %16, %c64_i32_12 : i32
    %59 = triton.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %60 = "triton.broadcast"(%58) : (i32) -> tensor<64xi32>
    %61 = arith.addi %60, %59 : tensor<64xi32>
    %62 = "triton.reshape"(%57) : (tensor<64xi32>) -> tensor<64x1xi32>
    %63 = "triton.broadcast"(%arg8) : (i32) -> tensor<64x1xi32>
    %64 = arith.muli %63, %62 : tensor<64x1xi32>
    %65 = "triton.broadcast"(%arg2) : (!triton.ptr<f16>) -> tensor<64x1x!triton.ptr<f16>>
    %66 = "triton.getelementptr"(%65, %64) : (tensor<64x1x!triton.ptr<f16>>, tensor<64x1xi32>) -> tensor<64x1x!triton.ptr<f16>>
    %67 = "triton.reshape"(%61) : (tensor<64xi32>) -> tensor<1x64xi32>
    %c1_i32_13 = arith.constant 1 : i32
    %68 = "triton.broadcast"(%c1_i32_13) : (i32) -> tensor<1x64xi32>
    %69 = arith.muli %67, %68 : tensor<1x64xi32>
    %70 = "triton.broadcast"(%66) : (tensor<64x1x!triton.ptr<f16>>) -> tensor<64x64x!triton.ptr<f16>>
    %71 = "triton.broadcast"(%69) : (tensor<1x64xi32>) -> tensor<64x64xi32>
    %72 = "triton.getelementptr"(%70, %71) : (tensor<64x64x!triton.ptr<f16>>, tensor<64x64xi32>) -> tensor<64x64x!triton.ptr<f16>>
    %73 = "triton.reshape"(%57) : (tensor<64xi32>) -> tensor<64x1xi32>
    %74 = "triton.broadcast"(%arg3) : (i32) -> tensor<64x1xi32>
    %75 = arith.cmpi slt, %73, %74 : tensor<64x1xi32>
    %76 = "triton.reshape"(%61) : (tensor<64xi32>) -> tensor<1x64xi32>
    %77 = "triton.broadcast"(%arg4) : (i32) -> tensor<1x64xi32>
    %78 = arith.cmpi slt, %76, %77 : tensor<1x64xi32>
    %79 = "triton.broadcast"(%75) : (tensor<64x1xi1>) -> tensor<64x64xi1>
    %80 = "triton.broadcast"(%78) : (tensor<1x64xi1>) -> tensor<64x64xi1>
    %81 = arith.andi %79, %80 : tensor<64x64xi1>
    "triton.store"(%72, %53, %81) : (tensor<64x64x!triton.ptr<f16>>, tensor<64x64xf16>, tensor<64x64xi1>) -> ()
    return
  }
}
