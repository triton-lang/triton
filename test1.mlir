module {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<64xi32>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_1 = arith.constant dense<5.000000e-01> : tensor<1x64xf32>
    %cst_2 = arith.constant dense<7.52316385E-37> : tensor<64xf32>
    %cst_3 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<1x64xf32>
    %cst_5 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %cst_7 = arith.constant dense<64> : tensor<64x1xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %2 = arith.muli %1, %cst_7 : tensor<64x1xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.broadcast %4 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
    %7 = tt.broadcast %5 : tensor<1x64xi32> -> tensor<64x64xi32>
    %8 = tt.addptr %6, %7 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %9 = tt.load %8 : tensor<64x64x!tt.ptr<f32>>
    %10 = tt.broadcast %1 : tensor<64x1xi32> -> tensor<64x64xi32>
    %11 = arith.cmpi eq, %10, %7 : tensor<64x64xi32>
    %12 = arith.uitofp %11 : tensor<64x64xi1> to tensor<64x64xf32>
    %13 = arith.xori %10, %7 : tensor<64x64xi32>
    %14 = arith.cmpi eq, %0, %cst : tensor<64xi32>
    %15:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %9, %arg6 = %12) -> (tensor<64x64xf32>, tensor<64x64xf32>)  : i32 {
      %24 = arith.mulf %arg5, %arg5 : tensor<64x64xf32>
      %25 = tt.gather %24[%13] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x64xi32>) -> tensor<64x64xf32>
      %26 = "tt.reduce"(%25) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %77 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %77 : f32
      }) : (tensor<64x64xf32>) -> tensor<64xf32>
      %27 = arith.select %14, %cst_0, %26 : tensor<64xi1>, tensor<64xf32>
      %28:2 = "tt.reduce"(%27, %0) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %77 = arith.cmpf oeq, %arg7, %arg9 : f32
        %78 = arith.cmpi slt, %arg8, %arg10 : i32
        %79 = arith.andi %77, %78 : i1
        %80 = arith.cmpf ogt, %arg7, %arg9 : f32
        %81 = arith.ori %80, %79 : i1
        %82 = arith.select %81, %arg7, %arg9 : f32
        %83 = arith.select %81, %arg8, %arg10 : i32
        tt.reduce.return %82, %83 : f32, i32
      }) : (tensor<64xf32>, tensor<64xi32>) -> (f32, i32)
      %29 = tt.gather %arg5[%1] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x1xi32>) -> tensor<64x1xf32>
      %30 = tt.reshape %29 : tensor<64x1xf32> -> tensor<1x64xf32>
      %31 = tt.splat %28#1 : i32 -> tensor<64x1xi32>
      %32 = arith.xori %1, %31 : tensor<64x1xi32>
      %33 = tt.gather %arg5[%32] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x1xi32>) -> tensor<64x1xf32>
      %34 = tt.reshape %33 : tensor<64x1xf32> -> tensor<1x64xf32>
      %35 = tt.splat %28#1 : i32 -> tensor<1x64xi32>
      %36 = arith.xori %5, %35 : tensor<1x64xi32>
      %37 = tt.gather %30[%36] {axis = 1 : i32} : (tensor<1x64xf32>, tensor<1x64xi32>) -> tensor<1x64xf32>
      %38 = tt.splat %28#1 : i32 -> tensor<64xi32>
      %39 = arith.xori %0, %38 : tensor<64xi32>
      %40 = arith.subi %0, %39 : tensor<64xi32>
      %41 = arith.subf %30, %37 : tensor<1x64xf32>
      %42 = arith.mulf %41, %cst_1 : tensor<1x64xf32>
      %43 = arith.sitofp %40 : tensor<64xi32> to tensor<64xf32>
      %44 = arith.mulf %43, %cst_2 : tensor<64xf32>
      %45 = tt.expand_dims %44 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
      %46 = arith.addf %42, %45 : tensor<1x64xf32>
      %47 = arith.cmpf olt, %46, %cst_6 : tensor<1x64xf32>
      %48 = arith.select %47, %cst_3, %cst_4 : tensor<1x64xi1>, tensor<1x64xf32>
      %49 = arith.mulf %34, %34 : tensor<1x64xf32>
      %50 = arith.addf %49, %cst_5 : tensor<1x64xf32>
      %51 = arith.mulf %42, %42 : tensor<1x64xf32>
      %52 = arith.addf %51, %50 : tensor<1x64xf32>
      %53 = math.sqrt %52 : tensor<1x64xf32>
      %54 = arith.mulf %48, %53 : tensor<1x64xf32>
      %55 = arith.addf %46, %54 : tensor<1x64xf32>
      %56 = arith.divf %34, %55 : tensor<1x64xf32>
      %57 = arith.mulf %56, %56 : tensor<1x64xf32>
      %58 = arith.addf %57, %cst_4 : tensor<1x64xf32>
      %59 = math.rsqrt %58 : tensor<1x64xf32>
      %60 = arith.mulf %56, %59 : tensor<1x64xf32>
      %61 = tt.broadcast %36 : tensor<1x64xi32> -> tensor<64x64xi32>
      %62 = tt.gather %arg5[%61] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x64xi32>) -> tensor<64x64xf32>
      %63 = tt.broadcast %59 : tensor<1x64xf32> -> tensor<64x64xf32>
      %64 = arith.mulf %63, %arg5 : tensor<64x64xf32>
      %65 = tt.broadcast %60 : tensor<1x64xf32> -> tensor<64x64xf32>
      %66 = arith.mulf %65, %62 : tensor<64x64xf32>
      %67 = arith.addf %64, %66 : tensor<64x64xf32>
      %68 = tt.trans %67 {order = array<i32: 1, 0>} : tensor<64x64xf32> -> tensor<64x64xf32>
      %69 = tt.gather %68[%61] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x64xi32>) -> tensor<64x64xf32>
      %70 = arith.mulf %63, %68 : tensor<64x64xf32>
      %71 = arith.mulf %65, %69 : tensor<64x64xf32>
      %72 = arith.addf %70, %71 : tensor<64x64xf32>
      %73 = tt.gather %arg6[%61] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x64xi32>) -> tensor<64x64xf32>
      %74 = arith.mulf %63, %arg6 : tensor<64x64xf32>
      %75 = arith.mulf %65, %73 : tensor<64x64xf32>
      %76 = arith.addf %74, %75 : tensor<64x64xf32>
      scf.yield %72, %76 : tensor<64x64xf32>, tensor<64x64xf32>
    }
    %16 = tt.gather %15#0[%1] {axis = 1 : i32} : (tensor<64x64xf32>, tensor<64x1xi32>) -> tensor<64x1xf32>
    %17 = tt.reshape %16 : tensor<64x1xf32> -> tensor<64xf32>
    %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>>
    %19 = tt.addptr %18, %2 : tensor<64x1x!tt.ptr<f32>>, tensor<64x1xi32>
    %20 = tt.broadcast %19 : tensor<64x1x!tt.ptr<f32>> -> tensor<64x64x!tt.ptr<f32>>
    %21 = tt.addptr %20, %7 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    tt.store %21, %15#1 : tensor<64x64x!tt.ptr<f32>>
    %22 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %23 = tt.addptr %22, %0 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %23, %17 : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

