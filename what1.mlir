#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x1xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64x1xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_3 = arith.constant dense<-1.000000e+00> : tensor<64x1xf32, #blocked>
    %cst_4 = arith.constant dense<-1.000000e+00> : tensor<64x1xf32, #blocked1>
    %cst_5 = arith.constant dense<7.52316385E-37> : tensor<64x1xf32, #blocked1>
    %cst_6 = arith.constant dense<5.000000e-01> : tensor<64x1xf32, #blocked>
    %cst_7 = arith.constant dense<5.000000e-01> : tensor<64x1xf32, #blocked1>
    %cst_8 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_9 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_10 = arith.constant dense<7.52316385E-37> : tensor<64x1xf32, #blocked>
    %cst_11 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %6 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %7 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %8 = arith.muli %7, %cst_9 : tensor<64x1xi32, #blocked2>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %10 = tt.addptr %9, %8 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %12 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %13 = tt.expand_dims %3 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %14 = tt.broadcast %10 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %15 = tt.broadcast %12 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %16 = tt.broadcast %13 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %17 = tt.addptr %14, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %18 = tt.load %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %19 = ttg.convert_layout %18 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked3>
    %20 = tt.broadcast %6 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %21 = arith.cmpi eq, %20, %16 : tensor<64x64xi32, #blocked1>
    %22 = arith.uitofp %21 : tensor<64x64xi1, #blocked1> to tensor<64x64xf32, #blocked1>
    %23 = arith.xori %20, %16 : tensor<64x64xi32, #blocked1>
    %24 = arith.cmpi eq, %3, %cst_11 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %25:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %19, %arg6 = %22) -> (tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked1>)  : i32 {
      %37 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked3>
      %38 = ttg.convert_layout %37 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %39 = tt.gather %38[%23] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %40 = "tt.reduce"(%39) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %129 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %129 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %41 = arith.select %24, %cst_8, %40 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %42:2 = "tt.reduce"(%41, %3) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %129 = arith.cmpf oeq, %arg7, %arg9 : f32
        %130 = arith.cmpi slt, %arg8, %arg10 : i32
        %131 = arith.andi %129, %130 : i1
        %132 = arith.cmpf ogt, %arg7, %arg9 : f32
        %133 = arith.ori %132, %131 : i1
        %134 = arith.select %133, %arg7, %arg9 : f32
        %135 = arith.select %133, %arg8, %arg10 : i32
        tt.reduce.return %134, %135 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %43 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %44 = tt.gather %43[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %45 = ttg.convert_layout %44 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked>
      %46 = tt.splat %42#1 : i32 -> tensor<64x1xi32, #blocked1>
      %47 = arith.xori %6, %46 : tensor<64x1xi32, #blocked1>
      %48 = tt.gather %43[%47] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %49 = ttg.convert_layout %48 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked>
      %50 = tt.splat %42#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %51 = tt.splat %42#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %52 = tt.splat %42#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %53 = arith.xori %0, %50 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %54 = arith.xori %1, %51 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %55 = arith.xori %2, %52 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %56 = tt.expand_dims %53 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
      %57 = tt.expand_dims %54 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %58 = tt.expand_dims %55 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %59 = tt.gather %44[%57] {axis = 0 : i32} : (tensor<64x1xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %60 = tt.gather %45[%58] {axis = 0 : i32, efficient_layout} : (tensor<64x1xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %61 = arith.subi %1, %54 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %62 = arith.subi %2, %55 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %63 = tt.expand_dims %61 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
      %64 = tt.expand_dims %62 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
      %65 = arith.subf %44, %59 : tensor<64x1xf32, #blocked1>
      %66 = arith.subf %45, %60 : tensor<64x1xf32, #blocked>
      %67 = arith.mulf %65, %cst_7 : tensor<64x1xf32, #blocked1>
      %68 = arith.mulf %66, %cst_6 : tensor<64x1xf32, #blocked>
      %69 = arith.sitofp %63 : tensor<64x1xi32, #blocked1> to tensor<64x1xf32, #blocked1>
      %70 = arith.sitofp %64 : tensor<64x1xi32, #blocked> to tensor<64x1xf32, #blocked>
      %71 = arith.mulf %69, %cst_5 : tensor<64x1xf32, #blocked1>
      %72 = arith.mulf %70, %cst_10 : tensor<64x1xf32, #blocked>
      %73 = arith.addf %67, %71 : tensor<64x1xf32, #blocked1>
      %74 = arith.addf %68, %72 : tensor<64x1xf32, #blocked>
      %75 = arith.cmpf olt, %73, %cst_0 : tensor<64x1xf32, #blocked1>
      %76 = arith.cmpf olt, %74, %cst : tensor<64x1xf32, #blocked>
      %77 = arith.select %75, %cst_4, %cst_2 : tensor<64x1xi1, #blocked1>, tensor<64x1xf32, #blocked1>
      %78 = arith.select %76, %cst_3, %cst_1 : tensor<64x1xi1, #blocked>, tensor<64x1xf32, #blocked>
      %79 = arith.mulf %48, %48 : tensor<64x1xf32, #blocked1>
      %80 = arith.addf %79, %cst_5 : tensor<64x1xf32, #blocked1>
      %81 = ttg.convert_layout %80 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked>
      %82 = arith.mulf %67, %67 : tensor<64x1xf32, #blocked1>
      %83 = arith.mulf %68, %68 : tensor<64x1xf32, #blocked>
      %84 = arith.addf %82, %80 : tensor<64x1xf32, #blocked1>
      %85 = arith.addf %83, %81 : tensor<64x1xf32, #blocked>
      %86 = math.sqrt %84 : tensor<64x1xf32, #blocked1>
      %87 = math.sqrt %85 : tensor<64x1xf32, #blocked>
      %88 = arith.mulf %77, %86 : tensor<64x1xf32, #blocked1>
      %89 = arith.mulf %78, %87 : tensor<64x1xf32, #blocked>
      %90 = arith.addf %73, %88 : tensor<64x1xf32, #blocked1>
      %91 = arith.addf %74, %89 : tensor<64x1xf32, #blocked>
      %92 = arith.divf %48, %90 : tensor<64x1xf32, #blocked1>
      %93 = arith.divf %49, %91 : tensor<64x1xf32, #blocked>
      %94 = arith.mulf %92, %92 : tensor<64x1xf32, #blocked1>
      %95 = arith.mulf %93, %93 : tensor<64x1xf32, #blocked>
      %96 = arith.addf %94, %cst_2 : tensor<64x1xf32, #blocked1>
      %97 = arith.addf %95, %cst_1 : tensor<64x1xf32, #blocked>
      %98 = math.rsqrt %96 : tensor<64x1xf32, #blocked1>
      %99 = math.rsqrt %97 : tensor<64x1xf32, #blocked>
      %100 = arith.mulf %92, %98 : tensor<64x1xf32, #blocked1>
      %101 = arith.mulf %93, %99 : tensor<64x1xf32, #blocked>
      %102 = tt.broadcast %56 : tensor<64x1xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
      %103 = tt.broadcast %57 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %104 = tt.gather %arg5[%102] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<64x64xi32, #blocked3>) -> tensor<64x64xf32, #blocked3>
      %105 = tt.broadcast %98 : tensor<64x1xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %106 = ttg.convert_layout %99 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked1>
      %107 = tt.broadcast %106 : tensor<64x1xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %108 = ttg.convert_layout %98 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked3>
      %109 = tt.broadcast %108 : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %110 = arith.mulf %109, %arg5 : tensor<64x64xf32, #blocked3>
      %111 = ttg.convert_layout %100 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked3>
      %112 = tt.broadcast %111 : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %113 = ttg.convert_layout %101 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked3>
      %114 = tt.broadcast %113 : tensor<64x1xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %115 = tt.broadcast %100 : tensor<64x1xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %116 = arith.mulf %112, %104 : tensor<64x64xf32, #blocked3>
      %117 = arith.addf %110, %116 : tensor<64x64xf32, #blocked3>
      %118 = tt.trans %117 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %119 = ttg.convert_layout %118 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
      %120 = tt.gather %119[%102] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked3>, tensor<64x64xi32, #blocked3>) -> tensor<64x64xf32, #blocked3>
      %121 = arith.mulf %107, %118 : tensor<64x64xf32, #blocked1>
      %122 = ttg.convert_layout %121 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
      %123 = arith.mulf %114, %120 : tensor<64x64xf32, #blocked3>
      %124 = arith.addf %122, %123 : tensor<64x64xf32, #blocked3>
      %125 = tt.gather %arg6[%103] {axis = 0 : i32} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %126 = arith.mulf %105, %arg6 : tensor<64x64xf32, #blocked1>
      %127 = arith.mulf %115, %125 : tensor<64x64xf32, #blocked1>
      %128 = arith.addf %126, %127 : tensor<64x64xf32, #blocked1>
      scf.yield %124, %128 : tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked1>
    }
    %26 = ttg.convert_layout %25#0 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
    %27 = tt.gather %26[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %28 = ttg.convert_layout %27 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
    %29 = tt.reshape %28 : tensor<64x1xf32, #blocked5> -> tensor<64xf32, #blocked4>
    %30 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %31 = tt.addptr %30, %8 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %32 = tt.broadcast %31 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %33 = tt.addptr %32, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %34 = ttg.convert_layout %25#1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
    tt.store %33, %34 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %35 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked4>
    %36 = tt.addptr %35, %4 : tensor<64x!tt.ptr<f32>, #blocked4>, tensor<64xi32, #blocked4>
    tt.store %36, %29 : tensor<64x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

