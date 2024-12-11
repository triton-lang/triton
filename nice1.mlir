#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked3>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked>
    %cst_7 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked3>
    %cst_8 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked3>
    %cst_9 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked3>
    %cst_10 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_11 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %cst_12 = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked3>
    %cst_13 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %7 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %8 = arith.muli %6, %cst_4 : tensor<64x1xi32, #blocked2>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %10 = tt.addptr %9, %8 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %12 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %14 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %15 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %16 = tt.broadcast %10 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %17 = tt.broadcast %14 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %18 = tt.broadcast %13 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %19 = tt.addptr %16, %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %20 = tt.load %19 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %21 = ttg.convert_layout %20 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked3>
    %22 = tt.broadcast %7 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %23 = arith.cmpi eq, %22, %18 : tensor<64x64xi32, #blocked1>
    %24 = arith.uitofp %23 : tensor<64x64xi1, #blocked1> to tensor<64x64xf32, #blocked1>
    %25 = arith.xori %22, %18 : tensor<64x64xi32, #blocked1>
    %26 = arith.cmpi eq, %2, %cst_13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %21, %arg6 = %24) -> (tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked1>)  : i32 {
      %39 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked3>
      %40 = ttg.convert_layout %39 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %41 = tt.gather %40[%25] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %128 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %128 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %43 = arith.select %26, %cst_0, %42 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %44:2 = "tt.reduce"(%43, %2) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %128 = arith.cmpf oeq, %arg7, %arg9 : f32
        %129 = arith.cmpi slt, %arg8, %arg10 : i32
        %130 = arith.andi %128, %129 : i1
        %131 = arith.cmpf ogt, %arg7, %arg9 : f32
        %132 = arith.ori %131, %130 : i1
        %133 = arith.select %132, %arg7, %arg9 : f32
        %134 = arith.select %132, %arg8, %arg10 : i32
        tt.reduce.return %133, %134 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %45 = tt.gather %arg5[%15] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<1x64xi32, #blocked3>) -> tensor<1x64xf32, #blocked3>
      %46 = tt.gather %arg5[%12] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %47 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked3>
      %48 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked>
      %49 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked1>
      %50 = arith.xori %15, %47 : tensor<1x64xi32, #blocked3>
      %51 = arith.xori %12, %48 : tensor<1x64xi32, #blocked>
      %52 = arith.xori %13, %49 : tensor<1x64xi32, #blocked1>
      %53 = tt.gather %arg5[%50] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<1x64xi32, #blocked3>) -> tensor<1x64xf32, #blocked3>
      %54 = tt.gather %arg5[%51] {axis = 0 : i32} : (tensor<64x64xf32, #blocked3>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %55 = tt.gather %45[%50] {axis = 1 : i32} : (tensor<1x64xf32, #blocked3>, tensor<1x64xi32, #blocked3>) -> tensor<1x64xf32, #blocked3>
      %56 = tt.gather %46[%51] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %57 = tt.splat %44#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %58 = tt.splat %44#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %59 = arith.xori %0, %57 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %60 = arith.xori %1, %58 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %61 = arith.subi %0, %59 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %62 = arith.subi %1, %60 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %63 = arith.subf %45, %55 : tensor<1x64xf32, #blocked3>
      %64 = arith.subf %46, %56 : tensor<1x64xf32, #blocked>
      %65 = arith.mulf %63, %cst_12 : tensor<1x64xf32, #blocked3>
      %66 = arith.mulf %64, %cst : tensor<1x64xf32, #blocked>
      %67 = arith.sitofp %61 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %68 = arith.sitofp %62 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %69 = arith.mulf %67, %cst_11 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked3}>>
      %70 = arith.mulf %68, %cst_10 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %71 = tt.expand_dims %69 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xf32, #blocked3>
      %72 = tt.expand_dims %70 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %73 = arith.addf %65, %71 : tensor<1x64xf32, #blocked3>
      %74 = arith.addf %66, %72 : tensor<1x64xf32, #blocked>
      %75 = arith.cmpf olt, %73, %cst_5 : tensor<1x64xf32, #blocked3>
      %76 = arith.cmpf olt, %74, %cst_3 : tensor<1x64xf32, #blocked>
      %77 = arith.select %75, %cst_9, %cst_8 : tensor<1x64xi1, #blocked3>, tensor<1x64xf32, #blocked3>
      %78 = arith.select %76, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %79 = arith.mulf %53, %53 : tensor<1x64xf32, #blocked3>
      %80 = arith.mulf %54, %54 : tensor<1x64xf32, #blocked>
      %81 = arith.addf %79, %cst_7 : tensor<1x64xf32, #blocked3>
      %82 = arith.addf %80, %cst_6 : tensor<1x64xf32, #blocked>
      %83 = arith.mulf %65, %65 : tensor<1x64xf32, #blocked3>
      %84 = arith.mulf %66, %66 : tensor<1x64xf32, #blocked>
      %85 = arith.addf %83, %81 : tensor<1x64xf32, #blocked3>
      %86 = arith.addf %84, %82 : tensor<1x64xf32, #blocked>
      %87 = math.sqrt %85 : tensor<1x64xf32, #blocked3>
      %88 = math.sqrt %86 : tensor<1x64xf32, #blocked>
      %89 = arith.mulf %77, %87 : tensor<1x64xf32, #blocked3>
      %90 = arith.mulf %78, %88 : tensor<1x64xf32, #blocked>
      %91 = arith.addf %73, %89 : tensor<1x64xf32, #blocked3>
      %92 = arith.addf %74, %90 : tensor<1x64xf32, #blocked>
      %93 = arith.divf %53, %91 : tensor<1x64xf32, #blocked3>
      %94 = arith.divf %54, %92 : tensor<1x64xf32, #blocked>
      %95 = arith.mulf %93, %93 : tensor<1x64xf32, #blocked3>
      %96 = arith.mulf %94, %94 : tensor<1x64xf32, #blocked>
      %97 = arith.addf %95, %cst_8 : tensor<1x64xf32, #blocked3>
      %98 = arith.addf %96, %cst_2 : tensor<1x64xf32, #blocked>
      %99 = math.rsqrt %97 : tensor<1x64xf32, #blocked3>
      %100 = math.rsqrt %98 : tensor<1x64xf32, #blocked>
      %101 = arith.mulf %94, %100 : tensor<1x64xf32, #blocked>
      %102 = arith.mulf %93, %99 : tensor<1x64xf32, #blocked3>
      %103 = tt.broadcast %52 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %104 = tt.broadcast %50 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
      %105 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %106 = tt.gather %105[%103] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %107 = tt.broadcast %99 : tensor<1x64xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %108 = ttg.convert_layout %100 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked3>
      %109 = tt.broadcast %108 : tensor<1x64xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %110 = ttg.convert_layout %100 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %111 = tt.broadcast %110 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %112 = arith.mulf %109, %arg5 : tensor<64x64xf32, #blocked3>
      %113 = ttg.convert_layout %112 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
      %114 = ttg.convert_layout %101 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %115 = tt.broadcast %114 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %116 = tt.broadcast %102 : tensor<1x64xf32, #blocked3> -> tensor<64x64xf32, #blocked3>
      %117 = arith.mulf %115, %106 : tensor<64x64xf32, #blocked1>
      %118 = arith.addf %113, %117 : tensor<64x64xf32, #blocked1>
      %119 = tt.trans %118 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
      %120 = tt.gather %119[%104] {axis = 1 : i32} : (tensor<64x64xf32, #blocked3>, tensor<64x64xi32, #blocked3>) -> tensor<64x64xf32, #blocked3>
      %121 = arith.mulf %107, %119 : tensor<64x64xf32, #blocked3>
      %122 = arith.mulf %116, %120 : tensor<64x64xf32, #blocked3>
      %123 = arith.addf %121, %122 : tensor<64x64xf32, #blocked3>
      %124 = tt.gather %arg6[%103] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %125 = arith.mulf %111, %arg6 : tensor<64x64xf32, #blocked1>
      %126 = arith.mulf %115, %124 : tensor<64x64xf32, #blocked1>
      %127 = arith.addf %125, %126 : tensor<64x64xf32, #blocked1>
      scf.yield %123, %127 : tensor<64x64xf32, #blocked3>, tensor<64x64xf32, #blocked1>
    }
    %28 = ttg.convert_layout %27#0 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
    %29 = tt.gather %28[%7] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %30 = ttg.convert_layout %29 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked5>
    %31 = tt.reshape %30 : tensor<64x1xf32, #blocked5> -> tensor<64xf32, #blocked4>
    %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %33 = tt.addptr %32, %8 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %34 = tt.broadcast %33 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %35 = tt.addptr %34, %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %36 = ttg.convert_layout %27#1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked2>
    tt.store %35, %36 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %37 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked4>
    %38 = tt.addptr %37, %3 : tensor<64x!tt.ptr<f32>, #blocked4>, tensor<64xi32, #blocked4>
    tt.store %38, %31 : tensor<64x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

