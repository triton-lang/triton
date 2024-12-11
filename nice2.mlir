#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked1>
    %cst_1 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_2 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %cst_3 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_4 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_5 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked1>
    %cst_6 = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked2>
    %cst_7 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_8 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked2>
    %cst_9 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked2>
    %cst_10 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_11 = arith.constant dense<64> : tensor<64x1xi32, #blocked3>
    %cst_12 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked1>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %7 = tt.expand_dims %5 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %8 = arith.muli %6, %cst_11 : tensor<64x1xi32, #blocked3>
    %9 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %10 = tt.addptr %9, %8 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %12 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %13 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %14 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %16 = tt.broadcast %10 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %17 = tt.broadcast %15 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %18 = tt.broadcast %14 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %19 = tt.addptr %16, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %20 = tt.load %19 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %21 = ttg.convert_layout %20 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked1>
    %22 = tt.broadcast %7 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %23 = arith.cmpi eq, %22, %18 : tensor<64x64xi32, #blocked>
    %24 = arith.uitofp %23 : tensor<64x64xi1, #blocked> to tensor<64x64xf32, #blocked>
    %25 = arith.xori %22, %18 : tensor<64x64xi32, #blocked>
    %26 = arith.cmpi eq, %2, %cst : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %27:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %21, %arg6 = %24) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>)  : i32 {
      %39 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked1>
      %40 = ttg.convert_layout %39 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %41 = tt.gather %40[%25] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %127 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %127 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %43 = arith.select %26, %cst_7, %42 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %44:2 = "tt.reduce"(%43, %2) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %127 = arith.cmpf oeq, %arg7, %arg9 : f32
        %128 = arith.cmpi slt, %arg8, %arg10 : i32
        %129 = arith.andi %127, %128 : i1
        %130 = arith.cmpf ogt, %arg7, %arg9 : f32
        %131 = arith.ori %130, %129 : i1
        %132 = arith.select %131, %arg7, %arg9 : f32
        %133 = arith.select %131, %arg8, %arg10 : i32
        tt.reduce.return %132, %133 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> (f32, i32)
      %45 = tt.gather %arg5[%12] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<1x64xi32, #blocked1>) -> tensor<1x64xf32, #blocked1>
      %46 = ttg.convert_layout %45 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked2>
      %47 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked1>
      %48 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked2>
      %49 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked>
      %50 = arith.xori %12, %47 : tensor<1x64xi32, #blocked1>
      %51 = arith.xori %13, %48 : tensor<1x64xi32, #blocked2>
      %52 = arith.xori %14, %49 : tensor<1x64xi32, #blocked>
      %53 = tt.gather %arg5[%50] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<1x64xi32, #blocked1>) -> tensor<1x64xf32, #blocked1>
      %54 = ttg.convert_layout %53 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked2>
      %55 = tt.gather %45[%50] {axis = 1 : i32} : (tensor<1x64xf32, #blocked1>, tensor<1x64xi32, #blocked1>) -> tensor<1x64xf32, #blocked1>
      %56 = tt.gather %46[%51] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked2>, tensor<1x64xi32, #blocked2>) -> tensor<1x64xf32, #blocked2>
      %57 = tt.splat %44#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %58 = tt.splat %44#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %59 = arith.xori %0, %57 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %60 = arith.xori %1, %58 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %61 = arith.subi %0, %59 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %62 = arith.subi %1, %60 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %63 = arith.subf %45, %55 : tensor<1x64xf32, #blocked1>
      %64 = arith.subf %46, %56 : tensor<1x64xf32, #blocked2>
      %65 = arith.mulf %63, %cst_0 : tensor<1x64xf32, #blocked1>
      %66 = arith.mulf %64, %cst_6 : tensor<1x64xf32, #blocked2>
      %67 = arith.sitofp %61 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %68 = arith.sitofp %62 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %69 = arith.mulf %67, %cst_1 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %70 = arith.mulf %68, %cst_2 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>>
      %71 = tt.expand_dims %69 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf32, #blocked1>
      %72 = tt.expand_dims %70 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xf32, #blocked2>
      %73 = arith.addf %65, %71 : tensor<1x64xf32, #blocked1>
      %74 = arith.addf %66, %72 : tensor<1x64xf32, #blocked2>
      %75 = arith.cmpf olt, %73, %cst_12 : tensor<1x64xf32, #blocked1>
      %76 = arith.cmpf olt, %74, %cst_10 : tensor<1x64xf32, #blocked2>
      %77 = arith.select %75, %cst_3, %cst_4 : tensor<1x64xi1, #blocked1>, tensor<1x64xf32, #blocked1>
      %78 = arith.select %76, %cst_8, %cst_9 : tensor<1x64xi1, #blocked2>, tensor<1x64xf32, #blocked2>
      %79 = arith.mulf %53, %53 : tensor<1x64xf32, #blocked1>
      %80 = arith.addf %79, %cst_5 : tensor<1x64xf32, #blocked1>
      %81 = ttg.convert_layout %80 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked2>
      %82 = arith.mulf %65, %65 : tensor<1x64xf32, #blocked1>
      %83 = arith.mulf %66, %66 : tensor<1x64xf32, #blocked2>
      %84 = arith.addf %82, %80 : tensor<1x64xf32, #blocked1>
      %85 = arith.addf %83, %81 : tensor<1x64xf32, #blocked2>
      %86 = math.sqrt %84 : tensor<1x64xf32, #blocked1>
      %87 = math.sqrt %85 : tensor<1x64xf32, #blocked2>
      %88 = arith.mulf %77, %86 : tensor<1x64xf32, #blocked1>
      %89 = arith.mulf %78, %87 : tensor<1x64xf32, #blocked2>
      %90 = arith.addf %73, %88 : tensor<1x64xf32, #blocked1>
      %91 = arith.addf %74, %89 : tensor<1x64xf32, #blocked2>
      %92 = arith.divf %53, %90 : tensor<1x64xf32, #blocked1>
      %93 = arith.divf %54, %91 : tensor<1x64xf32, #blocked2>
      %94 = arith.mulf %92, %92 : tensor<1x64xf32, #blocked1>
      %95 = arith.mulf %93, %93 : tensor<1x64xf32, #blocked2>
      %96 = arith.addf %94, %cst_4 : tensor<1x64xf32, #blocked1>
      %97 = arith.addf %95, %cst_9 : tensor<1x64xf32, #blocked2>
      %98 = math.rsqrt %96 : tensor<1x64xf32, #blocked1>
      %99 = math.rsqrt %97 : tensor<1x64xf32, #blocked2>
      %100 = arith.mulf %93, %99 : tensor<1x64xf32, #blocked2>
      %101 = arith.mulf %92, %98 : tensor<1x64xf32, #blocked1>
      %102 = tt.broadcast %52 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %103 = tt.broadcast %50 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %104 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %105 = tt.gather %104[%102] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %106 = tt.broadcast %98 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %107 = ttg.convert_layout %99 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked1>
      %108 = tt.broadcast %107 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %109 = ttg.convert_layout %99 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %110 = tt.broadcast %109 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %111 = arith.mulf %108, %arg5 : tensor<64x64xf32, #blocked1>
      %112 = ttg.convert_layout %111 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %113 = ttg.convert_layout %100 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %114 = tt.broadcast %113 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %115 = tt.broadcast %101 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %116 = arith.mulf %114, %105 : tensor<64x64xf32, #blocked>
      %117 = arith.addf %112, %116 : tensor<64x64xf32, #blocked>
      %118 = tt.trans %117 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
      %119 = tt.gather %118[%103] {axis = 1 : i32} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %120 = arith.mulf %106, %118 : tensor<64x64xf32, #blocked1>
      %121 = arith.mulf %115, %119 : tensor<64x64xf32, #blocked1>
      %122 = arith.addf %120, %121 : tensor<64x64xf32, #blocked1>
      %123 = tt.gather %arg6[%102] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %124 = arith.mulf %110, %arg6 : tensor<64x64xf32, #blocked>
      %125 = arith.mulf %114, %123 : tensor<64x64xf32, #blocked>
      %126 = arith.addf %124, %125 : tensor<64x64xf32, #blocked>
      scf.yield %122, %126 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>
    }
    %28 = ttg.convert_layout %27#0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    %29 = tt.gather %28[%7] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
    %30 = ttg.convert_layout %29 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked5>
    %31 = tt.reshape %30 : tensor<64x1xf32, #blocked5> -> tensor<64xf32, #blocked4>
    %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %33 = tt.addptr %32, %8 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %34 = tt.broadcast %33 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %35 = tt.addptr %34, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %36 = ttg.convert_layout %27#1 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked3>
    tt.store %35, %36 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %37 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked4>
    %38 = tt.addptr %37, %3 : tensor<64x!tt.ptr<f32>, #blocked4>, tensor<64xi32, #blocked4>
    tt.store %38, %31 : tensor<64x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

