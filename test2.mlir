#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_4 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked2>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32, #blocked3>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked3>
    %cst_7 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_8 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked4>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<64x1xi32, #blocked3>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %7 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked3>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %11 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>>
    %12 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %13 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %14 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %15 = tt.expand_dims %11 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked5}>> -> tensor<1x64xi32, #blocked5>
    %16 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %17 = tt.broadcast %14 : tensor<1x64xi32, #blocked3> -> tensor<64x64xi32, #blocked3>
    %18 = tt.broadcast %13 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %19 = tt.addptr %16, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %20 = tt.load %19 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %21 = ttg.convert_layout %20 : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #blocked5>
    %22 = tt.broadcast %6 : tensor<64x1xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
    %23 = arith.cmpi eq, %22, %18 : tensor<64x64xi32, #blocked1>
    %24 = arith.uitofp %23 : tensor<64x64xi1, #blocked1> to tensor<64x64xf32, #blocked1>
    %25 = arith.xori %22, %18 : tensor<64x64xi32, #blocked1>
    %26 = arith.cmpi eq, %1, %cst_8 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %27:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %21, %arg6 = %24) -> (tensor<64x64xf32, #blocked5>, tensor<64x64xf32, #blocked1>)  : i32 {
      %39 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked5>
      %40 = ttg.convert_layout %39 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %41 = tt.gather %40[%25] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %140 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %140 : f32
      }) : (tensor<64x64xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %43 = arith.select %26, %cst_0, %42 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %44:2 = "tt.reduce"(%43, %1) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %140 = arith.cmpf oeq, %arg7, %arg9 : f32
        %141 = arith.cmpi slt, %arg8, %arg10 : i32
        %142 = arith.andi %140, %141 : i1
        %143 = arith.cmpf ogt, %arg7, %arg9 : f32
        %144 = arith.ori %143, %142 : i1
        %145 = arith.select %144, %arg7, %arg9 : f32
        %146 = arith.select %144, %arg8, %arg10 : i32
        tt.reduce.return %145, %146 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>) -> (f32, i32)
      %45 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %46 = tt.gather %45[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %47 = ttg.convert_layout %46 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked6>
      %48 = ttg.convert_layout %46 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked7>
      %49 = tt.reshape %47 : tensor<64x1xf32, #blocked6> -> tensor<1x64xf32, #blocked3>
      %50 = ttg.convert_layout %49 : tensor<1x64xf32, #blocked3> -> tensor<1x64xf32, #blocked>
      %51 = tt.reshape %48 : tensor<64x1xf32, #blocked7> -> tensor<1x64xf32, #blocked2>
      %52 = ttg.convert_layout %51 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %53 = tt.splat %44#1 : i32 -> tensor<64x1xi32, #blocked1>
      %54 = arith.xori %6, %53 : tensor<64x1xi32, #blocked1>
      %55 = tt.gather %45[%54] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
      %56 = ttg.convert_layout %55 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked6>
      %57 = ttg.convert_layout %55 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked7>
      %58 = tt.reshape %56 : tensor<64x1xf32, #blocked6> -> tensor<1x64xf32, #blocked3>
      %59 = ttg.convert_layout %58 : tensor<1x64xf32, #blocked3> -> tensor<1x64xf32, #blocked>
      %60 = tt.reshape %57 : tensor<64x1xf32, #blocked7> -> tensor<1x64xf32, #blocked2>
      %61 = ttg.convert_layout %60 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %62 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked>
      %63 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked1>
      %64 = tt.splat %44#1 : i32 -> tensor<1x64xi32, #blocked5>
      %65 = arith.xori %12, %62 : tensor<1x64xi32, #blocked>
      %66 = arith.xori %13, %63 : tensor<1x64xi32, #blocked1>
      %67 = arith.xori %15, %64 : tensor<1x64xi32, #blocked5>
      %68 = tt.gather %50[%65] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %69 = tt.gather %52[%65] {axis = 1 : i32, efficient_layout} : (tensor<1x64xf32, #blocked>, tensor<1x64xi32, #blocked>) -> tensor<1x64xf32, #blocked>
      %70 = tt.splat %44#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %71 = arith.xori %0, %70 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %72 = arith.subi %0, %71 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %73 = arith.subf %50, %68 : tensor<1x64xf32, #blocked>
      %74 = arith.subf %52, %69 : tensor<1x64xf32, #blocked>
      %75 = arith.mulf %73, %cst : tensor<1x64xf32, #blocked>
      %76 = arith.mulf %74, %cst : tensor<1x64xf32, #blocked>
      %77 = arith.sitofp %72 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %78 = arith.mulf %77, %cst_7 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %79 = tt.expand_dims %78 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xf32, #blocked>
      %80 = arith.addf %75, %79 : tensor<1x64xf32, #blocked>
      %81 = arith.addf %76, %79 : tensor<1x64xf32, #blocked>
      %82 = arith.cmpf olt, %80, %cst_3 : tensor<1x64xf32, #blocked>
      %83 = arith.cmpf olt, %81, %cst_3 : tensor<1x64xf32, #blocked>
      %84 = arith.select %82, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %85 = arith.select %83, %cst_1, %cst_2 : tensor<1x64xi1, #blocked>, tensor<1x64xf32, #blocked>
      %86 = arith.mulf %58, %58 : tensor<1x64xf32, #blocked3>
      %87 = arith.mulf %60, %60 : tensor<1x64xf32, #blocked2>
      %88 = arith.addf %86, %cst_6 : tensor<1x64xf32, #blocked3>
      %89 = ttg.convert_layout %88 : tensor<1x64xf32, #blocked3> -> tensor<1x64xf32, #blocked>
      %90 = arith.addf %87, %cst_4 : tensor<1x64xf32, #blocked2>
      %91 = ttg.convert_layout %90 : tensor<1x64xf32, #blocked2> -> tensor<1x64xf32, #blocked>
      %92 = arith.mulf %75, %75 : tensor<1x64xf32, #blocked>
      %93 = arith.mulf %76, %76 : tensor<1x64xf32, #blocked>
      %94 = arith.addf %92, %89 : tensor<1x64xf32, #blocked>
      %95 = arith.addf %93, %91 : tensor<1x64xf32, #blocked>
      %96 = math.sqrt %94 : tensor<1x64xf32, #blocked>
      %97 = math.sqrt %95 : tensor<1x64xf32, #blocked>
      %98 = arith.mulf %84, %96 : tensor<1x64xf32, #blocked>
      %99 = arith.mulf %85, %97 : tensor<1x64xf32, #blocked>
      %100 = arith.addf %80, %98 : tensor<1x64xf32, #blocked>
      %101 = arith.addf %81, %99 : tensor<1x64xf32, #blocked>
      %102 = arith.divf %59, %100 : tensor<1x64xf32, #blocked>
      %103 = arith.divf %61, %101 : tensor<1x64xf32, #blocked>
      %104 = arith.mulf %102, %102 : tensor<1x64xf32, #blocked>
      %105 = arith.mulf %103, %103 : tensor<1x64xf32, #blocked>
      %106 = arith.addf %104, %cst_2 : tensor<1x64xf32, #blocked>
      %107 = arith.addf %105, %cst_2 : tensor<1x64xf32, #blocked>
      %108 = math.rsqrt %106 : tensor<1x64xf32, #blocked>
      %109 = math.rsqrt %107 : tensor<1x64xf32, #blocked>
      %110 = arith.mulf %102, %108 : tensor<1x64xf32, #blocked>
      %111 = arith.mulf %103, %109 : tensor<1x64xf32, #blocked>
      %112 = tt.broadcast %66 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %113 = tt.broadcast %67 : tensor<1x64xi32, #blocked5> -> tensor<64x64xi32, #blocked5>
      %114 = tt.gather %45[%112] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %115 = ttg.convert_layout %108 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked5>
      %116 = tt.broadcast %115 : tensor<1x64xf32, #blocked5> -> tensor<64x64xf32, #blocked5>
      %117 = ttg.convert_layout %109 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked5>
      %118 = tt.broadcast %117 : tensor<1x64xf32, #blocked5> -> tensor<64x64xf32, #blocked5>
      %119 = ttg.convert_layout %109 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %120 = tt.broadcast %119 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %121 = arith.mulf %116, %arg5 : tensor<64x64xf32, #blocked5>
      %122 = ttg.convert_layout %121 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %123 = ttg.convert_layout %110 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %124 = tt.broadcast %123 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %125 = ttg.convert_layout %111 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked5>
      %126 = tt.broadcast %125 : tensor<1x64xf32, #blocked5> -> tensor<64x64xf32, #blocked5>
      %127 = ttg.convert_layout %111 : tensor<1x64xf32, #blocked> -> tensor<1x64xf32, #blocked1>
      %128 = tt.broadcast %127 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %129 = arith.mulf %124, %114 : tensor<64x64xf32, #blocked1>
      %130 = arith.addf %122, %129 : tensor<64x64xf32, #blocked1>
      %131 = tt.trans %130 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked5>
      %132 = tt.gather %131[%113] {axis = 1 : i32} : (tensor<64x64xf32, #blocked5>, tensor<64x64xi32, #blocked5>) -> tensor<64x64xf32, #blocked5>
      %133 = arith.mulf %118, %131 : tensor<64x64xf32, #blocked5>
      %134 = arith.mulf %126, %132 : tensor<64x64xf32, #blocked5>
      %135 = arith.addf %133, %134 : tensor<64x64xf32, #blocked5>
      %136 = tt.gather %arg6[%112] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %137 = arith.mulf %120, %arg6 : tensor<64x64xf32, #blocked1>
      %138 = arith.mulf %128, %136 : tensor<64x64xf32, #blocked1>
      %139 = arith.addf %137, %138 : tensor<64x64xf32, #blocked1>
      scf.yield %135, %139 : tensor<64x64xf32, #blocked5>, tensor<64x64xf32, #blocked1>
    }
    %28 = ttg.convert_layout %27#0 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
    %29 = tt.gather %28[%6] {axis = 1 : i32, efficient_layout} : (tensor<64x64xf32, #blocked1>, tensor<64x1xi32, #blocked1>) -> tensor<64x1xf32, #blocked1>
    %30 = ttg.convert_layout %29 : tensor<64x1xf32, #blocked1> -> tensor<64x1xf32, #blocked7>
    %31 = tt.reshape %30 : tensor<64x1xf32, #blocked7> -> tensor<64xf32, #blocked4>
    %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked3>
    %33 = tt.addptr %32, %7 : tensor<64x1x!tt.ptr<f32>, #blocked3>, tensor<64x1xi32, #blocked3>
    %34 = tt.broadcast %33 : tensor<64x1x!tt.ptr<f32>, #blocked3> -> tensor<64x64x!tt.ptr<f32>, #blocked3>
    %35 = tt.addptr %34, %17 : tensor<64x64x!tt.ptr<f32>, #blocked3>, tensor<64x64xi32, #blocked3>
    %36 = ttg.convert_layout %27#1 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked3>
    tt.store %35, %36 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    %37 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked4>
    %38 = tt.addptr %37, %2 : tensor<64x!tt.ptr<f32>, #blocked4>, tensor<64xi32, #blocked4>
    tt.store %38, %31 : tensor<64x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}

