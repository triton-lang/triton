#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ortho_diag_test_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<-7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<5.000000e-01> : tensor<1x64xf32, #blocked1>
    %cst_1 = arith.constant dense<-1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_3 = arith.constant dense<7.52316385E-37> : tensor<1x64xf32, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked1>
    %cst_5 = arith.constant dense<64> : tensor<64x1xi32, #blocked2>
    %cst_6 = arith.constant dense<7.52316385E-37> : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %cst_7 = arith.constant dense<0> : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked3>
    %3 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.expand_dims %3 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<64x1xi32, #blocked2>
    %6 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %7 = arith.muli %5, %cst_5 : tensor<64x1xi32, #blocked2>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %9 = tt.addptr %8, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x64xi32, #blocked2>
    %12 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %13 = tt.expand_dims %1 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1>
    %14 = tt.broadcast %9 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %15 = tt.broadcast %11 : tensor<1x64xi32, #blocked2> -> tensor<64x64xi32, #blocked2>
    %16 = tt.broadcast %12 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %17 = tt.addptr %14, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %18 = tt.load %17 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %19 = ttg.convert_layout %18 : tensor<64x64xf32, #blocked2> -> tensor<64x64xf32, #blocked1>
    %20 = tt.broadcast %6 : tensor<64x1xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %21 = arith.cmpi eq, %20, %16 : tensor<64x64xi32, #blocked>
    %22 = arith.uitofp %21 : tensor<64x64xi1, #blocked> to tensor<64x64xf32, #blocked>
    %23 = arith.xori %20, %16 : tensor<64x64xi32, #blocked>
    %24 = arith.cmpi eq, %0, %cst_7 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %25:2 = scf.for %arg4 = %c0_i32 to %arg3 step %c1_i32 iter_args(%arg5 = %19, %arg6 = %22) -> (tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>)  : i32 {
      %40 = arith.mulf %arg5, %arg5 : tensor<64x64xf32, #blocked1>
      %41 = ttg.convert_layout %40 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %42 = ttg.convert_layout %41 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %43 = ttg.convert_layout %23 : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %44 = tt.gather %42[%43] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %45 = ttg.convert_layout %44 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %46 = "tt.reduce"(%45) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %123 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %123 : f32
      }) : (tensor<64x64xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %47 = arith.select %24, %cst, %46 : tensor<64xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %48:2 = "tt.reduce"(%47, %0) <{axis = 0 : i32}> ({
      ^bb0(%arg7: f32, %arg8: i32, %arg9: f32, %arg10: i32):
        %123 = arith.cmpf oeq, %arg7, %arg9 : f32
        %124 = arith.cmpi slt, %arg8, %arg10 : i32
        %125 = arith.andi %123, %124 : i1
        %126 = arith.cmpf ogt, %arg7, %arg9 : f32
        %127 = arith.ori %126, %125 : i1
        %128 = arith.select %127, %arg7, %arg9 : f32
        %129 = arith.select %127, %arg8, %arg10 : i32
        tt.reduce.return %128, %129 : f32, i32
      }) : (tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) -> (f32, i32)
      %49 = ttg.convert_layout %arg5 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
      %50 = ttg.convert_layout %49 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %51 = ttg.convert_layout %6 : tensor<64x1xi32, #blocked> -> tensor<64x1xi32, #blocked>
      %52 = tt.gather %50[%51] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %53 = ttg.convert_layout %52 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked>
      %54 = ttg.convert_layout %53 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
      %55 = tt.reshape %54 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked1>
      %56 = tt.splat %48#1 : i32 -> tensor<64x1xi32, #blocked>
      %57 = arith.xori %6, %56 : tensor<64x1xi32, #blocked>
      %58 = ttg.convert_layout %49 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %59 = ttg.convert_layout %57 : tensor<64x1xi32, #blocked> -> tensor<64x1xi32, #blocked>
      %60 = tt.gather %58[%59] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
      %61 = ttg.convert_layout %60 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked>
      %62 = ttg.convert_layout %61 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
      %63 = tt.reshape %62 : tensor<64x1xf32, #blocked4> -> tensor<1x64xf32, #blocked1>
      %64 = tt.splat %48#1 : i32 -> tensor<1x64xi32, #blocked>
      %65 = tt.splat %48#1 : i32 -> tensor<1x64xi32, #blocked1>
      %66 = arith.xori %12, %64 : tensor<1x64xi32, #blocked>
      %67 = arith.xori %13, %65 : tensor<1x64xi32, #blocked1>
      %68 = tt.gather %55[%67] {axis = 1 : i32} : (tensor<1x64xf32, #blocked1>, tensor<1x64xi32, #blocked1>) -> tensor<1x64xf32, #blocked1>
      %69 = tt.splat %48#1 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %70 = arith.xori %1, %69 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %71 = arith.subi %1, %70 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %72 = arith.subf %55, %68 : tensor<1x64xf32, #blocked1>
      %73 = arith.mulf %72, %cst_0 : tensor<1x64xf32, #blocked1>
      %74 = arith.sitofp %71 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> to tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %75 = arith.mulf %74, %cst_6 : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %76 = tt.expand_dims %75 {axis = 0 : i32} : tensor<64xf32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xf32, #blocked1>
      %77 = arith.addf %73, %76 : tensor<1x64xf32, #blocked1>
      %78 = arith.cmpf olt, %77, %cst_4 : tensor<1x64xf32, #blocked1>
      %79 = arith.select %78, %cst_1, %cst_2 : tensor<1x64xi1, #blocked1>, tensor<1x64xf32, #blocked1>
      %80 = arith.mulf %63, %63 : tensor<1x64xf32, #blocked1>
      %81 = arith.addf %80, %cst_3 : tensor<1x64xf32, #blocked1>
      %82 = arith.mulf %73, %73 : tensor<1x64xf32, #blocked1>
      %83 = arith.addf %82, %81 : tensor<1x64xf32, #blocked1>
      %84 = math.sqrt %83 : tensor<1x64xf32, #blocked1>
      %85 = arith.mulf %79, %84 : tensor<1x64xf32, #blocked1>
      %86 = arith.addf %77, %85 : tensor<1x64xf32, #blocked1>
      %87 = arith.divf %63, %86 : tensor<1x64xf32, #blocked1>
      %88 = arith.mulf %87, %87 : tensor<1x64xf32, #blocked1>
      %89 = arith.addf %88, %cst_2 : tensor<1x64xf32, #blocked1>
      %90 = math.rsqrt %89 : tensor<1x64xf32, #blocked1>
      %91 = arith.mulf %87, %90 : tensor<1x64xf32, #blocked1>
      %92 = tt.broadcast %66 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %93 = tt.broadcast %67 : tensor<1x64xi32, #blocked1> -> tensor<64x64xi32, #blocked1>
      %94 = ttg.convert_layout %49 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %95 = ttg.convert_layout %92 : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %96 = tt.gather %94[%95] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %97 = ttg.convert_layout %96 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %98 = ttg.convert_layout %90 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked5>
      %99 = tt.broadcast %98 : tensor<1x64xf32, #blocked5> -> tensor<64x64xf32, #blocked5>
      %100 = ttg.convert_layout %90 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
      %101 = tt.broadcast %100 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %102 = tt.broadcast %90 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %103 = arith.mulf %102, %arg5 : tensor<64x64xf32, #blocked1>
      %104 = ttg.convert_layout %91 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
      %105 = tt.broadcast %104 : tensor<1x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %106 = tt.broadcast %91 : tensor<1x64xf32, #blocked1> -> tensor<64x64xf32, #blocked1>
      %107 = arith.mulf %105, %97 : tensor<64x64xf32, #blocked>
      %108 = ttg.convert_layout %107 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked1>
      %109 = arith.addf %103, %108 : tensor<64x64xf32, #blocked1>
      %110 = tt.trans %109 {order = array<i32: 1, 0>} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked5>
      %111 = tt.gather %110[%93] {axis = 1 : i32} : (tensor<64x64xf32, #blocked5>, tensor<64x64xi32, #blocked1>) -> tensor<64x64xf32, #blocked1>
      %112 = arith.mulf %99, %110 : tensor<64x64xf32, #blocked5>
      %113 = ttg.convert_layout %112 : tensor<64x64xf32, #blocked5> -> tensor<64x64xf32, #blocked1>
      %114 = arith.mulf %106, %111 : tensor<64x64xf32, #blocked1>
      %115 = arith.addf %113, %114 : tensor<64x64xf32, #blocked1>
      %116 = ttg.convert_layout %arg6 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %117 = ttg.convert_layout %92 : tensor<64x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
      %118 = tt.gather %116[%117] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
      %119 = ttg.convert_layout %118 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
      %120 = arith.mulf %101, %arg6 : tensor<64x64xf32, #blocked>
      %121 = arith.mulf %105, %119 : tensor<64x64xf32, #blocked>
      %122 = arith.addf %120, %121 : tensor<64x64xf32, #blocked>
      scf.yield %115, %122 : tensor<64x64xf32, #blocked1>, tensor<64x64xf32, #blocked>
    }
    %26 = ttg.convert_layout %25#0 : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    %27 = ttg.convert_layout %26 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %28 = ttg.convert_layout %6 : tensor<64x1xi32, #blocked> -> tensor<64x1xi32, #blocked>
    %29 = tt.gather %27[%28] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
    %30 = ttg.convert_layout %29 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked>
    %31 = ttg.convert_layout %30 : tensor<64x1xf32, #blocked> -> tensor<64x1xf32, #blocked4>
    %32 = tt.reshape %31 : tensor<64x1xf32, #blocked4> -> tensor<64xf32, #blocked3>
    %33 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x1x!tt.ptr<f32>, #blocked2>
    %34 = tt.addptr %33, %7 : tensor<64x1x!tt.ptr<f32>, #blocked2>, tensor<64x1xi32, #blocked2>
    %35 = tt.broadcast %34 : tensor<64x1x!tt.ptr<f32>, #blocked2> -> tensor<64x64x!tt.ptr<f32>, #blocked2>
    %36 = tt.addptr %35, %15 : tensor<64x64x!tt.ptr<f32>, #blocked2>, tensor<64x64xi32, #blocked2>
    %37 = ttg.convert_layout %25#1 : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked2>
    tt.store %36, %37 : tensor<64x64x!tt.ptr<f32>, #blocked2>
    %38 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked3>
    %39 = tt.addptr %38, %2 : tensor<64x!tt.ptr<f32>, #blocked3>, tensor<64xi32, #blocked3>
    tt.store %39, %32 : tensor<64x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

