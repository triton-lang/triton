// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s


// CHECK: #shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
// CHECK: #shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
// CHECK: #shared2 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
// CHECK-NOT: #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mma = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @_jagged_hstu_attn_fwd_0d1d2d3d4d5de67d8de9de10c11de12de13c14de15de16c17de18de19de20de21c22232425de26de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<i64, 1> {tt.divisibility = 16 : i32} , %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg5: f32 , %arg6: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg17: i32 , %arg18: i32 , %arg19: i32 , %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<64x32xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %c95_i32 = arith.constant 95 : i32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked1>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked1>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #blocked2>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.divsi %2, %arg18 : i32
    %4 = arith.remsi %2, %arg18 : i32
    %5 = tt.addptr %arg3, %3 : !tt.ptr<i64, 1>, i32
    %6 = tt.load %5 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %7 = tt.addptr %5, %c1_i32 : !tt.ptr<i64, 1>, i32
    %8 = tt.load %7 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i64
    %9 = arith.subi %8, %6 : i64
    %10 = arith.extsi %1 : i32 to i64
    %11 = arith.cmpi sge, %10, %9 : i64
    cf.cond_br %11, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %12 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %15 = tt.splat %1 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %16 = tt.splat %1 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.splat %1 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %18 = arith.addi %15, %12 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = arith.addi %16, %13 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = arith.addi %17, %14 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %21 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %22 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %23 = tt.expand_dims %18 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
    %24 = tt.expand_dims %19 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %25 = tt.expand_dims %20 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<64x1xi32, #mma>
    %26 = tt.splat %6 : (i64) -> tensor<64x1xi64, #blocked1>
    %27 = tt.splat %6 : (i64) -> tensor<64x1xi64, #blocked>
    %28 = arith.extsi %23 : tensor<64x1xi32, #blocked1> to tensor<64x1xi64, #blocked1>
    %29 = arith.extsi %24 : tensor<64x1xi32, #blocked> to tensor<64x1xi64, #blocked>
    %30 = arith.addi %26, %28 : tensor<64x1xi64, #blocked1>
    %31 = arith.addi %27, %29 : tensor<64x1xi64, #blocked>
    %32 = arith.extsi %arg7 : i32 to i64
    %33 = tt.splat %32 : (i64) -> tensor<64x1xi64, #blocked1>
    %34 = arith.muli %30, %33 : tensor<64x1xi64, #blocked1>
    %35 = arith.muli %4, %arg8 : i32
    %36 = arith.extsi %35 : i32 to i64
    %37 = tt.splat %36 : (i64) -> tensor<64x1xi64, #blocked1>
    %38 = arith.addi %34, %37 : tensor<64x1xi64, #blocked1>
    %39 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %40 = tt.expand_dims %39 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi32, #blocked1>
    %41 = tt.broadcast %38 : (tensor<64x1xi64, #blocked1>) -> tensor<64x64xi64, #blocked1>
    %42 = arith.extsi %40 : tensor<1x64xi32, #blocked1> to tensor<1x64xi64, #blocked1>
    %43 = tt.broadcast %42 : (tensor<1x64xi64, #blocked1>) -> tensor<64x64xi64, #blocked1>
    %44 = arith.addi %41, %43 : tensor<64x64xi64, #blocked1>
    %45 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %46 = tt.expand_dims %22 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %47 = tt.splat %6 : (i64) -> tensor<32x1xi64, #blocked1>
    %48 = tt.splat %6 : (i64) -> tensor<32x1xi64, #blocked>
    %49 = arith.extsi %45 : tensor<32x1xi32, #blocked1> to tensor<32x1xi64, #blocked1>
    %50 = arith.extsi %46 : tensor<32x1xi32, #blocked> to tensor<32x1xi64, #blocked>
    %51 = arith.addi %47, %49 : tensor<32x1xi64, #blocked1>
    %52 = arith.addi %48, %50 : tensor<32x1xi64, #blocked>
    %53 = arith.extsi %arg9 : i32 to i64
    %54 = tt.splat %53 : (i64) -> tensor<32x1xi64, #blocked1>
    %55 = arith.muli %51, %54 : tensor<32x1xi64, #blocked1>
    %56 = arith.muli %4, %arg10 : i32
    %57 = arith.extsi %56 : i32 to i64
    %58 = tt.splat %57 : (i64) -> tensor<32x1xi64, #blocked1>
    %59 = arith.addi %55, %58 : tensor<32x1xi64, #blocked1>
    %60 = tt.broadcast %59 : (tensor<32x1xi64, #blocked1>) -> tensor<32x64xi64, #blocked1>
    %61 = tt.broadcast %42 : (tensor<1x64xi64, #blocked1>) -> tensor<32x64xi64, #blocked1>
    %62 = arith.addi %60, %61 : tensor<32x64xi64, #blocked1>
    %63 = arith.extsi %arg11 : i32 to i64
    %64 = tt.splat %63 : (i64) -> tensor<32x1xi64, #blocked>
    %65 = arith.muli %52, %64 : tensor<32x1xi64, #blocked>
    %66 = arith.muli %4, %arg12 : i32
    %67 = arith.extsi %66 : i32 to i64
    %68 = tt.splat %67 : (i64) -> tensor<32x1xi64, #blocked>
    %69 = arith.addi %65, %68 : tensor<32x1xi64, #blocked>
    %70 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %71 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>
    %72 = tt.expand_dims %70 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %73 = tt.expand_dims %71 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #mma}>>) -> tensor<1x32xi32, #mma>
    %74 = tt.broadcast %69 : (tensor<32x1xi64, #blocked>) -> tensor<32x32xi64, #blocked>
    %75 = arith.extsi %72 : tensor<1x32xi32, #blocked> to tensor<1x32xi64, #blocked>
    %76 = tt.broadcast %75 : (tensor<1x32xi64, #blocked>) -> tensor<32x32xi64, #blocked>
    %77 = arith.addi %74, %76 : tensor<32x32xi64, #blocked>
    %78 = arith.muli %3, %arg19 : i32
    %79 = arith.muli %78, %arg19 : i32
    %80 = tt.splat %arg19 : (i32) -> tensor<64x1xi32, #blocked>
    %81 = arith.muli %24, %80 : tensor<64x1xi32, #blocked>
    %82 = tt.splat %79 : (i32) -> tensor<64x1xi32, #blocked>
    %83 = arith.addi %82, %81 : tensor<64x1xi32, #blocked>
    %84 = tt.broadcast %83 : (tensor<64x1xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %85 = tt.broadcast %72 : (tensor<1x32xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %86 = arith.addi %84, %85 : tensor<64x32xi32, #blocked>
    %87 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked1>
    %88 = tt.addptr %87, %44 : tensor<64x64x!tt.ptr<f32, 1>, #blocked1>, tensor<64x64xi64, #blocked1>
    %89 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x64x!tt.ptr<f32, 1>, #blocked1>
    %90 = tt.addptr %89, %62 : tensor<32x64x!tt.ptr<f32, 1>, #blocked1>, tensor<32x64xi64, #blocked1>
    %91 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<32x32x!tt.ptr<f32, 1>, #blocked>
    %92 = tt.addptr %91, %77 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi64, #blocked>
    %93 = tt.splat %arg4 : (!tt.ptr<f32, 1>) -> tensor<64x32x!tt.ptr<f32, 1>, #blocked>
    %94 = tt.addptr %93, %86 : tensor<64x32x!tt.ptr<f32, 1>, #blocked>, tensor<64x32xi32, #blocked>
    %95 = tt.splat %9 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %96 = tt.splat %9 : (i64) -> tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %97 = arith.extsi %18 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %98 = arith.extsi %19 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %99 = arith.cmpi slt, %97, %95 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %100 = arith.cmpi slt, %98, %96 : tensor<64xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %101 = tt.expand_dims %99 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi1, #blocked1>
    %102 = tt.expand_dims %100 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi1, #blocked>
    %103 = tt.broadcast %101 : (tensor<64x1xi1, #blocked1>) -> tensor<64x64xi1, #blocked1>
    %104 = tt.load %88, %103, %cst_4 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked1>
    %105 = arith.addi %1, %c95_i32 : i32
    %106 = arith.divsi %105, %c32_i32 : i32
    %107 = arith.muli %106, %c32_i32 : i32
    %108 = arith.extsi %21 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> to tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %109 = arith.extsi %22 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> to tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %110 = tt.splat %arg5 : (f32) -> tensor<64x32xf32, #mma>
    %111 = arith.sitofp %arg19 : i32 to f32
    %112 = arith.divf %cst_1, %111 : f32
    %113 = tt.splat %112 : (f32) -> tensor<64x32xf32, #mma>
    %114 = tt.broadcast %25 : (tensor<64x1xi32, #mma>) -> tensor<64x32xi32, #mma>
    %115 = scf.for %arg22 = %c0_i32 to %107 step %c32_i32 iter_args(%arg23 = %cst_5) -> (tensor<64x32xf32, #blocked2>)  : i32 {
      %130 = arith.extsi %arg22 : i32 to i64
      %131 = arith.subi %9, %130 : i64
      %132 = tt.splat %131 : (i64) -> tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %133 = tt.splat %131 : (i64) -> tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %134 = arith.cmpi slt, %108, %132 : tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      %135 = arith.cmpi slt, %109, %133 : tensor<32xi64, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %136 = tt.expand_dims %134 {axis = 1 : i32} : (tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi1, #blocked1>
      %137 = tt.expand_dims %135 {axis = 1 : i32} : (tensor<32xi1, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi1, #blocked>
      %138 = arith.muli %arg22, %arg9 : i32
      %139 = tt.splat %138 : (i32) -> tensor<32x64xi32, #blocked1>
      %140 = tt.addptr %90, %139 : tensor<32x64x!tt.ptr<f32, 1>, #blocked1>, tensor<32x64xi32, #blocked1>
      %141 = tt.broadcast %136 : (tensor<32x1xi1, #blocked1>) -> tensor<32x64xi1, #blocked1>
      %142 = tt.load %140, %141, %cst_3 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf32, #blocked1>
      %143 = tt.splat %arg22 : (i32) -> tensor<64x32xi32, #blocked>
      %144 = tt.addptr %94, %143 : tensor<64x32x!tt.ptr<f32, 1>, #blocked>, tensor<64x32xi32, #blocked>
      %145 = tt.load %144 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf32, #blocked>
      %146 = triton_gpu.convert_layout %145 : (tensor<64x32xf32, #blocked>) -> tensor<64x32xf32, #mma>
      %147 = triton_gpu.convert_layout %104 : (tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %148 = triton_gpu.convert_layout %142 : (tensor<32x64xf32, #blocked1>) -> tensor<32x64xf32, #shared>
      %149 = tt.trans %148 : (tensor<32x64xf32, #shared>) -> tensor<64x32xf32, #shared1>
      %150 = triton_gpu.convert_layout %149 : (tensor<64x32xf32, #shared1>) -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %151 = tt.dot %147, %150, %cst_0 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x64xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x32xf32, #mma>
      %152 = arith.mulf %151, %110 : tensor<64x32xf32, #mma>
      %153 = arith.addf %152, %146 : tensor<64x32xf32, #mma>
      %154 = arith.subf %cst_0, %153 : tensor<64x32xf32, #mma>
      %155 = math.exp %154 : tensor<64x32xf32, #mma>
      %156 = arith.addf %155, %cst : tensor<64x32xf32, #mma>
      %157 = arith.divf %cst, %156 : tensor<64x32xf32, #mma>
      %158 = arith.mulf %153, %157 : tensor<64x32xf32, #mma>
      %159 = arith.mulf %158, %113 : tensor<64x32xf32, #mma>
      %160 = tt.splat %arg22 : (i32) -> tensor<1x32xi32, #mma>
      %161 = arith.addi %160, %73 : tensor<1x32xi32, #mma>
      %162 = tt.broadcast %161 : (tensor<1x32xi32, #mma>) -> tensor<64x32xi32, #mma>
      %163 = arith.cmpi sge, %114, %162 : tensor<64x32xi32, #mma>
      %164 = arith.select %163, %159, %cst_0 : tensor<64x32xi1, #mma>, tensor<64x32xf32, #mma>
      %165 = arith.muli %arg22, %arg11 : i32
      %166 = tt.splat %165 : (i32) -> tensor<32x32xi32, #blocked>
      %167 = tt.addptr %92, %166 : tensor<32x32x!tt.ptr<f32, 1>, #blocked>, tensor<32x32xi32, #blocked>
      %168 = tt.broadcast %137 : (tensor<32x1xi1, #blocked>) -> tensor<32x32xi1, #blocked>
      %169 = tt.load %167, %168, %cst_2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x32xf32, #blocked>
      %170 = triton_gpu.convert_layout %164 : (tensor<64x32xf32, #mma>) -> tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>>
      %171 = triton_gpu.convert_layout %169 : (tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>>
      %172 = tt.dot %170, %171, %arg23 {allowTF32 = false, maxNumImpreciseAcc = 0 : i32} : tensor<64x32xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<32x32xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<64x32xf32, #blocked2>
      scf.yield %172 : tensor<64x32xf32, #blocked2>
    }
    %116 = arith.extsi %arg15 : i32 to i64
    %117 = tt.splat %116 : (i64) -> tensor<64x1xi64, #blocked>
    %118 = arith.muli %31, %117 : tensor<64x1xi64, #blocked>
    %119 = arith.muli %4, %arg16 : i32
    %120 = arith.extsi %119 : i32 to i64
    %121 = tt.splat %120 : (i64) -> tensor<64x1xi64, #blocked>
    %122 = arith.addi %118, %121 : tensor<64x1xi64, #blocked>
    %123 = tt.broadcast %122 : (tensor<64x1xi64, #blocked>) -> tensor<64x32xi64, #blocked>
    %124 = tt.broadcast %75 : (tensor<1x32xi64, #blocked>) -> tensor<64x32xi64, #blocked>
    %125 = arith.addi %123, %124 : tensor<64x32xi64, #blocked>
    %126 = tt.splat %arg6 : (!tt.ptr<f32, 1>) -> tensor<64x32x!tt.ptr<f32, 1>, #blocked>
    %127 = tt.addptr %126, %125 : tensor<64x32x!tt.ptr<f32, 1>, #blocked>, tensor<64x32xi64, #blocked>
    %128 = tt.broadcast %102 : (tensor<64x1xi1, #blocked>) -> tensor<64x32xi1, #blocked>
    %129 = triton_gpu.convert_layout %115 : (tensor<64x32xf32, #blocked2>) -> tensor<64x32xf32, #blocked>
    tt.store %127, %129, %128 {cache = 1 : i32, evict = 1 : i32} : tensor<64x32xf32, #blocked>
    tt.return
  }
}
