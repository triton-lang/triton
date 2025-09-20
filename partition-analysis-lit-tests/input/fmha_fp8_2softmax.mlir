#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 2, 64], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 64, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.use-ttg-ws" = true} {
  tt.func public @fmha_fp8_2softmax(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg3: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> loc(unknown), %arg4: i32 loc(unknown), %arg5: i32 loc(unknown), %arg6: i64 loc(unknown), %arg7: i64 loc(unknown), %arg8: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> loc(unknown), %arg9: i32 loc(unknown), %arg10: i32 loc(unknown), %arg11: i64 loc(unknown), %arg12: i64 loc(unknown), %arg13: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> loc(unknown), %arg14: i32 loc(unknown), %arg15: i32 loc(unknown), %arg16: i64 loc(unknown), %arg17: i64 loc(unknown), %arg18: !tt.tensordesc<tensor<128x64xf8E5M2, #shared1>> loc(unknown), %arg19: i32 loc(unknown), %arg20: i32 loc(unknown), %arg21: i64 loc(unknown), %arg22: i64 loc(unknown), %arg23: !tt.tensordesc<tensor<128xf32, #shared2>> loc(unknown), %arg24: i32 loc(unknown), %arg25: i64 loc(unknown), %arg26: f32 loc(unknown), %arg27: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown), %arg28: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg29: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg30: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg31: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg32: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg33: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg34: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg35: i32 loc(unknown), %arg36: i32 loc(unknown), %arg37: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %4 = tt.splat %2 : i32 -> tensor<128xi32, #blocked1>
    %5 = arith.addi %4, %3 : tensor<128xi32, #blocked1>
    %6 = arith.mulf %arg26, %cst : f32
    %7 = arith.muli %1, %c2_i32 : i32
    %8 = arith.muli %7, %arg37 : i32
    %9 = arith.addi %8, %2 : i32
    %10 = tt.descriptor_load %arg3[%9, %c0_i32] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
    %11 = ttg.local_alloc %10 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
    %12 = arith.addi %7, %c1_i32 : i32
    %13 = arith.muli %12, %arg37 : i32
    %14 = arith.addi %13, %2 : i32
    %15 = tt.descriptor_load %arg3[%14, %c0_i32] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
    %16 = ttg.local_alloc %15 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %17 = ttng.tmem_store %cst_2, %result_7[%token_8], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %18 = ttng.tmem_store %cst_2, %result_3[%token_4], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %19:10 = scf.for %arg38 = %c0_i32 to %arg37 step %c128_i32 iter_args(%arg39 = %cst_1, %arg40 = %cst_0, %arg41 = %cst_1, %arg42 = %cst_0, %arg43 = %8, %arg44 = %13, %arg45 = %token, %arg46 = %18, %arg47 = %token_6, %arg48 = %17) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %50 = tt.descriptor_load %arg8[%arg43, %c0_i32] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %51 = arith.muli %1, %c256_i32 : i32
      %52 = tt.descriptor_load %arg13[%51, %arg38] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %53 = tt.descriptor_load %arg8[%arg44, %c0_i32] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %54 = arith.muli %12, %c128_i32 : i32
      %55 = tt.descriptor_load %arg13[%54, %arg38] : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
      %56 = ttg.local_alloc %50 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %57 = ttg.memdesc_trans %56 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>
      %58 = ttng.tc_gen5_mma %11, %57, %result[%arg45], %false, %true : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_15, %token_16 = ttng.tmem_load %result[%58] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %59 = "tt.reduce"(%result_15) <{axis = 1 : i32}> ({
      ^bb0(%arg49: f32 loc(unknown), %arg50: f32 loc(unknown)):
        %108 = arith.maxnumf %arg49, %arg50 : f32
        tt.reduce.return %108 : f32
      }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %60 = tt.splat %6 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %61 = arith.mulf %59, %60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %62 = arith.maxnumf %arg39, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %63 = tt.splat %6 : f32 -> tensor<128x128xf32, #blocked>
      %64 = arith.mulf %result_15, %63 : tensor<128x128xf32, #blocked>
      %65 = tt.expand_dims %62 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %66 = tt.broadcast %65 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %67 = arith.subf %64, %66 : tensor<128x128xf32, #blocked>
      %68 = arith.subf %arg39, %62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %69 = math.exp2 %68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %70 = math.exp2 %67 : tensor<128x128xf32, #blocked>
      %71 = "tt.reduce"(%70) <{axis = 1 : i32}> ({
      ^bb0(%arg49: f32 loc(unknown), %arg50: f32 loc(unknown)):
        %108 = arith.addf %arg49, %arg50 : f32
        tt.reduce.return %108 : f32
      }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %72 = tt.fp_to_fp %70, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E5M2, #blocked>
      %result_17 = ttng.tmem_alloc %72 : (tensor<128x128xf8E5M2, #blocked>) -> !ttg.memdesc<128x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      %73 = arith.mulf %arg40, %69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %74 = arith.addf %73, %71 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %75 = tt.expand_dims %69 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %76 = tt.broadcast %75 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %result_18, %token_19 = ttng.tmem_load %result_3[%arg46] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %77 = arith.mulf %result_18, %76 : tensor<128x128xf32, #blocked>
      %78 = ttg.local_alloc %52 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %79 = ttg.memdesc_trans %78 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>
      %80 = ttng.tmem_store %77, %result_3[%token_19], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %81 = ttng.tc_gen5_mma %result_17, %79, %result_3[%80], %true, %true : !ttg.memdesc<128x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %82 = ttg.local_alloc %53 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %83 = ttg.memdesc_trans %82 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>
      %84 = ttng.tc_gen5_mma %16, %83, %result_5[%arg47], %false, %true : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_20, %token_21 = ttng.tmem_load %result_5[%84] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %85 = "tt.reduce"(%result_20) <{axis = 1 : i32}> ({
      ^bb0(%arg49: f32 loc(unknown), %arg50: f32 loc(unknown)):
        %108 = arith.maxnumf %arg49, %arg50 : f32
        tt.reduce.return %108 : f32
      }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %86 = arith.mulf %85, %60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %87 = arith.maxnumf %arg41, %86 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %88 = arith.mulf %result_20, %63 : tensor<128x128xf32, #blocked>
      %89 = tt.expand_dims %87 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %90 = tt.broadcast %89 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %91 = arith.subf %88, %90 : tensor<128x128xf32, #blocked>
      %92 = arith.subf %arg41, %87 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %93 = math.exp2 %92 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %94 = math.exp2 %91 : tensor<128x128xf32, #blocked>
      %95 = "tt.reduce"(%94) <{axis = 1 : i32}> ({
      ^bb0(%arg49: f32 loc(unknown), %arg50: f32 loc(unknown)):
        %108 = arith.addf %arg49, %arg50 : f32
        tt.reduce.return %108 : f32
      }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %96 = tt.fp_to_fp %94, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E5M2, #blocked>
      %result_22 = ttng.tmem_alloc %96 : (tensor<128x128xf8E5M2, #blocked>) -> !ttg.memdesc<128x128xf8E5M2, #tmem1, #ttng.tensor_memory>
      %97 = arith.mulf %arg42, %93 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %98 = arith.addf %97, %95 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %99 = tt.expand_dims %93 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %100 = tt.broadcast %99 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %result_23, %token_24 = ttng.tmem_load %result_7[%arg48] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %101 = arith.mulf %result_23, %100 : tensor<128x128xf32, #blocked>
      %102 = ttg.local_alloc %55 : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
      %103 = ttg.memdesc_trans %102 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>
      %104 = ttng.tmem_store %101, %result_7[%token_24], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %105 = ttng.tc_gen5_mma %result_22, %103, %result_7[%104], %true, %true : !ttg.memdesc<128x128xf8E5M2, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf8E5M2, #shared3, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %106 = arith.addi %arg43, %c128_i32 : i32
      %107 = arith.addi %arg44, %c128_i32 : i32
      scf.yield %62, %74, %87, %98, %106, %107, %token_16, %81, %token_21, %105 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>}
    %result_9, %token_10 = ttng.tmem_load %result_3[%19#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %result_11, %token_12 = ttng.tmem_load %result_7[%19#9] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %20 = tt.addptr %arg27, %8 : !tt.ptr<f32>, i32
    %21 = tt.splat %20 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %22 = tt.addptr %21, %5 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %23 = math.log2 %19#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %24 = arith.addf %19#0, %23 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = tt.expand_dims %19#1 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %26 = tt.broadcast %25 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %27 = arith.divf %result_9, %26 : tensor<128x128xf32, #blocked>
    %28 = ttg.convert_layout %24 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %22, %28 : tensor<128x!tt.ptr<f32>, #blocked1>
    %29 = tt.reshape %27 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %30 = tt.trans %29 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS, %outRHS = tt.split %30 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
    %31 = tt.fp_to_fp %outLHS, rounding = rtne : tensor<128x64xf32, #blocked5> -> tensor<128x64xf8E5M2, #blocked5>
    %32 = ttg.convert_layout %31 : tensor<128x64xf8E5M2, #blocked5> -> tensor<128x64xf8E5M2, #blocked6>
    tt.descriptor_store %arg18[%9, %c0_i32], %32 : !tt.tensordesc<tensor<128x64xf8E5M2, #shared1>>, tensor<128x64xf8E5M2, #blocked6>
    %33 = tt.fp_to_fp %outRHS, rounding = rtne : tensor<128x64xf32, #blocked5> -> tensor<128x64xf8E5M2, #blocked5>
    %34 = ttg.convert_layout %33 : tensor<128x64xf8E5M2, #blocked5> -> tensor<128x64xf8E5M2, #blocked6>
    tt.descriptor_store %arg18[%9, %c64_i32], %34 : !tt.tensordesc<tensor<128x64xf8E5M2, #shared1>>, tensor<128x64xf8E5M2, #blocked6>
    %35 = tt.addptr %arg27, %13 : !tt.ptr<f32>, i32
    %36 = tt.splat %35 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %37 = tt.addptr %36, %5 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %38 = math.log2 %19#3 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %39 = arith.addf %19#2, %38 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %40 = tt.expand_dims %19#3 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %41 = tt.broadcast %40 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %42 = arith.divf %result_11, %41 : tensor<128x128xf32, #blocked>
    %43 = ttg.convert_layout %39 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    tt.store %37, %43 : tensor<128x!tt.ptr<f32>, #blocked1>
    %44 = tt.reshape %42 : tensor<128x128xf32, #blocked> -> tensor<128x2x64xf32, #blocked3>
    %45 = tt.trans %44 {order = array<i32: 0, 2, 1>} : tensor<128x2x64xf32, #blocked3> -> tensor<128x64x2xf32, #blocked4>
    %outLHS_13, %outRHS_14 = tt.split %45 : tensor<128x64x2xf32, #blocked4> -> tensor<128x64xf32, #blocked5>
    %46 = tt.fp_to_fp %outLHS_13, rounding = rtne : tensor<128x64xf32, #blocked5> -> tensor<128x64xf8E5M2, #blocked5>
    %47 = ttg.convert_layout %46 : tensor<128x64xf8E5M2, #blocked5> -> tensor<128x64xf8E5M2, #blocked6>
    tt.descriptor_store %arg18[%14, %c0_i32], %47 : !tt.tensordesc<tensor<128x64xf8E5M2, #shared1>>, tensor<128x64xf8E5M2, #blocked6>
    %48 = tt.fp_to_fp %outRHS_14, rounding = rtne : tensor<128x64xf32, #blocked5> -> tensor<128x64xf8E5M2, #blocked5>
    %49 = ttg.convert_layout %48 : tensor<128x64xf8E5M2, #blocked5> -> tensor<128x64xf8E5M2, #blocked6>
    tt.descriptor_store %arg18[%14, %c64_i32], %49 : !tt.tensordesc<tensor<128x64xf8E5M2, #shared1>>, tensor<128x64xf8E5M2, #blocked6>
    tt.return
  }
}


