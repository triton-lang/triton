#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel_data_part(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<128> : tensor<128xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %cst_2 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.descriptor_load %arg0[%1, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
    %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %4 = arith.addi %1, %c128_i32 : i32
    %5 = tt.descriptor_load %arg0[%4, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
    %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %7 = tt.splat %arg24 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %8 = tt.splat %arg24 : f32 -> tensor<128x128xf32, #blocked1>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_5, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_7, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %9 = ttng.tmem_store %cst_0, %result_7[%token_8], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %10 = ttng.tmem_store %cst_0, %result_3[%token_4], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %11:8 = scf.for %arg25 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %cst_2, %arg27 = %cst_1, %arg28 = %cst_2, %arg29 = %cst_1, %arg30 = %token, %arg31 = %10, %arg32 = %token_6, %arg33 = %9) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %33 = tt.descriptor_load %arg5[%arg25, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %34 = ttg.local_alloc %33 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %35 = ttg.memdesc_trans %34 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %36 = tt.descriptor_load %arg10[%arg25, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %37 = ttg.local_alloc %36 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %38 = ttng.tc_gen5_mma %3, %35, %result[%arg30], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_13, %token_14 = ttng.tmem_load %result[%38] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %39 = "tt.reduce"(%result_13) <{axis = 1 : i32}> ({
      ^bb0(%arg34: f32, %arg35: f32):
        %78 = arith.maxnumf %arg34, %arg35 : f32
        tt.reduce.return %78 : f32
      }) : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %40 = arith.mulf %39, %7 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %41 = arith.maxnumf %arg27, %40 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %42 = arith.mulf %result_13, %8 : tensor<128x128xf32, #blocked1>
      %43 = tt.expand_dims %41 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %44 = tt.broadcast %43 : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %45 = arith.subf %42, %44 : tensor<128x128xf32, #blocked1>
      %46 = math.exp2 %45 : tensor<128x128xf32, #blocked1>
      %47 = arith.subf %arg27, %41 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %48 = math.exp2 %47 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %49 = "tt.reduce"(%46) <{axis = 1 : i32}> ({
      ^bb0(%arg34: f32, %arg35: f32):
        %78 = arith.addf %arg34, %arg35 : f32
        tt.reduce.return %78 : f32
      }) : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %50 = tt.expand_dims %48 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %51 = tt.broadcast %50 : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %result_15, %token_16 = ttng.tmem_load %result_3[%arg31] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %52 = arith.mulf %result_15, %51 : tensor<128x128xf32, #blocked1>
      %53 = arith.truncf %46 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
      %result_17 = ttng.tmem_alloc %53 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
      %54 = ttng.tmem_store %52, %result_3[%token_16], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %55 = ttng.tc_gen5_mma %result_17, %37, %result_3[%54], %true, %true : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %56 = arith.mulf %arg26, %48 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %57 = arith.addf %56, %49 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %58 = ttng.tc_gen5_mma %6, %35, %result_5[%arg32], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_18, %token_19 = ttng.tmem_load %result_5[%58] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %59 = "tt.reduce"(%result_18) <{axis = 1 : i32}> ({
      ^bb0(%arg34: f32, %arg35: f32):
        %78 = arith.maxnumf %arg34, %arg35 : f32
        tt.reduce.return %78 : f32
      }) : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %60 = arith.mulf %59, %7 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %61 = arith.maxnumf %arg29, %60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %62 = arith.mulf %result_18, %8 : tensor<128x128xf32, #blocked1>
      %63 = tt.expand_dims %61 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %64 = tt.broadcast %63 : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %65 = arith.subf %62, %64 : tensor<128x128xf32, #blocked1>
      %66 = math.exp2 %65 : tensor<128x128xf32, #blocked1>
      %67 = arith.subf %arg29, %61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %68 = math.exp2 %67 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %69 = "tt.reduce"(%66) <{axis = 1 : i32}> ({
      ^bb0(%arg34: f32, %arg35: f32):
        %78 = arith.addf %arg34, %arg35 : f32
        tt.reduce.return %78 : f32
      }) : (tensor<128x128xf32, #blocked1>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %70 = tt.expand_dims %68 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xf32, #blocked1>
      %71 = tt.broadcast %70 : tensor<128x1xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %result_20, %token_21 = ttng.tmem_load %result_7[%arg33] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      %72 = arith.mulf %result_20, %71 : tensor<128x128xf32, #blocked1>
      %73 = arith.truncf %66 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
      %result_22 = ttng.tmem_alloc %73 : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>
      %74 = ttng.tmem_store %72, %result_7[%token_21], %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %75 = ttng.tc_gen5_mma %result_22, %37, %result_7[%74], %true, %true : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %76 = arith.mulf %arg28, %68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      %77 = arith.addf %76, %69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
      scf.yield %57, %41, %77, %61, %token_14, %55, %token_19, %75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
    %result_9, %token_10 = ttng.tmem_load %result_3[%11#5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %result_11, %token_12 = ttng.tmem_load %result_7[%11#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %12 = arith.truncf %result_9 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    %13 = ttg.convert_layout %12 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %arg15[%1, %c0_i32], %13 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    %14 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %16 = tt.splat %14 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %17 = tt.addptr %16, %15 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %18 = arith.truncf %11#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %19 = ttg.convert_layout %18 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf16, #blocked>
    tt.store %17, %19 : tensor<128x!tt.ptr<f16>, #blocked>
    %20 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %21 = tt.splat %20 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked>
    %22 = tt.addptr %21, %15 : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %23 = arith.truncf %11#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %24 = ttg.convert_layout %23 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf16, #blocked>
    tt.store %22, %24 : tensor<128x!tt.ptr<f16>, #blocked>
    %25 = arith.truncf %result_11 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    %26 = ttg.convert_layout %25 : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %arg15[%4, %c0_i32], %26 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    %27 = tt.addptr %17, %cst : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %28 = arith.truncf %11#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %29 = ttg.convert_layout %28 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf16, #blocked>
    tt.store %27, %29 : tensor<128x!tt.ptr<f16>, #blocked>
    %30 = tt.addptr %22, %cst : tensor<128x!tt.ptr<f16>, #blocked>, tensor<128xi32, #blocked>
    %31 = arith.truncf %11#3 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = ttg.convert_layout %31 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf16, #blocked>
    tt.store %30, %32 : tensor<128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
