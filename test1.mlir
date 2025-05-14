#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem2 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_attn_fwd_tma_dp(%arg0: f32, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg5: i32, %arg6: i32, %arg7: i64, %arg8: i64, %arg9: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg10: i32, %arg11: i32, %arg12: i64, %arg13: i64, %arg14: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg15: i32, %arg16: i32, %arg17: i64, %arg18: i64, %arg19: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg20: i32, %arg21: i32, %arg22: i64, %arg23: i64, %arg24: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.44269502 : f32
    %c256_i32 = arith.constant 256 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.divsi %1, %arg3 : i32
    %3 = arith.remsi %1, %arg3 : i32
    %4 = arith.muli %3, %arg24 : i32
    %5 = arith.addi %2, %4 : i32
    %6 = arith.muli %0, %c256_i32 : i32
    %7 = arith.addi %5, %6 : i32
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %9 = tt.splat %6 : i32 -> tensor<128xi32, #blocked1>
    %10 = arith.addi %9, %8 : tensor<128xi32, #blocked1>
    %11 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #blocked1>
    %12 = arith.addi %9, %11 : tensor<128xi32, #blocked1>
    %13 = arith.mulf %arg0, %cst : f32
    %14 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %15, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %15, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %16 = ttng.tensor_desc_to_tma_ptr %arg4 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %16[%7, %c0_i32] %14, %15, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %15, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %15 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %17 = arith.addi %7, %c128_i32 : i32
    %18 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %19, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %19, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %20 = ttng.tensor_desc_to_tma_ptr %arg4 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %20[%17, %c0_i32] %18, %19, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %19, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %19 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %21 = ttg.memdesc_subview %result[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %21, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %22 = ttg.memdesc_subview %result_2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %22, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %25 = ttg.memdesc_subview %24[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %26 = ttg.memdesc_subview %24[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %26, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %27 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %28 = ttg.memdesc_subview %27[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %29 = ttg.memdesc_subview %27[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %30 = ttg.memdesc_subview %result_3[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_4 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %31 = ttg.memdesc_subview %result_4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.arrive_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %26, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %32 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %33 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_subview %33[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %34, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %35 = ttg.memdesc_subview %33[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %35, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %36 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %37 = ttg.memdesc_subview %36[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %38 = ttg.memdesc_subview %36[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %34, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %35, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %39 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %40 = ttg.memdesc_subview %39[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.memdesc_subview %41[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %43 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.memdesc_subview %43[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %45 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %46 = ttg.memdesc_subview %45[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %46, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %46, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.memdesc_subview %47[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %48, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %49 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %50 = ttg.memdesc_subview %49[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %50, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %47, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %51 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %52 = ttg.memdesc_subview %51[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %52, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %53 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %54 = ttg.memdesc_subview %53[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %55 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %56 = ttg.memdesc_subview %55[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %56, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %57 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %58 = ttg.memdesc_subview %57[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %58, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %58, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %56, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %59 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %60 = ttg.memdesc_subview %59[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %60, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %61 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %62 = ttg.memdesc_subview %61[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %62, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %59, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %63 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %64 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %65 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %66 = ttg.memdesc_subview %63[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_1, %66 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %67 = ttg.memdesc_subview %64[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %68 = ttg.memdesc_subview %65[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %67, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %68, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %67, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %69 = ttg.memdesc_subview %64[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %70 = ttg.memdesc_subview %65[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %69, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %70, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %70, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %71 = ttg.memdesc_subview %64[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %72 = ttg.memdesc_subview %65[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %71, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %72, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %72, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %73 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %74 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %75 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %76 = ttg.memdesc_subview %73[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_1, %76 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %77 = ttg.memdesc_subview %74[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %78 = ttg.memdesc_subview %75[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %77, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %78, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %77, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %79 = ttg.memdesc_subview %74[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %80 = ttg.memdesc_subview %75[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %79, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %80, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %80, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %81 = ttg.memdesc_subview %74[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %82 = ttg.memdesc_subview %75[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %81, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %82, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %82, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %83 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %84 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %85 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %86 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %87:4 = ttg.warp_specialize(%24, %27, %23, %33, %36, %32, %42, %14, %30, %40, %63, %64, %65, %44, %49, %22, %46, %47, %54, %18, %31, %52, %73, %74, %75, %56, %61, %21, %58, %59, %6, %5, %arg9, %arg14, %84, %83, %86, %85, %13) attributes {requestedRegisters = array<i32: 24, 24, 200, 200>}
    default {
      %194:12 = scf.for %arg25 = %c0_i32 to %6 step %c128_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c0_i32, %arg31 = %c0_i32, %arg32 = %c-1_i32, %arg33 = %c0_i32, %arg34 = %c0_i32, %arg35 = %c0_i32, %arg36 = %c-1_i32, %arg37 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %195 = arith.xori %arg26, %c1_i32 : i32
        %196 = arith.addi %arg30, %c1_i32 : i32
        %197 = arith.xori %arg31, %c1_i32 : i32
        %198 = arith.cmpi eq, %196, %c3_i32 : i32
        %199 = arith.select %198, %197, %arg31 : i32
        %200 = arith.select %198, %c1_i32, %196 : i32
        %201 = ttg.memdesc_subview %63[%200, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %202 = ttg.memdesc_subview %64[%200] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %203 = ttg.memdesc_subview %65[%200] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %202, %199 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %204 = ttg.local_load %201 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %203, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %205 = arith.addi %arg32, %c1_i32 : i32
        %206 = arith.xori %arg33, %c1_i32 : i32
        %207 = arith.cmpi eq, %205, %c3_i32 : i32
        %208 = arith.select %207, %206, %arg33 : i32
        %209 = arith.select %207, %c1_i32, %205 : i32
        %210 = ttg.memdesc_subview %63[%209, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %211 = ttg.memdesc_subview %64[%209] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %212 = ttg.memdesc_subview %65[%209] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %211, %208 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %213 = ttg.local_load %210 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %212, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %214 = arith.subf %213, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %215 = math.exp2 %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %46, %arg27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %216 = ttng.tmem_subslice %22 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %217 = ttng.tmem_subslice %22 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %218 = arith.xori %arg27, %c1_i32 : i32
        %219 = tt.expand_dims %215 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %220 = tt.broadcast %219 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_13 = ttng.tmem_load %216 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %221 = arith.mulf %result_13, %220 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %221, %216, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_14 = ttng.tmem_load %217 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %222 = arith.mulf %result_14, %220 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %222, %217, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %223 = arith.xori %arg28, %c1_i32 : i32
        %224 = arith.addi %arg34, %c1_i32 : i32
        %225 = arith.xori %arg35, %c1_i32 : i32
        %226 = arith.cmpi eq, %224, %c3_i32 : i32
        %227 = arith.select %226, %225, %arg35 : i32
        %228 = arith.select %226, %c1_i32, %224 : i32
        %229 = ttg.memdesc_subview %73[%228, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %230 = ttg.memdesc_subview %74[%228] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %231 = ttg.memdesc_subview %75[%228] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %230, %227 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %232 = ttg.local_load %229 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %231, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %233 = arith.addi %arg36, %c1_i32 : i32
        %234 = arith.xori %arg37, %c1_i32 : i32
        %235 = arith.cmpi eq, %233, %c3_i32 : i32
        %236 = arith.select %235, %234, %arg37 : i32
        %237 = arith.select %235, %c1_i32, %233 : i32
        %238 = ttg.memdesc_subview %73[%237, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %239 = ttg.memdesc_subview %74[%237] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %240 = ttg.memdesc_subview %75[%237] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %239, %236 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %241 = ttg.local_load %238 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %240, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %242 = arith.subf %241, %232 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %243 = math.exp2 %242 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %58, %arg29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %244 = ttng.tmem_subslice %21 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %245 = ttng.tmem_subslice %21 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %246 = arith.xori %arg29, %c1_i32 : i32
        %247 = tt.expand_dims %243 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %248 = tt.broadcast %247 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_15 = ttng.tmem_load %244 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %249 = arith.mulf %result_15, %248 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %249, %244, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_16 = ttng.tmem_load %245 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %250 = arith.mulf %result_16, %248 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %250, %245, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %56, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %195, %218, %223, %246, %200, %199, %209, %208, %228, %227, %237, %236 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_yield %194#0, %194#1, %194#2, %194#3 : i32, i32, i32, i32
    }
    partition0(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: f32) num_warps(1) {
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %false = arith.constant false
      %194 = arith.cmpi sgt, %arg55, %c0_i32_15 : i32
      %195 = ttg.memdesc_subview %arg25[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %196 = ttg.memdesc_subview %arg26[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %197 = ttg.memdesc_subview %arg27[%c0_i32_15, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %198 = ttg.memdesc_trans %197 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      ttng.wait_barrier %196, %c0_i32_15, %194 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg31, %c0_i32_15, %194 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg32, %198, %arg33, %false, %194, %195[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %arg43, %c0_i32_15, %194 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg44, %198, %arg45, %false, %194, %195[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %199:10 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %c0_i32_15, %arg66 = %c0_i32_15, %arg67 = %c0_i32_15, %arg68 = %c0_i32_15, %arg69 = %c0_i32_15, %arg70 = %c0_i32_15, %arg71 = %c0_i32_15, %arg72 = %c0_i32_15, %arg73 = %c0_i32_15, %arg74 = %c0_i32_15) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %200 = arith.subi %arg55, %c128_i32_14 : i32
        %201 = arith.cmpi slt, %arg64, %200 : i32
        %202 = ttg.memdesc_subview %arg28[%arg67] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %203 = ttg.memdesc_subview %arg29[%arg67] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %204 = ttg.memdesc_subview %arg30[%arg67, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>

        %205 = arith.xori %arg70, %c1_i32_16 : i32

        %206 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %203, %arg68 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg38, %205, %true_17 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg39, %arg71 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %206, %204, %arg40, %true_17, %true_17, %202[%true_17], %arg41[%true_17], %arg42[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        %207 = arith.addi %arg67, %c1_i32_16 : i32
        %208 = arith.xori %arg68, %c1_i32_16 : i32
        %209 = arith.cmpi eq, %207, %c2_i32_13 : i32
        %210 = arith.select %209, %c0_i32_15, %207 : i32
        %211 = arith.select %209, %208, %arg68 : i32

        %212 = arith.xori %arg71, %c1_i32_16 : i32
        %213 = arith.xori %arg74, %c1_i32_16 : i32
        %214 = arith.xori %arg69, %c1_i32_16 : i32

        %215 = arith.addi %arg65, %c1_i32_16 : i32
        %216 = arith.xori %arg66, %c1_i32_16 : i32
        %217 = arith.cmpi eq, %215, %c2_i32_13 : i32
        %218 = arith.select %217, %c0_i32_15, %215 : i32
        %219 = arith.select %217, %216, %arg66 : i32
        %220 = ttg.memdesc_subview %arg25[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %221 = ttg.memdesc_subview %arg26[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %222 = ttg.memdesc_subview %arg27[%218, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %223 = ttg.memdesc_trans %222 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
        ttng.wait_barrier %221, %219, %201 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg31, %214, %201 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg32, %223, %arg33, %false, %201, %220[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        %224 = arith.xori %arg73, %c1_i32_16 : i32

        %225 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg50, %224, %true_17 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg51, %arg74 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %225, %204, %arg52, %true_17, %true_17, %202[%true_17], %arg53[%true_17], %arg54[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>

        %226 = arith.xori %arg72, %c1_i32_16 : i32

        ttng.wait_barrier %arg43, %226, %201 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg44, %223, %arg45, %false, %201, %220[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %218, %219, %210, %211, %214, %205, %212, %226, %224, %213 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition1(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: f32) num_warps(2) {
      %c256_i32_13 = arith.constant 256 : i32
      %c2_i32_14 = arith.constant 2 : i32
      %c128_i32_15 = arith.constant 128 : i32
      %c0_i32_16 = arith.constant 0 : i32
      %c1_i32_17 = arith.constant 1 : i32
      %true_18 = arith.constant true
      %194 = arith.cmpi sgt, %arg55, %c0_i32_16 : i32
      %195 = ttg.memdesc_subview %arg25[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %195, %c0_i32_16, %194 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %196 = ttg.memdesc_subview %arg26[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %196, 32768 {ttg.assigned_cluster = 2 : i32}, %194 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %197 = ttg.memdesc_subview %arg27[%c0_i32_16, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %198 = ttng.tensor_desc_to_tma_ptr %arg57 {ttg.assigned_cluster = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %198[%arg56, %c0_i32_16] %197, %196, %194 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %199 = arith.cmpi sgt, %arg55, %c128_i32_15 : i32
      %200 = arith.addi %arg56, %c128_i32_15 : i32
      %201 = ttg.memdesc_subview %arg25[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %201, %c0_i32_16, %199 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %202 = ttg.memdesc_subview %arg26[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %202, 32768 {ttg.assigned_cluster = 2 : i32}, %199 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %203 = ttg.memdesc_subview %arg27[%c1_i32_17, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.async_tma_copy_global_to_local %198[%200, %c0_i32_16] %203, %202, %199 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %204:6 = scf.for %arg64 = %c0_i32_16 to %arg55 step %c128_i32_15 iter_args(%arg65 = %200, %arg66 = %c1_i32_17, %arg67 = %c0_i32_16, %arg68 = %c0_i32_16, %arg69 = %c0_i32_16, %arg70 = %arg56) -> (i32, i32, i32, i32, i32, i32)  : i32 {
        %205 = arith.subi %arg55, %c256_i32_13 : i32
        %206 = arith.cmpi slt, %arg64, %205 : i32
        %207 = ttg.memdesc_subview %arg28[%arg68] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %207, %arg69 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %208 = ttg.memdesc_subview %arg29[%arg68] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %208, 32768 {ttg.assigned_cluster = 0 : i32}, %true_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %209 = ttg.memdesc_subview %arg30[%arg68, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %210 = ttng.tensor_desc_to_tma_ptr %arg58 {ttg.assigned_cluster = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %210[%arg70, %c0_i32_16] %209, %208, %true_18 {ttg.assigned_cluster = 0 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %211 = arith.addi %arg68, %c1_i32_17 : i32
        %212 = arith.xori %arg69, %c1_i32_17 : i32
        %213 = arith.cmpi eq, %211, %c2_i32_14 : i32
        %214 = arith.select %213, %c0_i32_16, %211 : i32
        %215 = arith.select %213, %212, %arg69 : i32
        %216 = arith.addi %arg65, %c128_i32_15 : i32
        %217 = arith.addi %arg66, %c1_i32_17 : i32
        %218 = arith.xori %arg67, %c1_i32_17 : i32
        %219 = arith.cmpi eq, %217, %c2_i32_14 : i32
        %220 = arith.select %219, %c0_i32_16, %217 : i32
        %221 = arith.select %219, %218, %arg67 : i32
        %222 = ttg.memdesc_subview %arg25[%220] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %222, %221, %206 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %223 = ttg.memdesc_subview %arg26[%220] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %223, 32768 {ttg.assigned_cluster = 2 : i32}, %206 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %224 = ttg.memdesc_subview %arg27[%220, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        ttng.async_tma_copy_global_to_local %198[%216, %c0_i32_16] %224, %223, %206 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        scf.yield %216, %220, %221, %214, %215, %arg65 : i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition2(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: f32) num_warps(4) {
      %c3_i32_13 = arith.constant 3 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %cst_18 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_19 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %194 = tt.splat %arg63 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.splat %arg63 : f32 -> tensor<128x128xf32, #blocked>
      %196:6 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %cst_19, %arg66 = %cst_18, %arg67 = %c0_i32_15, %arg68 = %c0_i32_15, %arg69 = %c0_i32_15, %arg70 = %c0_i32_15) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32)  : i32 {
        ttng.wait_barrier %arg34, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg33 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %197 = arith.xori %arg67, %c1_i32_16 : i32
        %198 = "tt.reduce"(%result_20) <{axis = 1 : i32}> ({
        ^bb0(%arg71: f32, %arg72: f32):
          %222 = arith.maxnumf %arg71, %arg72 : f32
          tt.reduce.return %222 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %199 = arith.mulf %198, %194 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %200 = arith.maxnumf %arg66, %199 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %201 = arith.addi %arg69, %c1_i32_16 : i32
        %202 = arith.xori %arg70, %c1_i32_16 : i32
        %203 = arith.cmpi eq, %201, %c3_i32_13 : i32
        %204 = arith.select %203, %202, %arg70 : i32
        %205 = arith.select %203, %c1_i32_16, %201 : i32
        %206 = ttg.memdesc_subview %arg35[%205, %c0_i32_15] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %207 = ttg.memdesc_subview %arg36[%205] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %208 = ttg.memdesc_subview %arg37[%205] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %208, %204 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %200, %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %207, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %209 = arith.mulf %result_20, %195 : tensor<128x128xf32, #blocked>
        %210 = tt.expand_dims %200 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %211 = tt.broadcast %210 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %212 = arith.subf %209, %211 : tensor<128x128xf32, #blocked>
        %213 = math.exp2 %212 : tensor<128x128xf32, #blocked>
        %214 = arith.subf %arg66, %200 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = math.exp2 %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %216 = "tt.reduce"(%213) <{axis = 1 : i32}> ({
        ^bb0(%arg71: f32, %arg72: f32):
          %222 = arith.addf %arg71, %arg72 : f32
          tt.reduce.return %222 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %217 = arith.truncf %213 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %218 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg42, %arg68 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %217, %218, %true_17 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %219 = arith.mulf %arg65, %215 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %220 = arith.addf %219, %216 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %221 = arith.xori %arg68, %c1_i32_16 : i32
        scf.yield %220, %200, %197, %221, %205, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %196#0, %arg59 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %196#1, %arg60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    }
    partition3(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: f32) num_warps(4) {
      %c3_i32_13 = arith.constant 3 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %cst_18 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_19 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %194 = tt.splat %arg63 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.splat %arg63 : f32 -> tensor<128x128xf32, #blocked>
      %196:6 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %cst_19, %arg66 = %cst_18, %arg67 = %c0_i32_15, %arg68 = %c0_i32_15, %arg69 = %c0_i32_15, %arg70 = %c0_i32_15) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32)  : i32 {
        ttng.wait_barrier %arg46, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg45 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %197 = arith.xori %arg67, %c1_i32_16 : i32
        %198 = "tt.reduce"(%result_20) <{axis = 1 : i32}> ({
        ^bb0(%arg71: f32, %arg72: f32):
          %222 = arith.maxnumf %arg71, %arg72 : f32
          tt.reduce.return %222 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %199 = arith.mulf %198, %194 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %200 = arith.maxnumf %arg66, %199 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %201 = arith.addi %arg69, %c1_i32_16 : i32
        %202 = arith.xori %arg70, %c1_i32_16 : i32
        %203 = arith.cmpi eq, %201, %c3_i32_13 : i32
        %204 = arith.select %203, %202, %arg70 : i32
        %205 = arith.select %203, %c1_i32_16, %201 : i32
        %206 = ttg.memdesc_subview %arg47[%205, %c0_i32_15] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %207 = ttg.memdesc_subview %arg48[%205] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %208 = ttg.memdesc_subview %arg49[%205] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %208, %204 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %200, %206 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %207, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %209 = arith.mulf %result_20, %195 : tensor<128x128xf32, #blocked>
        %210 = tt.expand_dims %200 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %211 = tt.broadcast %210 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %212 = arith.subf %209, %211 : tensor<128x128xf32, #blocked>
        %213 = math.exp2 %212 : tensor<128x128xf32, #blocked>
        %214 = arith.subf %arg66, %200 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = math.exp2 %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %216 = "tt.reduce"(%213) <{axis = 1 : i32}> ({
        ^bb0(%arg71: f32, %arg72: f32):
          %222 = arith.addf %arg71, %arg72 : f32
          tt.reduce.return %222 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %217 = arith.truncf %213 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %218 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg54, %arg68 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %217, %218, %true_17 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %219 = arith.mulf %arg65, %215 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %220 = arith.addf %219, %216 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %221 = arith.xori %arg68, %c1_i32_16 : i32
        scf.yield %220, %200, %197, %221, %205, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %196#0, %arg61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %196#1, %arg62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, i32, !tt.tensordesc<tensor<128x128xf16, #shared>>, !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, f32) -> (i32, i32, i32, i32)
    %88 = ttg.local_load %86 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %89 = ttg.local_alloc %88 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %86 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %90 = ttg.local_load %85 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %91 = ttg.local_alloc %90 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %85 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %92 = ttg.local_load %84 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %93 = ttg.local_alloc %92 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %84 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %94 = ttg.local_load %83 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %95 = ttg.local_alloc %94 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %83 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %73 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %74 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %75 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %63 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %68 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %69 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %72 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %64 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %65 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %77 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %78 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %79 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %80 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %81 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %82 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %58, %87#3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %62 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %61 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %60 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %59 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %58 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %56 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %54, %87#2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %53 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %52 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %46, %87#1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %50 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %49 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %48 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %46 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %42, %87#0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %38 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %36 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %35 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %33 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %32 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %28 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %27 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %25 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %24 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %23 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %96 = arith.muli %0, %c256_i32 {tt.divisibility = dense<256> : tensor<1xi32>} : i32
    %97 = arith.addi %0, %c1_i32 : i32
    %98 = arith.muli %97, %c256_i32 : i32
    %99 = arith.addi %5, %96 : i32
    %result_5 = ttng.tmem_load %21 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_6 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %100 = ttg.memdesc_subview %result_6[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %result_5, %100, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_7 = ttng.tmem_load %22 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_8 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %101 = ttg.memdesc_subview %result_8[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %result_7, %101, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %102 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %103 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %104 = ttg.memdesc_subview %103[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %104, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %105 = ttg.memdesc_subview %103[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %105, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %106 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %107 = ttg.memdesc_subview %106[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %107, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %108 = ttg.memdesc_subview %106[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %108, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %result_9 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %109 = ttg.memdesc_subview %result_9[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_10 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %110 = ttg.memdesc_subview %result_10[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.arrive_barrier %104, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %105, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %111 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %112 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %113 = ttg.memdesc_subview %112[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %113, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %114 = ttg.memdesc_subview %112[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %114, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %115 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %116 = ttg.memdesc_subview %115[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %116, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %117 = ttg.memdesc_subview %115[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %117, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %113, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %114, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %118 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %119 = ttg.memdesc_subview %118[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %119, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %120 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %121 = ttg.memdesc_subview %120[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %121, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %121, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %122 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %123 = ttg.memdesc_subview %122[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %123, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %124 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %125 = ttg.memdesc_subview %124[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %125, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %125, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %123, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %126 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %127 = ttg.memdesc_subview %126[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %127, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %128 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %129 = ttg.memdesc_subview %128[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %129, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %126, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %130 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %131 = ttg.memdesc_subview %130[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %131, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %132 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %133 = ttg.memdesc_subview %132[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %133, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %133, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %134 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %135 = ttg.memdesc_subview %134[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %135, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %136 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %137 = ttg.memdesc_subview %136[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %137, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %137, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %135, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %138 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %139 = ttg.memdesc_subview %138[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %139, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %140 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %141 = ttg.memdesc_subview %140[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %141, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %138, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %142 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %143 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %144 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %145 = ttg.memdesc_subview %142[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %94, %145 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %146 = ttg.memdesc_subview %143[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %147 = ttg.memdesc_subview %144[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %146, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %147, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %146, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %148 = ttg.memdesc_subview %143[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %149 = ttg.memdesc_subview %144[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %148, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %149, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %149, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %150 = ttg.memdesc_subview %143[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %151 = ttg.memdesc_subview %144[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %150, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %151, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %151, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %152 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %153 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %154 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %155 = ttg.memdesc_subview %152[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %90, %155 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %156 = ttg.memdesc_subview %153[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %157 = ttg.memdesc_subview %154[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %156, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %157, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %156, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %158 = ttg.memdesc_subview %153[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %159 = ttg.memdesc_subview %154[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %158, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %159, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %159, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %160 = ttg.memdesc_subview %153[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %161 = ttg.memdesc_subview %154[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %160, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %161, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %161, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %162 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %163 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %164 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %165 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %166:4 = ttg.warp_specialize(%103, %106, %102, %112, %115, %111, %121, %14, %109, %119, %142, %143, %144, %123, %128, %101, %125, %126, %133, %18, %110, %131, %152, %153, %154, %135, %140, %100, %137, %138, %96, %98, %99, %arg9, %arg14, %93, %95, %163, %162, %89, %91, %165, %164, %13, %6) attributes {requestedRegisters = array<i32: 24, 24, 200, 200>}
    default {
      %194:12 = scf.for %arg25 = %96 to %98 step %c128_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c0_i32, %arg31 = %c0_i32, %arg32 = %c-1_i32, %arg33 = %c0_i32, %arg34 = %c0_i32, %arg35 = %c0_i32, %arg36 = %c-1_i32, %arg37 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %195 = arith.xori %arg26, %c1_i32 : i32
        %196 = arith.addi %arg30, %c1_i32 : i32
        %197 = arith.xori %arg31, %c1_i32 : i32
        %198 = arith.cmpi eq, %196, %c3_i32 : i32
        %199 = arith.select %198, %197, %arg31 : i32
        %200 = arith.select %198, %c1_i32, %196 : i32
        %201 = ttg.memdesc_subview %142[%200, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %202 = ttg.memdesc_subview %143[%200] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %203 = ttg.memdesc_subview %144[%200] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %202, %199 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %204 = ttg.local_load %201 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %203, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %205 = arith.addi %arg32, %c1_i32 : i32
        %206 = arith.xori %arg33, %c1_i32 : i32
        %207 = arith.cmpi eq, %205, %c3_i32 : i32
        %208 = arith.select %207, %206, %arg33 : i32
        %209 = arith.select %207, %c1_i32, %205 : i32
        %210 = ttg.memdesc_subview %142[%209, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %211 = ttg.memdesc_subview %143[%209] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %212 = ttg.memdesc_subview %144[%209] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %211, %208 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %213 = ttg.local_load %210 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %212, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %214 = arith.subf %213, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %215 = math.exp2 %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %125, %arg27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %216 = ttng.tmem_subslice %101 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %217 = ttng.tmem_subslice %101 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %218 = arith.xori %arg27, %c1_i32 : i32
        %219 = tt.expand_dims %215 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %220 = tt.broadcast %219 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_13 = ttng.tmem_load %216 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %221 = arith.mulf %result_13, %220 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %221, %216, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_14 = ttng.tmem_load %217 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %222 = arith.mulf %result_14, %220 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %222, %217, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %123, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %223 = arith.xori %arg28, %c1_i32 : i32
        %224 = arith.addi %arg34, %c1_i32 : i32
        %225 = arith.xori %arg35, %c1_i32 : i32
        %226 = arith.cmpi eq, %224, %c3_i32 : i32
        %227 = arith.select %226, %225, %arg35 : i32
        %228 = arith.select %226, %c1_i32, %224 : i32
        %229 = ttg.memdesc_subview %152[%228, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %230 = ttg.memdesc_subview %153[%228] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %231 = ttg.memdesc_subview %154[%228] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %230, %227 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %232 = ttg.local_load %229 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %231, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %233 = arith.addi %arg36, %c1_i32 : i32
        %234 = arith.xori %arg37, %c1_i32 : i32
        %235 = arith.cmpi eq, %233, %c3_i32 : i32
        %236 = arith.select %235, %234, %arg37 : i32
        %237 = arith.select %235, %c1_i32, %233 : i32
        %238 = ttg.memdesc_subview %152[%237, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %239 = ttg.memdesc_subview %153[%237] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %240 = ttg.memdesc_subview %154[%237] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %239, %236 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %241 = ttg.local_load %238 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %240, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %242 = arith.subf %241, %232 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %243 = math.exp2 %242 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %137, %arg29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %244 = ttng.tmem_subslice %100 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %245 = ttng.tmem_subslice %100 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %246 = arith.xori %arg29, %c1_i32 : i32
        %247 = tt.expand_dims %243 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %248 = tt.broadcast %247 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_15 = ttng.tmem_load %244 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %249 = arith.mulf %result_15, %248 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %249, %244, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_16 = ttng.tmem_load %245 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %250 = arith.mulf %result_16, %248 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %250, %245, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %135, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %195, %218, %223, %246, %200, %199, %209, %208, %228, %227, %237, %236 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_yield %194#0, %194#1, %194#2, %194#3 : i32, i32, i32, i32
    }
    partition0(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem>, %arg66: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg67: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg68: f32, %arg69: i32) num_warps(1) {
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %false = arith.constant false
      %194 = arith.cmpi slt, %arg55, %arg56 : i32
      %195 = ttg.memdesc_subview %arg25[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %196 = ttg.memdesc_subview %arg26[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %197 = ttg.memdesc_subview %arg27[%c0_i32_15, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %198 = ttg.memdesc_trans %197 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      ttng.wait_barrier %196, %c0_i32_15, %194 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg31, %c0_i32_15, %194 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg32, %198, %arg33, %false, %194, %195[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %arg43, %c0_i32_15, %194 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg44, %198, %arg45, %false, %194, %195[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %199:10 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_14 iter_args(%arg71 = %c0_i32_15, %arg72 = %c0_i32_15, %arg73 = %c0_i32_15, %arg74 = %c0_i32_15, %arg75 = %c0_i32_15, %arg76 = %c0_i32_15, %arg77 = %c0_i32_15, %arg78 = %c0_i32_15, %arg79 = %c0_i32_15, %arg80 = %c0_i32_15) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %200 = arith.subi %arg56, %c128_i32_14 : i32
        %201 = arith.cmpi slt, %arg70, %200 : i32
        %202 = ttg.memdesc_subview %arg28[%arg73] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %203 = ttg.memdesc_subview %arg29[%arg73] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %204 = ttg.memdesc_subview %arg30[%arg73, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %205 = arith.xori %arg76, %c1_i32_16 : i32
        %206 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %203, %arg74 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg38, %205, %true_17 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg39, %arg77 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %206, %204, %arg40, %true_17, %true_17, %202[%true_17], %arg41[%true_17], %arg42[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %207 = arith.addi %arg73, %c1_i32_16 : i32
        %208 = arith.xori %arg74, %c1_i32_16 : i32
        %209 = arith.cmpi eq, %207, %c2_i32_13 : i32
        %210 = arith.select %209, %c0_i32_15, %207 : i32
        %211 = arith.select %209, %208, %arg74 : i32
        %212 = arith.xori %arg77, %c1_i32_16 : i32
        %213 = arith.xori %arg80, %c1_i32_16 : i32
        %214 = arith.xori %arg75, %c1_i32_16 : i32
        %215 = arith.addi %arg71, %c1_i32_16 : i32
        %216 = arith.xori %arg72, %c1_i32_16 : i32
        %217 = arith.cmpi eq, %215, %c2_i32_13 : i32
        %218 = arith.select %217, %c0_i32_15, %215 : i32
        %219 = arith.select %217, %216, %arg72 : i32
        %220 = ttg.memdesc_subview %arg25[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %221 = ttg.memdesc_subview %arg26[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %222 = ttg.memdesc_subview %arg27[%218, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %223 = ttg.memdesc_trans %222 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
        ttng.wait_barrier %221, %219, %201 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg31, %214, %201 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg32, %223, %arg33, %false, %201, %220[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %224 = arith.xori %arg79, %c1_i32_16 : i32
        %225 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg50, %224, %true_17 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg51, %arg80 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %225, %204, %arg52, %true_17, %true_17, %202[%true_17], %arg53[%true_17], %arg54[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %226 = arith.xori %arg78, %c1_i32_16 : i32
        ttng.wait_barrier %arg43, %226, %201 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg44, %223, %arg45, %false, %201, %220[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %218, %219, %210, %211, %214, %205, %212, %226, %224, %213 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition1(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem>, %arg66: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg67: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg68: f32, %arg69: i32) num_warps(2) {
      %c256_i32_13 = arith.constant 256 : i32
      %c2_i32_14 = arith.constant 2 : i32
      %c128_i32_15 = arith.constant 128 : i32
      %c0_i32_16 = arith.constant 0 : i32
      %c1_i32_17 = arith.constant 1 : i32
      %true_18 = arith.constant true
      %194 = arith.cmpi slt, %arg55, %arg56 : i32
      %195 = ttg.memdesc_subview %arg25[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %195, %c0_i32_16, %194 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %196 = ttg.memdesc_subview %arg26[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %196, 32768 {ttg.assigned_cluster = 2 : i32}, %194 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %197 = ttg.memdesc_subview %arg27[%c0_i32_16, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %198 = ttng.tensor_desc_to_tma_ptr %arg58 {ttg.assigned_cluster = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %198[%arg57, %c0_i32_16] %197, %196, %194 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %199 = arith.addi %arg55, %c128_i32_15 : i32
      %200 = arith.cmpi slt, %199, %arg56 : i32
      %201 = arith.addi %arg57, %c128_i32_15 : i32
      %202 = ttg.memdesc_subview %arg25[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %202, %c0_i32_16, %200 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %203 = ttg.memdesc_subview %arg26[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %203, 32768 {ttg.assigned_cluster = 2 : i32}, %200 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %204 = ttg.memdesc_subview %arg27[%c1_i32_17, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.async_tma_copy_global_to_local %198[%201, %c0_i32_16] %204, %203, %200 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %205:6 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %201, %arg72 = %c1_i32_17, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16, %arg76 = %arg57) -> (i32, i32, i32, i32, i32, i32)  : i32 {
        %206 = arith.subi %arg56, %c256_i32_13 : i32
        %207 = arith.cmpi slt, %arg70, %206 : i32
        %208 = ttg.memdesc_subview %arg28[%arg74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %208, %arg75 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %209 = ttg.memdesc_subview %arg29[%arg74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %209, 32768 {ttg.assigned_cluster = 0 : i32}, %true_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %210 = ttg.memdesc_subview %arg30[%arg74, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %211 = ttng.tensor_desc_to_tma_ptr %arg59 {ttg.assigned_cluster = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %211[%arg76, %c0_i32_16] %210, %209, %true_18 {ttg.assigned_cluster = 0 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %212 = arith.addi %arg74, %c1_i32_17 : i32
        %213 = arith.xori %arg75, %c1_i32_17 : i32
        %214 = arith.cmpi eq, %212, %c2_i32_14 : i32
        %215 = arith.select %214, %c0_i32_16, %212 : i32
        %216 = arith.select %214, %213, %arg75 : i32
        %217 = arith.addi %arg71, %c128_i32_15 : i32
        %218 = arith.addi %arg72, %c1_i32_17 : i32
        %219 = arith.xori %arg73, %c1_i32_17 : i32
        %220 = arith.cmpi eq, %218, %c2_i32_14 : i32
        %221 = arith.select %220, %c0_i32_16, %218 : i32
        %222 = arith.select %220, %219, %arg73 : i32
        %223 = ttg.memdesc_subview %arg25[%221] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %223, %222, %207 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %224 = ttg.memdesc_subview %arg26[%221] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %224, 32768 {ttg.assigned_cluster = 2 : i32}, %207 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %225 = ttg.memdesc_subview %arg27[%221, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        ttng.async_tma_copy_global_to_local %198[%217, %c0_i32_16] %225, %224, %207 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        scf.yield %217, %221, %222, %215, %216, %arg71 : i32, i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition2(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem>, %arg66: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg67: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg68: f32, %arg69: i32) num_warps(4) {
      %c3_i32_13 = arith.constant 3 : i32
      %cst_14 = arith.constant dense<-1.000000e+06> : tensor<128x128xf32, #blocked>
      %c128_i32_15 = arith.constant 128 : i32
      %c0_i32_16 = arith.constant 0 : i32
      %c1_i32_17 = arith.constant 1 : i32
      %true_18 = arith.constant true
      %cst_19 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %194 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.splat %arg69 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %196 = arith.addi %195, %194 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %197 = tt.splat %arg68 : f32 -> tensor<128x128xf32, #blocked>
      %198 = tt.expand_dims %196 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %199 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %200 = tt.expand_dims %199 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %201 = tt.broadcast %198 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %202 = ttg.local_load %arg61 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %203 = ttg.local_load %arg60 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %204:6 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %203, %arg72 = %202, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16, %arg76 = %c0_i32_16) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32)  : i32 {
        %205 = tt.splat %arg70 : i32 -> tensor<1x128xi32, #blocked>
        %206 = arith.addi %205, %200 : tensor<1x128xi32, #blocked>
        %207 = tt.broadcast %206 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %208 = arith.cmpi sge, %201, %207 : tensor<128x128xi32, #blocked>
        ttng.wait_barrier %arg34, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg33 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %209 = arith.xori %arg73, %c1_i32_17 : i32
        %210 = arith.mulf %result_20, %197 : tensor<128x128xf32, #blocked>
        %211 = arith.select %208, %cst_19, %cst_14 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %212 = arith.addf %210, %211 : tensor<128x128xf32, #blocked>
        %213 = "tt.reduce"(%212) <{axis = 1 : i32}> ({
        ^bb0(%arg77: f32, %arg78: f32):
          %235 = arith.maxnumf %arg77, %arg78 : f32
          tt.reduce.return %235 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %214 = arith.maxnumf %arg72, %213 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = arith.addi %arg75, %c1_i32_17 : i32
        %216 = arith.xori %arg76, %c1_i32_17 : i32
        %217 = arith.cmpi eq, %215, %c3_i32_13 : i32
        %218 = arith.select %217, %216, %arg76 : i32
        %219 = arith.select %217, %c1_i32_17, %215 : i32
        %220 = ttg.memdesc_subview %arg35[%219, %c0_i32_16] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %221 = ttg.memdesc_subview %arg36[%219] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %222 = ttg.memdesc_subview %arg37[%219] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %222, %218 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %214, %220 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %221, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %223 = tt.expand_dims %214 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %224 = tt.broadcast %223 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %225 = arith.subf %212, %224 : tensor<128x128xf32, #blocked>
        %226 = math.exp2 %225 : tensor<128x128xf32, #blocked>
        %227 = arith.subf %arg72, %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %228 = math.exp2 %227 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %229 = "tt.reduce"(%226) <{axis = 1 : i32}> ({
        ^bb0(%arg77: f32, %arg78: f32):
          %235 = arith.addf %arg77, %arg78 : f32
          tt.reduce.return %235 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %230 = arith.truncf %226 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %231 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg42, %arg74 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %230, %231, %true_18 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %232 = arith.mulf %arg71, %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %233 = arith.addf %232, %229 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %234 = arith.xori %arg74, %c1_i32_17 : i32
        scf.yield %233, %214, %209, %234, %219, %218 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %204#0, %arg62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %204#1, %arg63 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    }
    partition3(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem>, %arg66: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg67: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg68: f32, %arg69: i32) num_warps(4) {
      %c3_i32_13 = arith.constant 3 : i32
      %cst_14 = arith.constant dense<-1.000000e+06> : tensor<128x128xf32, #blocked>
      %c128_i32_15 = arith.constant 128 : i32
      %c0_i32_16 = arith.constant 0 : i32
      %c1_i32_17 = arith.constant 1 : i32
      %true_18 = arith.constant true
      %cst_19 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
      %194 = tt.splat %arg69 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %196 = arith.addi %194, %195 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %197 = tt.splat %arg68 : f32 -> tensor<128x128xf32, #blocked>
      %198 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %199 = tt.expand_dims %198 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %200 = tt.expand_dims %196 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %201 = tt.broadcast %200 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %202 = ttg.local_load %arg65 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %203 = ttg.local_load %arg64 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %204:6 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %203, %arg72 = %202, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16, %arg76 = %c0_i32_16) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32)  : i32 {
        %205 = tt.splat %arg70 : i32 -> tensor<1x128xi32, #blocked>
        %206 = arith.addi %205, %199 : tensor<1x128xi32, #blocked>
        %207 = tt.broadcast %206 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %208 = arith.cmpi sge, %201, %207 : tensor<128x128xi32, #blocked>
        ttng.wait_barrier %arg46, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg45 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %209 = arith.xori %arg73, %c1_i32_17 : i32
        %210 = arith.mulf %result_20, %197 : tensor<128x128xf32, #blocked>
        %211 = arith.select %208, %cst_19, %cst_14 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %212 = arith.addf %210, %211 : tensor<128x128xf32, #blocked>
        %213 = "tt.reduce"(%212) <{axis = 1 : i32}> ({
        ^bb0(%arg77: f32, %arg78: f32):
          %235 = arith.maxnumf %arg77, %arg78 : f32
          tt.reduce.return %235 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %214 = arith.maxnumf %arg72, %213 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = arith.addi %arg75, %c1_i32_17 : i32
        %216 = arith.xori %arg76, %c1_i32_17 : i32
        %217 = arith.cmpi eq, %215, %c3_i32_13 : i32
        %218 = arith.select %217, %216, %arg76 : i32
        %219 = arith.select %217, %c1_i32_17, %215 : i32
        %220 = ttg.memdesc_subview %arg47[%219, %c0_i32_16] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %221 = ttg.memdesc_subview %arg48[%219] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %222 = ttg.memdesc_subview %arg49[%219] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %222, %218 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %214, %220 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %221, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %223 = tt.expand_dims %214 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %224 = tt.broadcast %223 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %225 = arith.subf %212, %224 : tensor<128x128xf32, #blocked>
        %226 = math.exp2 %225 : tensor<128x128xf32, #blocked>
        %227 = arith.subf %arg72, %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %228 = math.exp2 %227 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %229 = "tt.reduce"(%226) <{axis = 1 : i32}> ({
        ^bb0(%arg77: f32, %arg78: f32):
          %235 = arith.addf %arg77, %arg78 : f32
          tt.reduce.return %235 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %230 = arith.truncf %226 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %231 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg54, %arg74 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %230, %231, %true_18 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %232 = arith.mulf %arg71, %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %233 = arith.addf %232, %229 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %234 = arith.xori %arg74, %c1_i32_17 : i32
        scf.yield %233, %214, %209, %234, %219, %218 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %204#0, %arg66 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %204#1, %arg67 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, i32, i32, !tt.tensordesc<tensor<128x128xf16, #shared>>, !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, f32, i32) -> (i32, i32, i32, i32)
    %167 = ttg.local_load %165 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    %168 = ttg.local_load %165 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    ttg.local_dealloc %165 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %169 = ttg.local_load %164 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %164 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %170 = ttg.local_load %163 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    %171 = ttg.local_load %163 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    ttg.local_dealloc %163 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %172 = ttg.local_load %162 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %162 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %152 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %153 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %154 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %142 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %146 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %147 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %148 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %149 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %150 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %151 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %143 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %144 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %156 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %157 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %158 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %159 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %160 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %161 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %137, %166#3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %141 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %140 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %139 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %138 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %137 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %136 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %135 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %134 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %133, %166#2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %132 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %131 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %130 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %125, %166#1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %129 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %128 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %127 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %126 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %125 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %124 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %123 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %122 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %121, %166#0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %120 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %119 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %118 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %116 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %117 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %115 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %114 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %112 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %111 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %107 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %108 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %106 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %104 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %103 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %102 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %173 = math.log2 %170 : tensor<128xf32, #blocked1>
    %174 = arith.addf %172, %173 : tensor<128xf32, #blocked1>
    %175 = tt.expand_dims %171 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %176 = tt.broadcast %175 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %177 = arith.muli %1, %arg24 : i32
    %178 = tt.addptr %arg1, %177 : !tt.ptr<f32>, i32
    %179 = tt.splat %178 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %180 = tt.addptr %179, %10 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    tt.store %180, %174 : tensor<128x!tt.ptr<f32>, #blocked1>
    %result_11 = ttng.tmem_load %101 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %181 = arith.divf %result_11, %176 : tensor<128x128xf32, #blocked>
    %182 = arith.truncf %181 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %183 = ttg.local_alloc %182 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %184 = ttng.tensor_desc_to_tma_ptr %arg19 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %184[%7, %c0_i32] %183 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %185 = math.log2 %167 : tensor<128xf32, #blocked1>
    %186 = arith.addf %169, %185 : tensor<128xf32, #blocked1>
    %187 = tt.expand_dims %168 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %188 = tt.broadcast %187 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %189 = tt.addptr %179, %12 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    tt.store %189, %186 : tensor<128x!tt.ptr<f32>, #blocked1>
    %result_12 = ttng.tmem_load %100 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %190 = arith.divf %result_12, %188 : tensor<128x128xf32, #blocked>
    %191 = arith.truncf %190 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %192 = ttg.local_alloc %191 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %193 = ttng.tensor_desc_to_tma_ptr %arg19 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %193[%17, %c0_i32] %192 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

