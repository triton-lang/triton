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
    ttng.async_tma_copy_global_to_local %16[%17, %c0_i32] %18, %19, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %19, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %19 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %20 = ttg.memdesc_subview %result[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %20, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %21 = ttg.memdesc_subview %result_2[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %21, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %24 = ttg.memdesc_subview %23[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %24, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %25 = ttg.memdesc_subview %23[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %26 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %27 = ttg.memdesc_subview %26[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %28 = ttg.memdesc_subview %26[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %28, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %29 = ttg.memdesc_subview %result_3[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_4 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %30 = ttg.memdesc_subview %result_4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.arrive_barrier %24, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %31 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %32 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %33 = ttg.memdesc_subview %32[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %33, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %34 = ttg.memdesc_subview %32[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %34, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %35 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %36 = ttg.memdesc_subview %35[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %36, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %37 = ttg.memdesc_subview %35[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %33, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %34, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %38 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %39 = ttg.memdesc_subview %38[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %40 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %41 = ttg.memdesc_subview %40[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %41, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %43 = ttg.memdesc_subview %42[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %45 = ttg.memdesc_subview %44[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %45, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %46 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.memdesc_subview %46[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %49 = ttg.memdesc_subview %48[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %46, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %50 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %51 = ttg.memdesc_subview %50[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %52 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %53 = ttg.memdesc_subview %52[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %53, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %53, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %54 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %55 = ttg.memdesc_subview %54[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %56 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %57 = ttg.memdesc_subview %56[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %57, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %57, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %58 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %59 = ttg.memdesc_subview %58[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %59, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %60 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %61 = ttg.memdesc_subview %60[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %61, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %58, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %62 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %63 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %64 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %65 = ttg.memdesc_subview %62[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_1, %65 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %66 = ttg.memdesc_subview %63[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %67 = ttg.memdesc_subview %64[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %66, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %67, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %66, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %68 = ttg.memdesc_subview %63[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %69 = ttg.memdesc_subview %64[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %68, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %69, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %69, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %70 = ttg.memdesc_subview %63[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %71 = ttg.memdesc_subview %64[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %70, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %71, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %71, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %72 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %73 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %74 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %75 = ttg.memdesc_subview %72[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_1, %75 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %76 = ttg.memdesc_subview %73[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %77 = ttg.memdesc_subview %74[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %76, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %77, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %76, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %78 = ttg.memdesc_subview %73[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %79 = ttg.memdesc_subview %74[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %78, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %79, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %79, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %80 = ttg.memdesc_subview %73[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %81 = ttg.memdesc_subview %74[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %80, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %81, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %81, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %82 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %83 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %84 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %85 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %86:4 = ttg.warp_specialize(%23, %26, %22, %32, %35, %31, %41, %14, %29, %39, %62, %63, %64, %43, %48, %21, %45, %46, %53, %18, %30, %51, %72, %73, %74, %55, %60, %20, %57, %58, %6, %5, %arg9, %arg14, %83, %82, %85, %84, %13) attributes {requestedRegisters = array<i32: 24, 24, 200, 200>}
    default {
      %192:5 = scf.for %arg25 = %c0_i32 to %6 step %c128_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c-1_i32, %arg30 = %c0_i32) -> (i32, i32, i32, i32, i32)  : i32 {
        %193 = arith.xori %arg26, %c1_i32 : i32
        %194 = arith.addi %arg27, %c1_i32 : i32
        %195 = arith.xori %arg28, %c1_i32 : i32
        %196 = arith.cmpi eq, %194, %c3_i32 : i32
        %197 = arith.select %196, %195, %arg28 : i32
        %198 = arith.select %196, %c1_i32, %194 : i32
        %199 = ttg.memdesc_subview %62[%198, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %200 = ttg.memdesc_subview %63[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %201 = ttg.memdesc_subview %64[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %200, %197 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %202 = ttg.local_load %199 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %201, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %203 = arith.addi %arg29, %c1_i32 : i32
        %204 = arith.xori %arg30, %c1_i32 : i32
        %205 = arith.cmpi eq, %203, %c3_i32 : i32
        %206 = arith.select %205, %204, %arg30 : i32
        %207 = arith.select %205, %c1_i32, %203 : i32
        %208 = ttg.memdesc_subview %62[%207, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %209 = ttg.memdesc_subview %63[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %210 = ttg.memdesc_subview %64[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %209, %206 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %211 = ttg.local_load %208 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %210, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %212 = arith.subf %211, %202 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %213 = math.exp2 %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %45, %arg26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %214 = ttng.tmem_subslice %21 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %215 = ttng.tmem_subslice %21 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %216 = tt.expand_dims %213 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %217 = tt.broadcast %216 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_13 = ttng.tmem_load %214 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %218 = arith.mulf %result_13, %217 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %218, %214, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_14 = ttng.tmem_load %215 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %219 = arith.mulf %result_14, %217 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %219, %215, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %220 = ttg.memdesc_subview %72[%198, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %221 = ttg.memdesc_subview %73[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %222 = ttg.memdesc_subview %74[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %221, %197 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %223 = ttg.local_load %220 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %222, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %224 = ttg.memdesc_subview %72[%207, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %225 = ttg.memdesc_subview %73[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %226 = ttg.memdesc_subview %74[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %225, %206 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %227 = ttg.local_load %224 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %226, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %228 = arith.subf %227, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %229 = math.exp2 %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %57, %arg26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %230 = ttng.tmem_subslice %20 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %231 = ttng.tmem_subslice %20 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %232 = tt.expand_dims %229 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %233 = tt.broadcast %232 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_15 = ttng.tmem_load %230 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %234 = arith.mulf %result_15, %233 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %234, %230, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_16 = ttng.tmem_load %231 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %235 = arith.mulf %result_16, %233 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %235, %231, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %193, %198, %197, %207, %206 : i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_yield %192#0, %192#0, %192#0, %192#0 : i32, i32, i32, i32
    }
    partition0(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: f32) num_warps(1) {
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %false = arith.constant false
      %192 = arith.cmpi sgt, %arg55, %c0_i32_15 : i32
      %193 = ttg.memdesc_subview %arg25[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %194 = ttg.memdesc_subview %arg26[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %195 = ttg.memdesc_subview %arg27[%c0_i32_15, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %196 = ttg.memdesc_trans %195 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      ttng.wait_barrier %194, %c0_i32_15, %192 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg31, %c0_i32_15, %192 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg32, %196, %arg33, %false, %192, %193[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %arg43, %c0_i32_15, %192 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg44, %196, %arg45, %false, %192, %193[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %197:3 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %c0_i32_15, %arg66 = %c0_i32_15, %arg67 = %c0_i32_15) -> (i32, i32, i32)  : i32 {
        %198 = arith.subi %arg55, %c128_i32_14 : i32
        %199 = arith.cmpi slt, %arg64, %198 : i32
        %200 = ttg.memdesc_subview %arg28[%arg65] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %201 = ttg.memdesc_subview %arg29[%arg65] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %202 = ttg.memdesc_subview %arg30[%arg65, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %203 = arith.xori %arg67, %c1_i32_16 : i32
        %204 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %201, %arg66 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg38, %203, %true_17 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg39, %arg67 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %204, %202, %arg40, %true_17, %true_17, %200[%true_17], %arg41[%true_17], %arg42[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %205 = arith.addi %arg65, %c1_i32_16 : i32
        %206 = arith.xori %arg66, %c1_i32_16 : i32
        %207 = arith.cmpi eq, %205, %c2_i32_13 : i32
        %208 = arith.select %207, %c0_i32_15, %205 : i32
        %209 = arith.select %207, %206, %arg66 : i32
        %210 = ttg.memdesc_subview %arg25[%208] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %211 = ttg.memdesc_subview %arg26[%208] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %212 = ttg.memdesc_subview %arg27[%208, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %213 = ttg.memdesc_trans %212 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
        ttng.wait_barrier %211, %209, %199 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg31, %203, %199 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg32, %213, %arg33, %false, %199, %210[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %214 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg50, %203, %true_17 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg51, %arg67 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %214, %202, %arg52, %true_17, %true_17, %200[%true_17], %arg53[%true_17], %arg54[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg43, %203, %199 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg44, %213, %arg45, %false, %199, %210[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %208, %209, %203 : i32, i32, i32
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
      %192 = arith.cmpi sgt, %arg55, %c0_i32_16 : i32
      %193 = ttg.memdesc_subview %arg25[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %193, %c0_i32_16, %192 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %194 = ttg.memdesc_subview %arg26[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %194, 32768 {ttg.assigned_cluster = 2 : i32}, %192 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %195 = ttg.memdesc_subview %arg27[%c0_i32_16, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %196 = ttng.tensor_desc_to_tma_ptr %arg57 {ttg.assigned_cluster = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %196[%arg56, %c0_i32_16] %195, %194, %192 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %197 = arith.cmpi sgt, %arg55, %c128_i32_15 : i32
      %198 = arith.addi %arg56, %c128_i32_15 : i32
      %199 = ttg.memdesc_subview %arg25[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %199, %c0_i32_16, %197 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %200 = ttg.memdesc_subview %arg26[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %200, 32768 {ttg.assigned_cluster = 2 : i32}, %197 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %201 = ttg.memdesc_subview %arg27[%c1_i32_17, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.async_tma_copy_global_to_local %196[%198, %c0_i32_16] %201, %200, %197 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %202:6 = scf.for %arg64 = %c0_i32_16 to %arg55 step %c128_i32_15 iter_args(%arg65 = %198, %arg66 = %c1_i32_17, %arg67 = %c0_i32_16, %arg68 = %c0_i32_16, %arg69 = %c0_i32_16, %arg70 = %arg56) -> (i32, i32, i32, i32, i32, i32)  : i32 {
        %203 = arith.subi %arg55, %c256_i32_13 : i32
        %204 = arith.cmpi slt, %arg64, %203 : i32
        %205 = ttg.memdesc_subview %arg28[%arg68] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %205, %arg69 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %206 = ttg.memdesc_subview %arg29[%arg68] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %206, 32768 {ttg.assigned_cluster = 0 : i32}, %true_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %207 = ttg.memdesc_subview %arg30[%arg68, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %208 = ttng.tensor_desc_to_tma_ptr %arg58 {ttg.assigned_cluster = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %208[%arg70, %c0_i32_16] %207, %206, %true_18 {ttg.assigned_cluster = 0 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %209 = arith.addi %arg68, %c1_i32_17 : i32
        %210 = arith.xori %arg69, %c1_i32_17 : i32
        %211 = arith.cmpi eq, %209, %c2_i32_14 : i32
        %212 = arith.select %211, %c0_i32_16, %209 : i32
        %213 = arith.select %211, %210, %arg69 : i32
        %214 = arith.addi %arg65, %c128_i32_15 : i32
        %215 = arith.addi %arg66, %c1_i32_17 : i32
        %216 = arith.xori %arg67, %c1_i32_17 : i32
        %217 = arith.cmpi eq, %215, %c2_i32_14 : i32
        %218 = arith.select %217, %c0_i32_16, %215 : i32
        %219 = arith.select %217, %216, %arg67 : i32
        %220 = ttg.memdesc_subview %arg25[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %220, %219, %204 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %221 = ttg.memdesc_subview %arg26[%218] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %221, 32768 {ttg.assigned_cluster = 2 : i32}, %204 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %222 = ttg.memdesc_subview %arg27[%218, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        ttng.async_tma_copy_global_to_local %196[%214, %c0_i32_16] %222, %221, %204 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        scf.yield %214, %218, %219, %212, %213, %arg65 : i32, i32, i32, i32, i32, i32
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
      %192 = tt.splat %arg63 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %193 = tt.splat %arg63 : f32 -> tensor<128x128xf32, #blocked>
      %194:5 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %cst_19, %arg66 = %cst_18, %arg67 = %c0_i32_15, %arg68 = %c0_i32_15, %arg69 = %c0_i32_15) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32)  : i32 {
        ttng.wait_barrier %arg34, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg33 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %195 = arith.xori %arg67, %c1_i32_16 : i32
        %196 = "tt.reduce"(%result_20) <{axis = 1 : i32}> ({
        ^bb0(%arg70: f32, %arg71: f32):
          %219 = arith.maxnumf %arg70, %arg71 : f32
          tt.reduce.return %219 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %197 = arith.mulf %196, %192 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %198 = arith.maxnumf %arg66, %197 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %199 = arith.addi %arg68, %c1_i32_16 : i32
        %200 = arith.xori %arg69, %c1_i32_16 : i32
        %201 = arith.cmpi eq, %199, %c3_i32_13 : i32
        %202 = arith.select %201, %200, %arg69 : i32
        %203 = arith.select %201, %c1_i32_16, %199 : i32
        %204 = ttg.memdesc_subview %arg35[%203, %c0_i32_15] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %205 = ttg.memdesc_subview %arg36[%203] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %206 = ttg.memdesc_subview %arg37[%203] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %206, %202 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %198, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %205, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %207 = arith.mulf %result_20, %193 : tensor<128x128xf32, #blocked>
        %208 = tt.expand_dims %198 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %209 = tt.broadcast %208 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %210 = arith.subf %207, %209 : tensor<128x128xf32, #blocked>
        %211 = math.exp2 %210 : tensor<128x128xf32, #blocked>
        %212 = arith.subf %arg66, %198 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %213 = math.exp2 %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %214 = "tt.reduce"(%211) <{axis = 1 : i32}> ({
        ^bb0(%arg70: f32, %arg71: f32):
          %219 = arith.addf %arg70, %arg71 : f32
          tt.reduce.return %219 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = arith.truncf %211 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %216 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg42, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %215, %216, %true_17 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %217 = arith.mulf %arg65, %213 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %218 = arith.addf %217, %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %218, %198, %195, %203, %202 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %194#0, %arg59 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %194#1, %arg60 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
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
      %192 = tt.splat %arg63 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %193 = tt.splat %arg63 : f32 -> tensor<128x128xf32, #blocked>
      %194:5 = scf.for %arg64 = %c0_i32_15 to %arg55 step %c128_i32_14 iter_args(%arg65 = %cst_19, %arg66 = %cst_18, %arg67 = %c0_i32_15, %arg68 = %c0_i32_15, %arg69 = %c0_i32_15) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32)  : i32 {
        ttng.wait_barrier %arg46, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg45 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %195 = arith.xori %arg67, %c1_i32_16 : i32
        %196 = "tt.reduce"(%result_20) <{axis = 1 : i32}> ({
        ^bb0(%arg70: f32, %arg71: f32):
          %219 = arith.maxnumf %arg70, %arg71 : f32
          tt.reduce.return %219 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %197 = arith.mulf %196, %192 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %198 = arith.maxnumf %arg66, %197 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %199 = arith.addi %arg68, %c1_i32_16 : i32
        %200 = arith.xori %arg69, %c1_i32_16 : i32
        %201 = arith.cmpi eq, %199, %c3_i32_13 : i32
        %202 = arith.select %201, %200, %arg69 : i32
        %203 = arith.select %201, %c1_i32_16, %199 : i32
        %204 = ttg.memdesc_subview %arg47[%203, %c0_i32_15] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %205 = ttg.memdesc_subview %arg48[%203] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %206 = ttg.memdesc_subview %arg49[%203] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %206, %202 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %198, %204 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %205, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %207 = arith.mulf %result_20, %193 : tensor<128x128xf32, #blocked>
        %208 = tt.expand_dims %198 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %209 = tt.broadcast %208 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %210 = arith.subf %207, %209 : tensor<128x128xf32, #blocked>
        %211 = math.exp2 %210 : tensor<128x128xf32, #blocked>
        %212 = arith.subf %arg66, %198 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %213 = math.exp2 %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %214 = "tt.reduce"(%211) <{axis = 1 : i32}> ({
        ^bb0(%arg70: f32, %arg71: f32):
          %219 = arith.addf %arg70, %arg71 : f32
          tt.reduce.return %219 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %215 = arith.truncf %211 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %216 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg54, %arg67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %215, %216, %true_17 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %217 = arith.mulf %arg65, %213 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %218 = arith.addf %217, %214 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %218, %198, %195, %203, %202 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %194#0, %arg61 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %194#1, %arg62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, i32, !tt.tensordesc<tensor<128x128xf16, #shared>>, !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, f32) -> (i32, i32, i32, i32)
    %87 = ttg.local_load %85 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %88 = ttg.local_alloc %87 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %85 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %89 = ttg.local_load %84 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %90 = ttg.local_alloc %89 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %84 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %91 = ttg.local_load %83 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %92 = ttg.local_alloc %91 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %83 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %93 = ttg.local_load %82 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %94 = ttg.local_alloc %93 : (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> !ttg.memdesc<128xf32, #shared1, #smem>
    ttg.local_dealloc %82 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %72 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %73 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %74 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %62 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %66 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %67 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %68 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %69 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %63 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %64 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %76 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %77 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %78 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %79 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %80 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %81 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %57, %86#3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %61 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %60 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %59 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %58 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %56 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %53, %86#2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %53 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %52 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %50 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %45, %86#1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %48 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %46 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %41, %86#0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %38 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %36 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %35 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %32 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %31 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %28 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %26 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %24 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %25 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %23 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %95 = arith.muli %0, %c256_i32 {tt.divisibility = dense<256> : tensor<1xi32>} : i32
    %96 = arith.addi %0, %c1_i32 : i32
    %97 = arith.muli %96, %c256_i32 : i32
    %98 = arith.addi %5, %95 : i32
    %result_5 = ttng.tmem_load %20 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_6 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %99 = ttg.memdesc_subview %result_6[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %result_5, %99, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_7 = ttng.tmem_load %21 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_8 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %100 = ttg.memdesc_subview %result_8[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %result_7, %100, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %101 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %102 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %103 = ttg.memdesc_subview %102[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %103, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %104 = ttg.memdesc_subview %102[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %104, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %105 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %106 = ttg.memdesc_subview %105[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %106, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %107 = ttg.memdesc_subview %105[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %107, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %result_9 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %108 = ttg.memdesc_subview %result_9[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_10 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %109 = ttg.memdesc_subview %result_10[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.arrive_barrier %103, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %104, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %110 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %111 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %112 = ttg.memdesc_subview %111[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %112, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %113 = ttg.memdesc_subview %111[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %113, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %114 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %115 = ttg.memdesc_subview %114[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %115, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %116 = ttg.memdesc_subview %114[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %116, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %112, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %113, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %117 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %118 = ttg.memdesc_subview %117[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %118, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %119 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %120 = ttg.memdesc_subview %119[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %120, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %120, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %121 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %122 = ttg.memdesc_subview %121[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %122, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %123 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %124 = ttg.memdesc_subview %123[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %124, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %124, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %122, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %125 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %126 = ttg.memdesc_subview %125[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %126, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %127 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %128 = ttg.memdesc_subview %127[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %128, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %125, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %129 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %130 = ttg.memdesc_subview %129[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %130, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %131 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %132 = ttg.memdesc_subview %131[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %132, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %132, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %133 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %134 = ttg.memdesc_subview %133[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %134, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %135 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %136 = ttg.memdesc_subview %135[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %136, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %136, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %134, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %137 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %138 = ttg.memdesc_subview %137[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %138, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %139 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %140 = ttg.memdesc_subview %139[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %140, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %137, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %141 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %142 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %143 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %144 = ttg.memdesc_subview %141[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %93, %144 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %145 = ttg.memdesc_subview %142[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %146 = ttg.memdesc_subview %143[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %145, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %146, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %145, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %147 = ttg.memdesc_subview %142[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %148 = ttg.memdesc_subview %143[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %147, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %148, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %148, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %149 = ttg.memdesc_subview %142[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %150 = ttg.memdesc_subview %143[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %149, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %150, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %150, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %151 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %152 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %153 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %154 = ttg.memdesc_subview %151[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %89, %154 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %155 = ttg.memdesc_subview %152[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %156 = ttg.memdesc_subview %153[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %155, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %156, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %155, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %157 = ttg.memdesc_subview %152[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %158 = ttg.memdesc_subview %153[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %157, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %158, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %158, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %159 = ttg.memdesc_subview %152[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %160 = ttg.memdesc_subview %153[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %159, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %160, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %160, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %161 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %162 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %163 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %164 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %165:4 = ttg.warp_specialize(%102, %105, %101, %111, %114, %110, %120, %14, %108, %118, %141, %142, %143, %122, %127, %100, %124, %125, %132, %18, %109, %130, %151, %152, %153, %134, %139, %99, %136, %137, %95, %97, %98, %arg9, %arg14, %92, %94, %162, %161, %88, %90, %164, %163, %13, %6) attributes {requestedRegisters = array<i32: 24, 24, 200, 200>}
    default {
      %192:5 = scf.for %arg25 = %95 to %97 step %c128_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c-1_i32, %arg30 = %c0_i32) -> (i32, i32, i32, i32, i32)  : i32 {
        %193 = arith.xori %arg26, %c1_i32 : i32
        %194 = arith.addi %arg27, %c1_i32 : i32
        %195 = arith.xori %arg28, %c1_i32 : i32
        %196 = arith.cmpi eq, %194, %c3_i32 : i32
        %197 = arith.select %196, %195, %arg28 : i32
        %198 = arith.select %196, %c1_i32, %194 : i32
        %199 = ttg.memdesc_subview %141[%198, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %200 = ttg.memdesc_subview %142[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %201 = ttg.memdesc_subview %143[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %200, %197 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %202 = ttg.local_load %199 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %201, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %203 = arith.addi %arg29, %c1_i32 : i32
        %204 = arith.xori %arg30, %c1_i32 : i32
        %205 = arith.cmpi eq, %203, %c3_i32 : i32
        %206 = arith.select %205, %204, %arg30 : i32
        %207 = arith.select %205, %c1_i32, %203 : i32
        %208 = ttg.memdesc_subview %141[%207, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %209 = ttg.memdesc_subview %142[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %210 = ttg.memdesc_subview %143[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %209, %206 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %211 = ttg.local_load %208 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %210, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %212 = arith.subf %211, %202 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %213 = math.exp2 %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %124, %arg26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %214 = ttng.tmem_subslice %100 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %215 = ttng.tmem_subslice %100 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %216 = tt.expand_dims %213 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %217 = tt.broadcast %216 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_13 = ttng.tmem_load %214 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %218 = arith.mulf %result_13, %217 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %218, %214, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_14 = ttng.tmem_load %215 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %219 = arith.mulf %result_14, %217 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %219, %215, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %122, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %220 = ttg.memdesc_subview %151[%198, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %221 = ttg.memdesc_subview %152[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %222 = ttg.memdesc_subview %153[%198] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %221, %197 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %223 = ttg.local_load %220 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %222, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %224 = ttg.memdesc_subview %151[%207, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %225 = ttg.memdesc_subview %152[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %226 = ttg.memdesc_subview %153[%207] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %225, %206 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %227 = ttg.local_load %224 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.arrive_barrier %226, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %228 = arith.subf %227, %223 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        %229 = math.exp2 %228 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>>
        ttng.wait_barrier %136, %arg26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %230 = ttng.tmem_subslice %99 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %231 = ttng.tmem_subslice %99 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %232 = tt.expand_dims %229 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xf32, #blocked2>
        %233 = tt.broadcast %232 : tensor<128x1xf32, #blocked2> -> tensor<128x64xf32, #blocked2>
        %result_15 = ttng.tmem_load %230 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %234 = arith.mulf %result_15, %233 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %234, %230, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        %result_16 = ttng.tmem_load %231 : !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked2>
        %235 = arith.mulf %result_16, %233 : tensor<128x64xf32, #blocked2>
        ttng.tmem_store %235, %231, %true : tensor<128x64xf32, #blocked2> -> !ttg.memdesc<128x64xf32, #tmem1, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %134, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %193, %198, %197, %207, %206 : i32, i32, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_yield %192#0, %192#0, %192#0, %192#0 : i32, i32, i32, i32
    }
    partition0(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg34: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg41: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg42: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg44: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg45: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg46: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg49: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg53: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: i32, %arg56: i32, %arg57: i32, %arg58: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg59: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg60: !ttg.memdesc<128xf32, #shared1, #smem>, %arg61: !ttg.memdesc<128xf32, #shared1, #smem>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem>, %arg66: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg67: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg68: f32, %arg69: i32) num_warps(1) {
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %c0_i32_15 = arith.constant 0 : i32
      %c1_i32_16 = arith.constant 1 : i32
      %true_17 = arith.constant true
      %false = arith.constant false
      %192 = arith.cmpi slt, %arg55, %arg56 : i32
      %193 = ttg.memdesc_subview %arg25[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %194 = ttg.memdesc_subview %arg26[%c0_i32_15] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %195 = ttg.memdesc_subview %arg27[%c0_i32_15, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %196 = ttg.memdesc_trans %195 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      ttng.wait_barrier %194, %c0_i32_15, %192 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg31, %c0_i32_15, %192 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg32, %196, %arg33, %false, %192, %193[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %arg43, %c0_i32_15, %192 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %arg44, %196, %arg45, %false, %192, %193[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %197:3 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_14 iter_args(%arg71 = %c0_i32_15, %arg72 = %c0_i32_15, %arg73 = %c0_i32_15) -> (i32, i32, i32)  : i32 {
        %198 = arith.subi %arg56, %c128_i32_14 : i32
        %199 = arith.cmpi slt, %arg70, %198 : i32
        %200 = ttg.memdesc_subview %arg28[%arg71] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %201 = ttg.memdesc_subview %arg29[%arg71] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %202 = ttg.memdesc_subview %arg30[%arg71, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %203 = arith.xori %arg73, %c1_i32_16 : i32
        %204 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %201, %arg72 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg38, %203, %true_17 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg39, %arg73 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %204, %202, %arg40, %true_17, %true_17, %200[%true_17], %arg41[%true_17], %arg42[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %205 = arith.addi %arg71, %c1_i32_16 : i32
        %206 = arith.xori %arg72, %c1_i32_16 : i32
        %207 = arith.cmpi eq, %205, %c2_i32_13 : i32
        %208 = arith.select %207, %c0_i32_15, %205 : i32
        %209 = arith.select %207, %206, %arg72 : i32
        %210 = ttg.memdesc_subview %arg25[%208] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %211 = ttg.memdesc_subview %arg26[%208] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %212 = ttg.memdesc_subview %arg27[%208, %c0_i32_15, %c0_i32_15] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %213 = ttg.memdesc_trans %212 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
        ttng.wait_barrier %211, %209, %199 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg31, %203, %199 {ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg32, %213, %arg33, %false, %199, %210[%true_17], %arg34[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %214 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg50, %203, %true_17 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg51, %arg73 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %214, %202, %arg52, %true_17, %true_17, %200[%true_17], %arg53[%true_17], %arg54[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %arg43, %203, %199 {ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tc_gen5_mma %arg44, %213, %arg45, %false, %199, %210[%true_17], %arg46[%true_17] {tt.self_latency = 1 : i32, ttg.assigned_cluster = 3 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %208, %209, %203 : i32, i32, i32
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
      %192 = arith.cmpi slt, %arg55, %arg56 : i32
      %193 = ttg.memdesc_subview %arg25[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %193, %c0_i32_16, %192 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %194 = ttg.memdesc_subview %arg26[%c0_i32_16] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %194, 32768 {ttg.assigned_cluster = 2 : i32}, %192 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %195 = ttg.memdesc_subview %arg27[%c0_i32_16, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %196 = ttng.tensor_desc_to_tma_ptr %arg58 {ttg.assigned_cluster = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %196[%arg57, %c0_i32_16] %195, %194, %192 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %197 = arith.addi %arg55, %c128_i32_15 : i32
      %198 = arith.cmpi slt, %197, %arg56 : i32
      %199 = arith.addi %arg57, %c128_i32_15 : i32
      %200 = ttg.memdesc_subview %arg25[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %200, %c0_i32_16, %198 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %201 = ttg.memdesc_subview %arg26[%c1_i32_17] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %201, 32768 {ttg.assigned_cluster = 2 : i32}, %198 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %202 = ttg.memdesc_subview %arg27[%c1_i32_17, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.async_tma_copy_global_to_local %196[%199, %c0_i32_16] %202, %201, %198 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %203:6 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %199, %arg72 = %c1_i32_17, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16, %arg76 = %arg57) -> (i32, i32, i32, i32, i32, i32)  : i32 {
        %204 = arith.subi %arg56, %c256_i32_13 : i32
        %205 = arith.cmpi slt, %arg70, %204 : i32
        %206 = ttg.memdesc_subview %arg28[%arg74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %206, %arg75 {ttg.assigned_cluster = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %207 = ttg.memdesc_subview %arg29[%arg74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %207, 32768 {ttg.assigned_cluster = 0 : i32}, %true_18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %208 = ttg.memdesc_subview %arg30[%arg74, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %209 = ttng.tensor_desc_to_tma_ptr %arg59 {ttg.assigned_cluster = 0 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %209[%arg76, %c0_i32_16] %208, %207, %true_18 {ttg.assigned_cluster = 0 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %210 = arith.addi %arg74, %c1_i32_17 : i32
        %211 = arith.xori %arg75, %c1_i32_17 : i32
        %212 = arith.cmpi eq, %210, %c2_i32_14 : i32
        %213 = arith.select %212, %c0_i32_16, %210 : i32
        %214 = arith.select %212, %211, %arg75 : i32
        %215 = arith.addi %arg71, %c128_i32_15 : i32
        %216 = arith.addi %arg72, %c1_i32_17 : i32
        %217 = arith.xori %arg73, %c1_i32_17 : i32
        %218 = arith.cmpi eq, %216, %c2_i32_14 : i32
        %219 = arith.select %218, %c0_i32_16, %216 : i32
        %220 = arith.select %218, %217, %arg73 : i32
        %221 = ttg.memdesc_subview %arg25[%219] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %221, %220, %205 {ttg.assigned_cluster = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %222 = ttg.memdesc_subview %arg26[%219] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %222, 32768 {ttg.assigned_cluster = 2 : i32}, %205 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %223 = ttg.memdesc_subview %arg27[%219, %c0_i32_16, %c0_i32_16] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        ttng.async_tma_copy_global_to_local %196[%215, %c0_i32_16] %223, %222, %205 {ttg.assigned_cluster = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        scf.yield %215, %219, %220, %213, %214, %arg71 : i32, i32, i32, i32, i32, i32
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
      %192 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %193 = tt.splat %arg69 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %194 = arith.addi %193, %192 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.splat %arg68 : f32 -> tensor<128x128xf32, #blocked>
      %196 = tt.expand_dims %194 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %197 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %198 = tt.expand_dims %197 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %199 = tt.broadcast %196 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %200 = ttg.local_load %arg61 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %201 = ttg.local_load %arg60 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %202:5 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %201, %arg72 = %200, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32)  : i32 {
        %203 = tt.splat %arg70 : i32 -> tensor<1x128xi32, #blocked>
        %204 = arith.addi %203, %198 : tensor<1x128xi32, #blocked>
        %205 = tt.broadcast %204 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %206 = arith.cmpi sge, %199, %205 : tensor<128x128xi32, #blocked>
        ttng.wait_barrier %arg34, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg33 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %207 = arith.xori %arg73, %c1_i32_17 : i32
        %208 = arith.mulf %result_20, %195 : tensor<128x128xf32, #blocked>
        %209 = arith.select %206, %cst_19, %cst_14 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %210 = arith.addf %208, %209 : tensor<128x128xf32, #blocked>
        %211 = "tt.reduce"(%210) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %232 = arith.maxnumf %arg76, %arg77 : f32
          tt.reduce.return %232 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %212 = arith.maxnumf %arg72, %211 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %213 = arith.addi %arg74, %c1_i32_17 : i32
        %214 = arith.xori %arg75, %c1_i32_17 : i32
        %215 = arith.cmpi eq, %213, %c3_i32_13 : i32
        %216 = arith.select %215, %214, %arg75 : i32
        %217 = arith.select %215, %c1_i32_17, %213 : i32
        %218 = ttg.memdesc_subview %arg35[%217, %c0_i32_16] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %219 = ttg.memdesc_subview %arg36[%217] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %220 = ttg.memdesc_subview %arg37[%217] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %220, %216 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %212, %218 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %219, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %221 = tt.expand_dims %212 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %222 = tt.broadcast %221 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %223 = arith.subf %210, %222 : tensor<128x128xf32, #blocked>
        %224 = math.exp2 %223 : tensor<128x128xf32, #blocked>
        %225 = arith.subf %arg72, %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %226 = math.exp2 %225 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %227 = "tt.reduce"(%224) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %232 = arith.addf %arg76, %arg77 : f32
          tt.reduce.return %232 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %228 = arith.truncf %224 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %229 = "ttg.memdesc_reinterpret"(%arg33) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg42, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %228, %229, %true_18 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %230 = arith.mulf %arg71, %226 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %231 = arith.addf %230, %227 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %231, %212, %207, %217, %216 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %202#0, %arg62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %202#1, %arg63 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
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
      %192 = tt.splat %arg69 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %193 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %194 = arith.addi %192, %193 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %195 = tt.splat %arg68 : f32 -> tensor<128x128xf32, #blocked>
      %196 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %197 = tt.expand_dims %196 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %198 = tt.expand_dims %194 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
      %199 = tt.broadcast %198 : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
      %200 = ttg.local_load %arg65 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %201 = ttg.local_load %arg64 : !ttg.memdesc<128xf32, #shared1, #smem> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %202:5 = scf.for %arg70 = %arg55 to %arg56 step %c128_i32_15 iter_args(%arg71 = %201, %arg72 = %200, %arg73 = %c0_i32_16, %arg74 = %c0_i32_16, %arg75 = %c0_i32_16) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32)  : i32 {
        %203 = tt.splat %arg70 : i32 -> tensor<1x128xi32, #blocked>
        %204 = arith.addi %203, %197 : tensor<1x128xi32, #blocked>
        %205 = tt.broadcast %204 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
        %206 = arith.cmpi sge, %199, %205 : tensor<128x128xi32, #blocked>
        ttng.wait_barrier %arg46, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_20 = ttng.tmem_load %arg45 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %arg43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %207 = arith.xori %arg73, %c1_i32_17 : i32
        %208 = arith.mulf %result_20, %195 : tensor<128x128xf32, #blocked>
        %209 = arith.select %206, %cst_19, %cst_14 : tensor<128x128xi1, #blocked>, tensor<128x128xf32, #blocked>
        %210 = arith.addf %208, %209 : tensor<128x128xf32, #blocked>
        %211 = "tt.reduce"(%210) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %232 = arith.maxnumf %arg76, %arg77 : f32
          tt.reduce.return %232 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %212 = arith.maxnumf %arg72, %211 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %213 = arith.addi %arg74, %c1_i32_17 : i32
        %214 = arith.xori %arg75, %c1_i32_17 : i32
        %215 = arith.cmpi eq, %213, %c3_i32_13 : i32
        %216 = arith.select %215, %214, %arg75 : i32
        %217 = arith.select %215, %c1_i32_17, %213 : i32
        %218 = ttg.memdesc_subview %arg47[%217, %c0_i32_16] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %219 = ttg.memdesc_subview %arg48[%217] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %220 = ttg.memdesc_subview %arg49[%217] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %220, %216 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %212, %218 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %219, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %221 = tt.expand_dims %212 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %222 = tt.broadcast %221 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %223 = arith.subf %210, %222 : tensor<128x128xf32, #blocked>
        %224 = math.exp2 %223 : tensor<128x128xf32, #blocked>
        %225 = arith.subf %arg72, %212 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %226 = math.exp2 %225 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %227 = "tt.reduce"(%224) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %232 = arith.addf %arg76, %arg77 : f32
          tt.reduce.return %232 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %228 = arith.truncf %224 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        %229 = "ttg.memdesc_reinterpret"(%arg45) : (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>) -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.wait_barrier %arg54, %arg73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.tmem_store %228, %229, %true_18 : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem2, #ttng.tensor_memory, mutable>
        ttng.arrive_barrier %arg51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %230 = arith.mulf %arg71, %226 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %231 = arith.addf %230, %227 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %231, %212, %207, %217, %216 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32
      } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %202#0, %arg66 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %202#1, %arg67 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, i32, i32, !tt.tensordesc<tensor<128x128xf16, #shared>>, !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, f32, i32) -> (i32, i32, i32, i32)
    %166 = ttg.local_load %164 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    %167 = ttg.local_load %164 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    ttg.local_dealloc %164 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %168 = ttg.local_load %163 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %163 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %169 = ttg.local_load %162 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    %170 = ttg.local_load %162 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    ttg.local_dealloc %162 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %171 = ttg.local_load %161 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %161 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %151 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %152 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %153 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %141 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %145 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %146 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %147 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %148 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %149 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %150 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %142 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %143 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %155 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %156 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %157 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %158 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %159 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %160 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %136, %165#3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %140 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %139 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %138 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %137 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %136 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %135 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %134 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %132, %165#2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %132 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %131 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %130 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %129 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %124, %165#1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %128 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %127 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %126 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %125 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %124 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %123 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %122 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %120, %165#0 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %120 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %119 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %118 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %117 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %115 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %116 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %114 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %112 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %111 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %110 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %106 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %107 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %105 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %103 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %104 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %102 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %101 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %172 = math.log2 %169 : tensor<128xf32, #blocked1>
    %173 = arith.addf %171, %172 : tensor<128xf32, #blocked1>
    %174 = tt.expand_dims %170 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %175 = tt.broadcast %174 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %176 = arith.muli %1, %arg24 : i32
    %177 = tt.addptr %arg1, %176 : !tt.ptr<f32>, i32
    %178 = tt.splat %177 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %179 = tt.addptr %178, %10 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    tt.store %179, %173 : tensor<128x!tt.ptr<f32>, #blocked1>
    %result_11 = ttng.tmem_load %100 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %180 = arith.divf %result_11, %175 : tensor<128x128xf32, #blocked>
    %181 = arith.truncf %180 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %182 = ttg.local_alloc %181 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %183 = ttng.tensor_desc_to_tma_ptr %arg19 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %183[%7, %c0_i32] %182 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %184 = math.log2 %166 : tensor<128xf32, #blocked1>
    %185 = arith.addf %168, %184 : tensor<128xf32, #blocked1>
    %186 = tt.expand_dims %167 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
    %187 = tt.broadcast %186 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %188 = tt.addptr %178, %12 : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    tt.store %188, %185 : tensor<128x!tt.ptr<f32>, #blocked1>
    %result_12 = ttng.tmem_load %99 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %189 = arith.divf %result_12, %187 : tensor<128x128xf32, #blocked>
    %190 = arith.truncf %189 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %191 = ttg.local_alloc %190 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %183[%17, %c0_i32] %191 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

