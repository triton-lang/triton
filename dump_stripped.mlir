#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel_data_part(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<128xi32, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32

    %2 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %3 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %3, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %4 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %4[%1, %c0_i32] %2, %3, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %3, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %5 = arith.addi %1, %c128_i32 : i32

    %6 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>

    %7 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %7, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %7, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %8 = ttng.tensor_desc_to_tma_ptr %arg0 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_global_to_local %8[%5, %c0_i32] %6, %7, %true : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.wait_barrier %7, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %9 = ttg.memdesc_subview %result_3[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_4 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %10 = ttg.memdesc_subview %result_4[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %10, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    ttng.tmem_store %cst_0, %9, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>

    %11 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>

    %12 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %13 = ttg.memdesc_subview %12[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %13, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %14 = ttg.memdesc_subview %12[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %14, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %16 = ttg.memdesc_subview %15[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %16, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %17 = ttg.memdesc_subview %15[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %17, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %13, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %14, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>

    %18 = ttg.local_alloc : () -> !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>

    %19 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %20 = ttg.memdesc_subview %19[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %20, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %21 = ttg.memdesc_subview %19[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %21, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %22 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %23 = ttg.memdesc_subview %22[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %24 = ttg.memdesc_subview %22[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %24, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %20, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %21, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %25 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %26 = ttg.memdesc_subview %25[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %27 = ttg.memdesc_subview %25[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %28 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %29 = ttg.memdesc_subview %28[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %30 = ttg.memdesc_subview %28[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %31 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %32 = ttg.memdesc_subview %31[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %32, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %33 = ttg.memdesc_subview %31[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %34 = ttg.local_alloc : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %35 = ttg.memdesc_subview %34[%c0_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %36 = ttg.memdesc_subview %34[%c1_i32] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.init_barrier %36, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.arrive_barrier %36, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    %37 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %38 = ttg.memdesc_subview %37[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %38, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %39 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %40 = ttg.memdesc_subview %39[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %38, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result_5 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    %41 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.memdesc_subview %41[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %43 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.memdesc_subview %43[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %result_6 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>

    %45 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>

    %46 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.memdesc_subview %46[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %49 = ttg.memdesc_subview %47[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %48, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %50 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>

    %51 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %52 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %53 = ttg.memdesc_subview %50[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst, %53 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %54 = ttg.memdesc_subview %51[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %55 = ttg.memdesc_subview %52[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %55, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %56 = ttg.memdesc_subview %51[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %57 = ttg.memdesc_subview %52[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %56, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %57, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %57, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %58 = ttg.memdesc_subview %51[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %59 = ttg.memdesc_subview %52[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %58, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %59, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %59, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    %60 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>

    %61 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %62 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %63 = ttg.memdesc_subview %61[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %64 = ttg.memdesc_subview %62[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %63, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %64, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %64, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %65 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>

    %66 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %67 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %68 = ttg.memdesc_subview %65[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst, %68 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %69 = ttg.memdesc_subview %66[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %70 = ttg.memdesc_subview %67[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %69, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %70, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %69, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %71 = ttg.memdesc_subview %66[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %72 = ttg.memdesc_subview %67[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %71, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %72, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %72, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %73 = ttg.memdesc_subview %66[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %74 = ttg.memdesc_subview %67[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %73, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %74, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %74, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>

    %75 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %76 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %77 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %78 = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>

    %79:6 = ttg.warp_specialize(%12, %15, %11, %19, %22, %18, %result, %25, %2, %28, %50, %51, %52, %45, %46, %47, %result_2, %31, %6, %34, %65, %66, %67, %60, %61, %62, %38, %result_5, %9, %40, %42, %result_6, %10, %44, %arg23, %arg5, %arg10, %76, %75, %78, %77, %arg24) attributes {requestedRegisters = array<i32: 24, 24, 192, 192>}
    default {
      %105:18 = scf.for %arg25 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %c0_i32, %arg27 = %c0_i32, %arg28 = %c0_i32, %arg29 = %c0_i32, %arg30 = %c0_i32, %arg31 = %c0_i32, %arg32 = %c-1_i32, %arg33 = %c0_i32, %arg34 = %c0_i32, %arg35 = %c0_i32, %arg36 = %c-1_i32, %arg37 = %c0_i32, %arg38 = %c-1_i32, %arg39 = %c0_i32, %arg40 = %c0_i32, %arg41 = %c0_i32, %arg42 = %c-1_i32, %arg43 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
        %106 = arith.addi %arg26, %c1_i32 : i32
        %107 = arith.xori %arg27, %c1_i32 : i32
        %108 = arith.cmpi eq, %106, %c2_i32 : i32
        %109 = arith.select %108, %c0_i32, %106 : i32
        %110 = arith.select %108, %107, %arg27 : i32
        %111 = arith.addi %arg34, %c1_i32 : i32
        %112 = arith.xori %arg35, %c1_i32 : i32
        %113 = arith.cmpi eq, %111, %c3_i32 : i32
        %114 = arith.select %113, %112, %arg35 : i32
        %115 = arith.select %113, %c1_i32, %111 : i32
        %116 = ttg.memdesc_subview %50[%115, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %117 = ttg.memdesc_subview %51[%115] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %118 = ttg.memdesc_subview %52[%115] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %117, %114 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %119 = ttg.local_load %116 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %118, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %120 = arith.addi %arg36, %c1_i32 : i32
        %121 = arith.xori %arg37, %c1_i32 : i32
        %122 = arith.cmpi eq, %120, %c3_i32 : i32
        %123 = arith.select %122, %121, %arg37 : i32
        %124 = arith.select %122, %c1_i32, %120 : i32
        %125 = ttg.memdesc_subview %50[%124, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %126 = ttg.memdesc_subview %51[%124] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %127 = ttg.memdesc_subview %52[%124] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %126, %123 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %128 = ttg.local_load %125 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %127, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %129 = arith.subf %128, %119 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %130 = math.exp2 %129 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %131 = tt.expand_dims %130 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %132 = tt.broadcast %131 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        ttng.wait_barrier %40, %arg30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_9 = ttng.tmem_load %9 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %133 = arith.xori %arg30, %c1_i32 : i32
        %134 = arith.mulf %result_9, %132 : tensor<128x128xf32, #blocked>
        %135 = arith.addi %arg28, %c1_i32 : i32
        %136 = arith.xori %arg29, %c1_i32 : i32
        %137 = arith.cmpi eq, %135, %c2_i32 : i32
        %138 = arith.select %137, %c0_i32, %135 : i32
        %139 = arith.select %137, %136, %arg29 : i32
        %140 = arith.addi %arg40, %c1_i32 : i32
        %141 = arith.xori %arg41, %c1_i32 : i32
        %142 = arith.cmpi eq, %140, %c3_i32 : i32
        %143 = arith.select %142, %141, %arg41 : i32
        %144 = arith.select %142, %c1_i32, %140 : i32
        %145 = ttg.memdesc_subview %65[%144, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %146 = ttg.memdesc_subview %66[%144] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %147 = ttg.memdesc_subview %67[%144] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %146, %143 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %148 = ttg.local_load %145 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %147, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %149 = arith.addi %arg42, %c1_i32 : i32
        %150 = arith.xori %arg43, %c1_i32 : i32
        %151 = arith.cmpi eq, %149, %c3_i32 : i32
        %152 = arith.select %151, %150, %arg43 : i32
        %153 = arith.select %151, %c1_i32, %149 : i32
        %154 = ttg.memdesc_subview %65[%153, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %155 = ttg.memdesc_subview %66[%153] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %156 = ttg.memdesc_subview %67[%153] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %155, %152 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %157 = ttg.local_load %154 : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        ttng.arrive_barrier %156, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %158 = arith.subf %157, %148 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %159 = math.exp2 %158 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %160 = tt.expand_dims %159 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %161 = tt.broadcast %160 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        ttng.wait_barrier %44, %arg31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %result_10 = ttng.tmem_load %10 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        %162 = arith.xori %arg31, %c1_i32 : i32
        %163 = arith.mulf %result_10, %161 : tensor<128x128xf32, #blocked>
        %164 = arith.addi %arg32, %c1_i32 : i32
        %165 = arith.xori %arg33, %c1_i32 : i32
        %166 = arith.cmpi eq, %164, %c1_i32 : i32
        %167 = arith.select %166, %165, %arg33 : i32
        %168 = arith.select %166, %c0_i32, %164 : i32
        %169 = ttg.memdesc_subview %45[%168, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        %170 = ttg.memdesc_subview %46[%168] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %171 = ttg.memdesc_subview %47[%168] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %170, %167 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %172 = ttg.local_load %169 : !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %171, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %173 = arith.truncf %172 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        ttng.tmem_store %173, %result_5, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
        ttng.tmem_store %134, %9, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        ttng.arrive_barrier %38, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %174 = arith.addi %arg38, %c1_i32 : i32
        %175 = arith.xori %arg39, %c1_i32 : i32
        %176 = arith.cmpi eq, %174, %c1_i32 : i32
        %177 = arith.select %176, %175, %arg39 : i32
        %178 = arith.select %176, %c0_i32, %174 : i32
        %179 = ttg.memdesc_subview %60[%178, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        %180 = ttg.memdesc_subview %61[%178] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %181 = ttg.memdesc_subview %62[%178] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %180, %177 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %182 = ttg.local_load %179 : !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
        ttng.arrive_barrier %181, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %183 = arith.truncf %182 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
        ttng.tmem_store %183, %result_6, %true : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
        ttng.tmem_store %163, %10, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        ttng.arrive_barrier %42, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        scf.yield %109, %110, %138, %139, %133, %162, %168, %167, %115, %114, %124, %123, %178, %177, %144, %143, %153, %152 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_yield %105#0, %105#1, %105#2, %105#3, %105#4, %105#5 : i32, i32, i32, i32, i32, i32
    }
    partition0(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg32: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg42: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg44: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg49: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg53: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg56: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg57: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg58: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg59: i32, %arg60: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg61: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg66: f32) num_warps(1) {
      %c768_i32 = arith.constant 768 : i32
      %c640_i32 = arith.constant 640 : i32
      %c512_i32 = arith.constant 512 : i32
      %c384_i32 = arith.constant 384 : i32
      %c256_i32_9 = arith.constant 256 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %false = arith.constant false
      %true_11 = arith.constant true
      %c1_i32_12 = arith.constant 1 : i32
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %105 = arith.cmpi sgt, %arg59, %c0_i32_10 : i32
      %106 = ttg.memdesc_subview %arg25[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %107 = ttg.memdesc_subview %arg26[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %108 = ttg.memdesc_subview %arg27[%c0_i32_10, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %109 = ttg.memdesc_trans %108 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
      ttng.wait_barrier %107, %c0_i32_10, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %110 = ttg.memdesc_subview %arg31[%c0_i32_10, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %111 = ttg.memdesc_subview %arg32[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %109, %110, %false, %105, %106[%true_11], %111[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %112 = ttg.memdesc_subview %arg34[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %112, %c0_i32_10, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %113 = arith.cmpi sgt, %arg59, %c128_i32_14 : i32
      %114 = ttg.memdesc_subview %arg25[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %115 = ttg.memdesc_subview %arg26[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %116 = ttg.memdesc_subview %arg27[%c1_i32_12, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %117 = ttg.memdesc_trans %116 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
      ttng.wait_barrier %115, %c0_i32_10, %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %118 = ttg.memdesc_subview %arg31[%c1_i32_12, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %119 = ttg.memdesc_subview %arg32[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %117, %118, %false, %113, %114[%true_11], %119[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %120 = ttg.memdesc_subview %arg34[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %120, %c1_i32_12, %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %121 = arith.cmpi sgt, %arg59, %c256_i32_9 : i32
      %122 = ttg.memdesc_subview %arg41[%c0_i32_10, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %123 = ttg.memdesc_subview %arg42[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg43, %109, %122, %false, %105, %106[%true_11], %123[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %124 = ttg.memdesc_subview %arg44[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %124, %c0_i32_10, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %107, %c1_i32_12, %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %109, %110, %false, %121, %106[%true_11], %111[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %112, %c1_i32_12, %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %125 = arith.cmpi sgt, %arg59, %c384_i32 : i32
      %126 = ttg.memdesc_subview %arg41[%c1_i32_12, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %127 = ttg.memdesc_subview %arg42[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg43, %117, %126, %false, %113, %114[%true_11], %127[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %128 = ttg.memdesc_subview %arg44[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %128, %c1_i32_12, %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %115, %c1_i32_12, %125 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %117, %118, %false, %125, %114[%true_11], %119[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %120, %c0_i32_10, %125 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %129 = arith.cmpi sgt, %arg59, %c512_i32 : i32
      %130 = ttg.memdesc_subview %arg28[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %131 = ttg.memdesc_subview %arg29[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %132 = ttg.memdesc_subview %arg30[%c0_i32_10, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.wait_barrier %131, %c0_i32_10, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg51, %c1_i32_12, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg52, %132, %arg53, %true_11, %105, %130[%true_11], %arg54[%true_11] : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg43, %109, %122, %false, %121, %106[%true_11], %123[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %124, %c1_i32_12, %121 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %107, %c0_i32_10, %129 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %109, %110, %false, %129, %106[%true_11], %111[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %112, %c0_i32_10, %129 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %133 = arith.cmpi sgt, %arg59, %c640_i32 : i32
      %134 = ttg.memdesc_subview %arg28[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %135 = ttg.memdesc_subview %arg29[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %136 = ttg.memdesc_subview %arg30[%c1_i32_12, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.wait_barrier %135, %c0_i32_10, %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %arg51, %c0_i32_10, %113 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg52, %136, %arg53, %true_11, %113, %134[%true_11], %arg54[%true_11] : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg43, %117, %126, %false, %125, %114[%true_11], %127[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %128, %c0_i32_10, %125 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %115, %c0_i32_10, %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.fence_async_shared {bCluster = false}
      ttng.tc_gen5_mma %arg33, %117, %118, %false, %133, %114[%true_11], %119[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %120, %c1_i32_12, %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %137:18 = scf.for %arg67 = %c0_i32_10 to %arg59 step %c128_i32_14 iter_args(%arg68 = %c1_i32_12, %arg69 = %c0_i32_10, %arg70 = %c1_i32_12, %arg71 = %c0_i32_10, %arg72 = %c0_i32_10, %arg73 = %c1_i32_12, %arg74 = %c0_i32_10, %arg75 = %c0_i32_10, %arg76 = %c0_i32_10, %arg77 = %c0_i32_10, %arg78 = %132, %arg79 = %136, %arg80 = %130, %arg81 = %134, %arg82 = %109, %arg83 = %117, %arg84 = %106, %arg85 = %114) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>)  : i32 {
        %138 = arith.subi %arg59, %c768_i32 : i32
        %139 = arith.cmpi slt, %arg67, %138 : i32
        %140 = arith.subi %arg59, %c512_i32 : i32
        %141 = arith.cmpi slt, %arg67, %140 : i32
        %142 = arith.subi %arg59, %c256_i32_9 : i32
        %143 = arith.cmpi slt, %arg67, %142 : i32
        %144 = arith.xori %arg77, %c1_i32_12 : i32
        ttng.wait_barrier %arg55, %144, %true_11 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        ttng.tc_gen5_mma %arg56, %arg78, %arg57, %true_11, %true_11, %arg80[%true_11], %arg58[%true_11] : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %145 = arith.addi %arg70, %c1_i32_12 : i32
        %146 = arith.xori %arg71, %c1_i32_12 : i32
        %147 = arith.cmpi eq, %145, %c2_i32_13 : i32
        %148 = arith.select %147, %c0_i32_10, %145 : i32
        %149 = arith.select %147, %146, %arg71 : i32
        %150 = ttg.memdesc_subview %arg28[%148] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %151 = ttg.memdesc_subview %arg29[%148] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %152 = ttg.memdesc_subview %arg30[%148, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %153 = arith.xori %arg76, %c1_i32_12 : i32
        ttng.wait_barrier %151, %149, %143 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %arg51, %153, %143 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.fence_async_shared {bCluster = false}
        ttng.tc_gen5_mma %arg52, %152, %arg53, %true_11, %143, %150[%true_11], %arg54[%true_11] : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %154 = ttg.memdesc_subview %arg41[%arg74, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %155 = ttg.memdesc_subview %arg42[%arg74] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.fence_async_shared {bCluster = false}
        ttng.tc_gen5_mma %arg43, %arg82, %154, %false, %141, %arg84[%true_11], %155[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %156 = arith.addi %arg74, %c1_i32_12 : i32
        %157 = arith.xori %arg75, %c1_i32_12 : i32
        %158 = arith.cmpi eq, %156, %c2_i32_13 : i32
        %159 = arith.select %158, %c0_i32_10, %156 : i32
        %160 = arith.select %158, %157, %arg75 : i32
        %161 = ttg.memdesc_subview %arg44[%159] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %161, %160, %141 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %162 = arith.addi %arg68, %c1_i32_12 : i32
        %163 = arith.xori %arg69, %c1_i32_12 : i32
        %164 = arith.cmpi eq, %162, %c2_i32_13 : i32
        %165 = arith.select %164, %c0_i32_10, %162 : i32
        %166 = arith.select %164, %163, %arg69 : i32
        %167 = ttg.memdesc_subview %arg25[%165] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %168 = ttg.memdesc_subview %arg26[%165] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %169 = ttg.memdesc_subview %arg27[%165, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %170 = ttg.memdesc_trans %169 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128> -> !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>
        ttng.wait_barrier %168, %166, %139 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %171 = ttg.memdesc_subview %arg31[%arg72, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %172 = ttg.memdesc_subview %arg32[%arg72] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.fence_async_shared {bCluster = false}
        ttng.tc_gen5_mma %arg33, %170, %171, %false, %139, %167[%true_11], %172[%true_11] : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %173 = arith.addi %arg72, %c1_i32_12 : i32
        %174 = arith.xori %arg73, %c1_i32_12 : i32
        %175 = arith.cmpi eq, %173, %c2_i32_13 : i32
        %176 = arith.select %175, %c0_i32_10, %173 : i32
        %177 = arith.select %175, %174, %arg73 : i32
        %178 = ttg.memdesc_subview %arg34[%176] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %178, %177, %139 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        scf.yield %165, %166, %148, %149, %176, %177, %159, %160, %153, %144, %arg79, %152, %arg81, %150, %arg83, %170, %arg85, %167 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared3, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition1(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg32: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg42: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg44: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg49: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg53: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg56: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg57: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg58: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg59: i32, %arg60: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg61: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg66: f32) num_warps(2) {
      %c256_i32_9 = arith.constant 256 : i32
      %c0_i32_10 = arith.constant 0 : i32
      %true_11 = arith.constant true
      %c1_i32_12 = arith.constant 1 : i32
      %c2_i32_13 = arith.constant 2 : i32
      %c128_i32_14 = arith.constant 128 : i32
      %105 = arith.cmpi sgt, %arg59, %c0_i32_10 : i32
      %106 = ttg.memdesc_subview %arg25[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %106, %c0_i32_10, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %107 = ttg.memdesc_subview %arg26[%c0_i32_10] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %107, 32768, %105 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %108 = ttg.memdesc_subview %arg27[%c0_i32_10, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %109 = ttng.tensor_desc_to_tma_ptr %arg60 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %109[%c0_i32_10, %c0_i32_10] %108, %107, %105 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %110 = arith.cmpi sgt, %arg59, %c128_i32_14 : i32
      %111 = ttg.memdesc_subview %arg25[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.wait_barrier %111, %c0_i32_10, %110 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %112 = ttg.memdesc_subview %arg26[%c1_i32_12] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      ttng.barrier_expect %112, 32768, %110 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
      %113 = ttg.memdesc_subview %arg27[%c1_i32_12, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      ttng.async_tma_copy_global_to_local %109[%c128_i32_14, %c0_i32_10] %113, %112, %110 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
      %114:4 = scf.for %arg67 = %c0_i32_10 to %arg59 step %c128_i32_14 iter_args(%arg68 = %c1_i32_12, %arg69 = %c0_i32_10, %arg70 = %c0_i32_10, %arg71 = %c0_i32_10) -> (i32, i32, i32, i32)  : i32 {
        %115 = arith.subi %arg59, %c256_i32_9 : i32
        %116 = arith.cmpi slt, %arg67, %115 : i32
        %117 = ttg.memdesc_subview %arg28[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %117, %arg71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %118 = ttg.memdesc_subview %arg29[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %118, 32768, %true_11 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %119 = ttg.memdesc_subview %arg30[%arg70, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %120 = ttng.tensor_desc_to_tma_ptr %arg61 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
        ttng.async_tma_copy_global_to_local %120[%arg67, %c0_i32_10] %119, %118, %true_11 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %121 = arith.addi %arg70, %c1_i32_12 : i32
        %122 = arith.xori %arg71, %c1_i32_12 : i32
        %123 = arith.cmpi eq, %121, %c2_i32_13 : i32
        %124 = arith.select %123, %c0_i32_10, %121 : i32
        %125 = arith.select %123, %122, %arg71 : i32
        %126 = arith.addi %arg68, %c1_i32_12 : i32
        %127 = arith.xori %arg69, %c1_i32_12 : i32
        %128 = arith.cmpi eq, %126, %c2_i32_13 : i32
        %129 = arith.select %128, %c0_i32_10, %126 : i32
        %130 = arith.select %128, %127, %arg69 : i32
        %131 = ttg.memdesc_subview %arg25[%129] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %131, %130, %116 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %132 = ttg.memdesc_subview %arg26[%129] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.barrier_expect %132, 32768, %116 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %133 = ttg.memdesc_subview %arg27[%129, %c0_i32_10, %c0_i32_10] : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        %134 = arith.addi %arg67, %c256_i32_9 : i32
        ttng.async_tma_copy_global_to_local %109[%134, %c0_i32_10] %133, %132, %116 : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 2x128x128>
        scf.yield %129, %130, %124, %125 : i32, i32, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.warp_return
    }
    partition2(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg32: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg42: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg44: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg49: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg53: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg56: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg57: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg58: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg59: i32, %arg60: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg61: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg66: f32) num_warps(4) {
      %c0_i32_9 = arith.constant 0 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c2_i32_11 = arith.constant 2 : i32
      %c3_i32_12 = arith.constant 3 : i32
      %c128_i32_13 = arith.constant 128 : i32
      %c-1_i32_14 = arith.constant -1 : i32
      %cst_15 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_16 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %105 = tt.splat %arg66 : f32 -> tensor<128x128xf32, #blocked>
      %106 = tt.splat %arg66 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %107:8 = scf.for %arg67 = %c0_i32_9 to %arg59 step %c128_i32_13 iter_args(%arg68 = %cst_16, %arg69 = %cst_15, %arg70 = %c0_i32_9, %arg71 = %c0_i32_9, %arg72 = %c-1_i32_14, %arg73 = %c0_i32_9, %arg74 = %c0_i32_9, %arg75 = %c0_i32_9) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32, i32, i32)  : i32 {
        %108 = ttg.memdesc_subview %arg31[%arg70, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %109 = ttg.memdesc_subview %arg32[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %109, %arg71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %result_17 = ttng.tmem_load %108 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %110 = ttg.memdesc_subview %arg34[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.arrive_barrier %110, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %111 = arith.addi %arg70, %c1_i32_10 : i32
        %112 = arith.xori %arg71, %c1_i32_10 : i32
        %113 = arith.cmpi eq, %111, %c2_i32_11 : i32
        %114 = arith.select %113, %c0_i32_9, %111 : i32
        %115 = arith.select %113, %112, %arg71 : i32
        %116 = "tt.reduce"(%result_17) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %145 = arith.maxnumf %arg76, %arg77 : f32
          tt.reduce.return %145 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %117 = arith.mulf %116, %106 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %118 = arith.maxnumf %arg69, %117 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %119 = arith.addi %arg74, %c1_i32_10 : i32
        %120 = arith.xori %arg75, %c1_i32_10 : i32
        %121 = arith.cmpi eq, %119, %c3_i32_12 : i32
        %122 = arith.select %121, %120, %arg75 : i32
        %123 = arith.select %121, %c1_i32_10, %119 : i32
        %124 = ttg.memdesc_subview %arg35[%123, %c0_i32_9] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %125 = ttg.memdesc_subview %arg36[%123] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %126 = ttg.memdesc_subview %arg37[%123] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %126, %122 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %118, %124 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %125, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %127 = arith.mulf %result_17, %105 : tensor<128x128xf32, #blocked>
        %128 = tt.expand_dims %118 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %129 = tt.broadcast %128 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %130 = arith.subf %127, %129 : tensor<128x128xf32, #blocked>
        %131 = math.exp2 %130 : tensor<128x128xf32, #blocked>
        %132 = arith.addi %arg72, %c1_i32_10 : i32
        %133 = arith.xori %arg73, %c1_i32_10 : i32
        %134 = arith.cmpi eq, %132, %c1_i32_10 : i32
        %135 = arith.select %134, %133, %arg73 : i32
        %136 = arith.select %134, %c0_i32_9, %132 : i32
        %137 = ttg.memdesc_subview %arg38[%136, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        %138 = ttg.memdesc_subview %arg39[%136] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %139 = ttg.memdesc_subview %arg40[%136] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %139, %135 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttg.local_store %131, %137 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        ttng.arrive_barrier %138, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %140 = arith.subf %arg69, %118 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %141 = math.exp2 %140 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %142 = "tt.reduce"(%131) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %145 = arith.addf %arg76, %arg77 : f32
          tt.reduce.return %145 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %143 = arith.mulf %arg68, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %144 = arith.addf %143, %142 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %144, %118, %114, %115, %136, %135, %123, %122 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %107#0, %arg62 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %107#1, %arg63 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    }
    partition3(%arg25: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg26: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg27: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg28: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg29: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg30: !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, %arg31: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg32: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg34: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg37: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg38: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg39: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg40: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg41: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, %arg42: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg43: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg44: !ttg.memdesc<2xi64, #shared1, #smem, mutable>, %arg45: !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, %arg46: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg47: !ttg.memdesc<3xi64, #shared1, #smem, mutable>, %arg48: !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, %arg49: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg50: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg51: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg52: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg53: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg54: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg55: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg56: !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, %arg57: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, %arg58: !ttg.memdesc<1xi64, #shared1, #smem, mutable>, %arg59: i32, %arg60: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg61: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg62: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg63: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg64: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg65: !ttg.memdesc<128xf32, #shared1, #smem, mutable>, %arg66: f32) num_warps(4) {
      %c0_i32_9 = arith.constant 0 : i32
      %c1_i32_10 = arith.constant 1 : i32
      %c2_i32_11 = arith.constant 2 : i32
      %c3_i32_12 = arith.constant 3 : i32
      %c128_i32_13 = arith.constant 128 : i32
      %c-1_i32_14 = arith.constant -1 : i32
      %cst_15 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %cst_16 = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %105 = tt.splat %arg66 : f32 -> tensor<128x128xf32, #blocked>
      %106 = tt.splat %arg66 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %107:8 = scf.for %arg67 = %c0_i32_9 to %arg59 step %c128_i32_13 iter_args(%arg68 = %cst_16, %arg69 = %cst_15, %arg70 = %c0_i32_9, %arg71 = %c0_i32_9, %arg72 = %c-1_i32_14, %arg73 = %c0_i32_9, %arg74 = %c0_i32_9, %arg75 = %c0_i32_9) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32, i32, i32)  : i32 {
        %108 = ttg.memdesc_subview %arg41[%arg70, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %109 = ttg.memdesc_subview %arg42[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.wait_barrier %109, %arg71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %result_17 = ttng.tmem_load %108 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        %110 = ttg.memdesc_subview %arg44[%arg70] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        ttng.arrive_barrier %110, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
        %111 = arith.addi %arg70, %c1_i32_10 : i32
        %112 = arith.xori %arg71, %c1_i32_10 : i32
        %113 = arith.cmpi eq, %111, %c2_i32_11 : i32
        %114 = arith.select %113, %c0_i32_9, %111 : i32
        %115 = arith.select %113, %112, %arg71 : i32
        %116 = "tt.reduce"(%result_17) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %145 = arith.maxnumf %arg76, %arg77 : f32
          tt.reduce.return %145 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %117 = arith.mulf %116, %106 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %118 = arith.maxnumf %arg69, %117 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %119 = arith.addi %arg74, %c1_i32_10 : i32
        %120 = arith.xori %arg75, %c1_i32_10 : i32
        %121 = arith.cmpi eq, %119, %c3_i32_12 : i32
        %122 = arith.select %121, %120, %arg75 : i32
        %123 = arith.select %121, %c1_i32_10, %119 : i32
        %124 = ttg.memdesc_subview %arg45[%123, %c0_i32_9] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        %125 = ttg.memdesc_subview %arg46[%123] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %126 = ttg.memdesc_subview %arg47[%123] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttng.wait_barrier %126, %122 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        ttg.local_store %118, %124 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
        ttng.arrive_barrier %125, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
        %127 = arith.mulf %result_17, %105 : tensor<128x128xf32, #blocked>
        %128 = tt.expand_dims %118 {axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
        %129 = tt.broadcast %128 : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
        %130 = arith.subf %127, %129 : tensor<128x128xf32, #blocked>
        %131 = math.exp2 %130 : tensor<128x128xf32, #blocked>
        %132 = arith.addi %arg72, %c1_i32_10 : i32
        %133 = arith.xori %arg73, %c1_i32_10 : i32
        %134 = arith.cmpi eq, %132, %c1_i32_10 : i32
        %135 = arith.select %134, %133, %arg73 : i32
        %136 = arith.select %134, %c0_i32_9, %132 : i32
        %137 = ttg.memdesc_subview %arg48[%136, %c0_i32_9, %c0_i32_9] : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        %138 = ttg.memdesc_subview %arg49[%136] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %139 = ttg.memdesc_subview %arg50[%136] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttng.wait_barrier %139, %135 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        ttg.local_store %131, %137 : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #shared2, #smem, mutable, 1x128x128>
        ttng.arrive_barrier %138, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
        %140 = arith.subf %arg69, %118 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %141 = math.exp2 %140 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %142 = "tt.reduce"(%131) <{axis = 1 : i32}> ({
        ^bb0(%arg76: f32, %arg77: f32):
          %145 = arith.addf %arg76, %arg77 : f32
          tt.reduce.return %145 : f32
        }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %143 = arith.mulf %arg68, %141 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        %144 = arith.addf %143, %142 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
        scf.yield %144, %118, %114, %115, %136, %135, %123, %122 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, i32, i32, i32, i32, i32, i32
      } {tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.warp_specialize}
      ttg.local_store %107#0, %arg64 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.local_store %107#1, %arg65 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<2xi64, #shared1, #smem, mutable>, !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<3xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, i32, !tt.tensordesc<tensor<128x128xf16, #shared>>, !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, !ttg.memdesc<128xf32, #shared1, #smem, mutable>, f32) -> (i32, i32, i32, i32, i32, i32)
    %80 = ttg.local_load %78 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %78 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %81 = ttg.local_load %77 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %77 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %82 = ttg.local_load %76 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %76 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    %83 = ttg.local_load %75 : !ttg.memdesc<128xf32, #shared1, #smem, mutable> -> tensor<128xf32, #blocked1>
    ttg.local_dealloc %75 : !ttg.memdesc<128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %65 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %66 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %67 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %60 : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>
    ttg.local_dealloc %61 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %62 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %50 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %51 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %52 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %45 : !ttg.memdesc<1x128x128xf32, #shared2, #smem, mutable>
    ttng.inval_barrier %48 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %46 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %56 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %58 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %59 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %63 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %64 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %69 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %72 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %74 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %44, %79#5 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %40, %79#4 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %38 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %84 = ttg.memdesc_subview %34[%79#2] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.wait_barrier %84, %79#3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %35 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %36 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %34 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %31 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %85 = ttg.memdesc_subview %28[%79#0] : !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.wait_barrier %85, %79#1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %28 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %25 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %23 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %24 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %22 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %20 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %21 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %19 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %18 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %16 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %17 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %15 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %13 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttng.inval_barrier %14 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 2>
    ttg.local_dealloc %12 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<2x128x128xf16, #shared, #smem, mutable>
    %result_7 = ttng.tmem_load %9 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_8 = ttng.tmem_load %10 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %86 = arith.truncf %result_7 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %87 = ttg.local_alloc %86 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %88 = ttng.tensor_desc_to_tma_ptr %arg15 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %88[%1, %c0_i32] %87 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %89 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %90 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %91 = tt.splat %89 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %92 = tt.addptr %91, %90 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %93 = arith.truncf %82 : tensor<128xf32, #blocked1> to tensor<128xf16, #blocked1>
    tt.store %92, %93 : tensor<128x!tt.ptr<f16>, #blocked1>
    %94 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %95 = tt.splat %94 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %96 = tt.addptr %95, %90 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %97 = arith.truncf %83 : tensor<128xf32, #blocked1> to tensor<128xf16, #blocked1>
    tt.store %96, %97 : tensor<128x!tt.ptr<f16>, #blocked1>
    %98 = arith.truncf %result_8 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %99 = ttg.local_alloc %98 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    %100 = ttng.tensor_desc_to_tma_ptr %arg15 : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
    ttng.async_tma_copy_local_to_global %100[%5, %c0_i32] %99 : !tt.ptr<i8>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    %101 = tt.addptr %92, %cst_1 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %102 = arith.truncf %80 : tensor<128xf32, #blocked1> to tensor<128xf16, #blocked1>
    tt.store %101, %102 : tensor<128x!tt.ptr<f16>, #blocked1>
    %103 = tt.addptr %96, %cst_1 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %104 = arith.truncf %81 : tensor<128xf32, #blocked1> to tensor<128xf16, #blocked1>
    tt.store %103, %104 : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

