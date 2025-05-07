#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = false>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel_data_part(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c-1_i32 = arith.constant -1 : i32
    %0 = ub.poison : !ttg.async.token
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_2 = arith.constant dense<128> : tensor<128xi32, #blocked1>
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %true = arith.constant true
    %false = arith.constant false
    %1 = tt.get_program_id x : i32
    %2 = arith.muli %1, %c256_i32 : i32
    %3 = tt.descriptor_load %arg0[%2, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
    %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %5 = arith.addi %2, %c128_i32 : i32
    %6 = tt.descriptor_load %arg0[%5, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
    %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    %8 = tt.splat %arg24 : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %arg24 : f32 -> tensor<128x128xf32, #blocked>

    %TMEM_ALLOC_QK0, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %TMEM_ALLOC_O0, %token_6 = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    %TMEM_ALLOC_QK1, %token_4 = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %TMEM_ALLOW_QK1_0 = ttg.memdesc_subview %TMEM_ALLOC_QK1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %13 = ttng.tmem_store %cst_1, %TMEM_ALLOW_QK1_0[%token_4], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>

    %TMEM_ALLOC_O1, %token_8 = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    %TMEM_ALLOW_O1_0 = ttg.memdesc_subview %TMEM_ALLOC_O1[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %12 = ttng.tmem_store %cst_1, %TMEM_ALLOW_O1_0[%token_8], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>

    %TMEM_ALLOC_P0 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    %TMEM_ALLOC_P1 = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>

    %14 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %15 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %16 = ttg.memdesc_subview %15[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %16, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %17 = ttg.memdesc_subview %15[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %17, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %18 = ttg.memdesc_subview %15[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %18, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %19 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %20 = ttg.memdesc_subview %19[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %20, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %21 = ttg.memdesc_subview %19[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %22 = ttg.memdesc_subview %19[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %22, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %16, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %17, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %18, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %23 = ttg.local_alloc : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %24 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %25 = ttg.memdesc_subview %24[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %26 = ttg.memdesc_subview %24[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %26, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %27 = ttg.memdesc_subview %24[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %27, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %28 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %29 = ttg.memdesc_subview %28[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %30 = ttg.memdesc_subview %28[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %31 = ttg.memdesc_subview %28[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %31, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %25, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %26, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %27, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %32 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %33 = ttg.memdesc_subview %32[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %34 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %35 = ttg.memdesc_subview %34[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %35, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %36 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %37 = ttg.memdesc_subview %36[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %38 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %39 = ttg.memdesc_subview %38[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %39, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %40 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %41 = ttg.memdesc_subview %40[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %43 = ttg.memdesc_subview %42[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %45 = ttg.memdesc_subview %44[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %45, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %46 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %47 = ttg.memdesc_subview %46[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %47, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %49 = ttg.memdesc_subview %48[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %50 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %51 = ttg.memdesc_subview %50[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %52 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %53 = ttg.memdesc_subview %52[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %53, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %54 = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %55 = ttg.memdesc_subview %54[%c0_i32] : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.arrive_barrier %52, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %56 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %57 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %58 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %59 = ttg.memdesc_subview %56[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_0, %59 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %60 = ttg.memdesc_subview %57[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %61 = ttg.memdesc_subview %58[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %60, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %61, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %60, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %62 = ttg.memdesc_subview %57[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %63 = ttg.memdesc_subview %58[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %62, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %63, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %63, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %64 = ttg.memdesc_subview %57[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %65 = ttg.memdesc_subview %58[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %64, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %65, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %65, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %66 = ttg.local_alloc : () -> !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    %67 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %68 = ttg.local_alloc : () -> !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    %69 = ttg.memdesc_subview %66[%c0_i32, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    ttg.local_store %cst_0, %69 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
    %70 = ttg.memdesc_subview %67[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %71 = ttg.memdesc_subview %68[%c0_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %70, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %71, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %70, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %72 = ttg.memdesc_subview %67[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %73 = ttg.memdesc_subview %68[%c1_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %72, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %73, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %73, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %74 = ttg.memdesc_subview %67[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    %75 = ttg.memdesc_subview %68[%c2_i32] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %74, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.init_barrier %75, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.arrive_barrier %75, 2 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>


    %76:36 = scf.for %arg25 = %c0_i32 to %arg23 step %c128_i32 iter_args(%arg26 = %cst, %arg27 = %cst_0, %arg28 = %cst, %arg29 = %cst_0, %arg30 = %token, %arg31 = %13, %arg32 = %token_6, %arg33 = %12, %arg34 = %c0_i32, %arg35 = %c0_i32, %arg36 = %c0_i32, %arg37 = %c0_i32, %arg38 = %c0_i32, %arg39 = %c0_i32, %arg40 = %c0_i32, %arg41 = %c0_i32, %arg42 = %c0_i32, %arg43 = %c0_i32, %arg44 = %c0_i32, %arg45 = %c0_i32, %arg46 = %c0_i32, %arg47 = %c0_i32, %arg48 = %c0_i32, %arg49 = %c0_i32, %arg50 = %c0_i32, %arg51 = %c0_i32, %arg52 = %c-1_i32, %arg53 = %c0_i32, %arg54 = %c0_i32, %arg55 = %c0_i32, %arg56 = %c0_i32, %arg57 = %c0_i32, %arg58 = %c-1_i32, %arg59 = %c0_i32, %arg60 = %c0_i32, %arg61 = %c0_i32) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
      %98 = ttg.memdesc_subview %15[%arg34] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %98, %arg35 {ttg.assigned_stage = 0 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %99 = ttg.memdesc_subview %19[%arg34] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %99, 32768 {ttg.assigned_stage = 0 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %100 = ttg.memdesc_subview %14[%arg34, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>
      %101 = ttng.tensor_desc_to_tma_ptr %arg5 {ttg.assigned_stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %101[%arg25, %c0_i32] %100, %99, %true {ttg.assigned_stage = 0 : i32, ttg.partition = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>
      %102 = ttg.memdesc_trans %100 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128> -> !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>
      %103 = ttg.memdesc_subview %24[%arg36] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %103, %arg37 {ttg.assigned_stage = 2 : i32, ttg.partition = 2 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %104 = ttg.memdesc_subview %28[%arg36] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.barrier_expect %104, 32768 {ttg.assigned_stage = 2 : i32, ttg.partition = 2 : i32}, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %105 = ttg.memdesc_subview %23[%arg36, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>
      %106 = ttng.tensor_desc_to_tma_ptr %arg10 {ttg.assigned_stage = 2 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x128xf16, #shared>> to !tt.ptr<i8>
      ttng.async_tma_copy_global_to_local %106[%arg25, %c0_i32] %105, %104, %true {ttg.assigned_stage = 2 : i32, ttg.partition = 2 : i32} : !tt.ptr<i8>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>
      ttng.wait_barrier %99, %arg35 {ttg.assigned_stage = 0 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %107 = ttg.memdesc_subview %TMEM_ALLOC_QK0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %108 = ttng.tc_gen5_mma %4, %102, %107[], %false, %true, %98[%true], %33[%true] {ttg.assigned_stage = 0 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %33, %arg39 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_15, %token_16 = ttng.tmem_load %107[] {ttg.partition = 3 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      ttng.arrive_barrier %35, 1 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %109 = arith.xori %arg39, %c1_i32 : i32
      %110 = arith.select %true, %109, %arg39 : i32
      %111 = arith.select %true, %110, %arg39 : i32
      ttng.wait_barrier %35, %111, %true {ttg.assigned_stage = 0 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %112 = "tt.reduce"(%result_15) <{axis = 1 : i32}> ({
      ^bb0(%arg62: f32, %arg63: f32):
        %233 = arith.maxnumf %arg62, %arg63 : f32
        tt.reduce.return %233 : f32
      }) {ttg.partition = 3 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %113 = arith.mulf %112, %8 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %114 = arith.maxnumf %arg27, %113 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %115 = arith.addi %arg54, %c1_i32 : i32
      %116 = arith.xori %arg55, %c1_i32 : i32
      %117 = arith.cmpi eq, %115, %c3_i32 : i32
      %118 = arith.select %117, %116, %arg55 : i32
      %119 = arith.select %117, %c1_i32, %115 : i32
      %120 = ttg.memdesc_subview %56[%119, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %121 = ttg.memdesc_subview %57[%119] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %122 = ttg.memdesc_subview %58[%119] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %122, %118 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttg.local_store %114, %120 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      ttng.arrive_barrier %121, 1 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %123 = arith.mulf %result_15, %9 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked>
      %124 = tt.expand_dims %114 {axis = 1 : i32, ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %125 = tt.broadcast %124 {ttg.partition = 3 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %126 = arith.subf %123, %125 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked>
      %127 = math.exp2 %126 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked>
      %128 = arith.addi %arg50, %c1_i32 : i32
      %129 = arith.xori %arg51, %c1_i32 : i32
      %130 = arith.cmpi eq, %128, %c3_i32 : i32
      %131 = arith.select %130, %129, %arg51 : i32
      %132 = arith.select %130, %c1_i32, %128 : i32
      %133 = ttg.memdesc_subview %56[%132, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %134 = ttg.memdesc_subview %57[%132] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %135 = ttg.memdesc_subview %58[%132] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %134, %131 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %136 = ttg.local_load %133 {ttg.partition = 0 : i32} : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      ttng.arrive_barrier %135, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %137 = arith.addi %arg52, %c1_i32 : i32
      %138 = arith.xori %arg53, %c1_i32 : i32
      %139 = arith.cmpi eq, %137, %c3_i32 : i32
      %140 = arith.select %139, %138, %arg53 : i32
      %141 = arith.select %139, %c1_i32, %137 : i32
      %142 = ttg.memdesc_subview %56[%141, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %143 = ttg.memdesc_subview %57[%141] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %144 = ttg.memdesc_subview %58[%141] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %143, %140 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %145 = ttg.local_load %142 {ttg.partition = 0 : i32} : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      ttng.arrive_barrier %144, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %146 = arith.subf %145, %136 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %147 = arith.subf %arg27, %114 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %148 = math.exp2 %146 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %149 = math.exp2 %147 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %150 = "tt.reduce"(%127) <{axis = 1 : i32}> ({
      ^bb0(%arg62: f32, %arg63: f32):
        %233 = arith.addf %arg62, %arg63 : f32
        tt.reduce.return %233 : f32
      }) {ttg.partition = 3 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %151 = tt.expand_dims %148 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %152 = tt.expand_dims %149 {axis = 1 : i32, ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %153 = tt.broadcast %151 {ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      ttng.wait_barrier %39, %arg41 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_17, %token_18 = ttng.tmem_load %TMEM_ALLOW_QK1_0[] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %154 = arith.xori %arg41, %c1_i32 : i32
      %155 = arith.select %true, %154, %arg41 : i32
      %156 = arith.select %true, %155, %arg41 : i32
      %157 = arith.mulf %result_17, %153 {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked>
      %158 = arith.truncf %127 {ttg.partition = 3 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttng.wait_barrier %40, %arg43 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tmem_store %158, %TMEM_ALLOC_P0, %true {ttg.partition = 3 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
      ttng.arrive_barrier %42, 1 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %159 = ttng.tmem_store %157, %TMEM_ALLOW_QK1_0[], %true {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      ttng.arrive_barrier %37, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %104, %arg37 {ttg.assigned_stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %37, %156, %true {ttg.assigned_stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %42, %arg43 {ttg.assigned_stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %160 = ttng.tc_gen5_mma %TMEM_ALLOC_P0, %105, %TMEM_ALLOW_QK1_0[], %true, %true, %103[%true], %39[%true], %40[%true] {ttg.assigned_stage = 2 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %161 = arith.mulf %arg26, %149 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %162 = arith.addf %161, %150 {ttg.partition = 3 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %163 = ttg.memdesc_subview %TMEM_ALLOC_O0[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %164 = ttng.tc_gen5_mma %7, %102, %163[], %false, %true, %98[%true], %45[%true] {ttg.assigned_stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %45, %arg45 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_19, %token_20 = ttng.tmem_load %163[] {ttg.partition = 4 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      ttng.arrive_barrier %47, 1 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %165 = arith.xori %arg45, %c1_i32 : i32
      %166 = arith.select %true, %165, %arg45 : i32
      %167 = arith.select %true, %166, %arg45 : i32
      ttng.wait_barrier %47, %167, %true {ttg.assigned_stage = 4 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %168 = "tt.reduce"(%result_19) <{axis = 1 : i32}> ({
      ^bb0(%arg62: f32, %arg63: f32):
        %233 = arith.maxnumf %arg62, %arg63 : f32
        tt.reduce.return %233 : f32
      }) {ttg.partition = 4 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %169 = arith.mulf %168, %8 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %170 = arith.maxnumf %arg29, %169 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %171 = arith.addi %arg60, %c1_i32 : i32
      %172 = arith.xori %arg61, %c1_i32 : i32
      %173 = arith.cmpi eq, %171, %c3_i32 : i32
      %174 = arith.select %173, %172, %arg61 : i32
      %175 = arith.select %173, %c1_i32, %171 : i32
      %176 = ttg.memdesc_subview %66[%175, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %177 = ttg.memdesc_subview %67[%175] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %178 = ttg.memdesc_subview %68[%175] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %178, %174 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttg.local_store %170, %176 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      ttng.arrive_barrier %177, 1 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %179 = arith.mulf %result_19, %9 {ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked>
      %180 = tt.expand_dims %170 {axis = 1 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %181 = tt.broadcast %180 {ttg.partition = 4 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %182 = arith.subf %179, %181 {ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked>
      %183 = math.exp2 %182 {ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked>
      %184 = arith.addi %arg56, %c1_i32 : i32
      %185 = arith.xori %arg57, %c1_i32 : i32
      %186 = arith.cmpi eq, %184, %c3_i32 : i32
      %187 = arith.select %186, %185, %arg57 : i32
      %188 = arith.select %186, %c1_i32, %184 : i32
      %189 = ttg.memdesc_subview %66[%188, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %190 = ttg.memdesc_subview %67[%188] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %191 = ttg.memdesc_subview %68[%188] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %190, %187 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %192 = ttg.local_load %189 {ttg.partition = 0 : i32} : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      ttng.arrive_barrier %191, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %193 = arith.addi %arg58, %c1_i32 : i32
      %194 = arith.xori %arg59, %c1_i32 : i32
      %195 = arith.cmpi eq, %193, %c3_i32 : i32
      %196 = arith.select %195, %194, %arg59 : i32
      %197 = arith.select %195, %c1_i32, %193 : i32
      %198 = ttg.memdesc_subview %66[%197, %c0_i32] : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable> -> !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128>
      %199 = ttg.memdesc_subview %67[%197] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %200 = ttg.memdesc_subview %68[%197] : !ttg.memdesc<3xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      ttng.wait_barrier %199, %196 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %201 = ttg.local_load %198 {ttg.partition = 0 : i32} : !ttg.memdesc<128xf32, #shared1, #smem, mutable, 3x128> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      ttng.arrive_barrier %200, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
      %202 = arith.subf %201, %192 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %203 = arith.subf %arg29, %170 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %204 = math.exp2 %202 {ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %205 = math.exp2 %203 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %206 = "tt.reduce"(%183) <{axis = 1 : i32}> ({
      ^bb0(%arg62: f32, %arg63: f32):
        %233 = arith.addf %arg62, %arg63 : f32
        tt.reduce.return %233 : f32
      }) {ttg.partition = 4 : i32} : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %207 = tt.expand_dims %204 {axis = 1 : i32, ttg.partition = 0 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %208 = tt.expand_dims %205 {axis = 1 : i32, ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xf32, #blocked>
      %209 = tt.broadcast %207 {ttg.partition = 0 : i32} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      ttng.wait_barrier %51, %arg47 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_21, %token_22 = ttng.tmem_load %TMEM_ALLOW_O1_0[] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
      %210 = arith.xori %arg47, %c1_i32 : i32
      %211 = arith.select %true, %210, %arg47 : i32
      %212 = arith.select %true, %211, %arg47 : i32
      %213 = arith.mulf %result_21, %209 {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked>
      %214 = arith.truncf %183 {ttg.partition = 4 : i32} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
      ttng.wait_barrier %52, %arg49 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tmem_store %214, %TMEM_ALLOC_P1, %true {ttg.partition = 4 : i32} : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
      ttng.arrive_barrier %54, 1 {ttg.partition = 4 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %215 = ttng.tmem_store %213, %TMEM_ALLOW_O1_0[], %true {ttg.partition = 0 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      ttng.arrive_barrier %49, 1 {ttg.partition = 0 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %49, %212, %true {ttg.assigned_stage = 6 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %54, %arg49 {ttg.assigned_stage = 6 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %216 = ttng.tc_gen5_mma %TMEM_ALLOC_P1, %105, %TMEM_ALLOW_O1_0[], %true, %true, %103[%true], %51[%true], %52[%true] {ttg.assigned_stage = 6 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable, 3x128x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %217 = arith.mulf %arg28, %205 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %218 = arith.addf %217, %206 {ttg.partition = 4 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %219 = arith.addi %arg34, %c1_i32 : i32
      %220 = arith.xori %arg35, %c1_i32 : i32
      %221 = arith.cmpi eq, %219, %c3_i32 : i32
      %222 = arith.select %221, %c0_i32, %219 : i32
      %223 = arith.select %221, %220, %arg35 : i32
      %224 = arith.addi %arg36, %c1_i32 : i32
      %225 = arith.xori %arg37, %c1_i32 : i32
      %226 = arith.cmpi eq, %224, %c3_i32 : i32
      %227 = arith.select %226, %c0_i32, %224 : i32
      %228 = arith.select %226, %225, %arg37 : i32
      %229 = arith.xori %arg43, %c1_i32 : i32
      %230 = arith.select %true, %229, %arg43 : i32
      %231 = arith.xori %arg49, %c1_i32 : i32
      %232 = arith.select %true, %231, %arg49 : i32
      scf.yield %162, %114, %218, %170, %0, %0, %0, %0, %222, %223, %227, %228, %c0_i32, %111, %c0_i32, %156, %c0_i32, %230, %c0_i32, %167, %c0_i32, %212, %c0_i32, %232, %132, %131, %141, %140, %119, %118, %188, %187, %197, %196, %175, %174 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token, !ttg.async.token, !ttg.async.token, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
    } {tt.disallow_acc_multi_buffer, tt.divisibility_arg1 = dense<128> : tensor<1xi32>, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition.stages = [1 : i32, 1 : i32, 0 : i32, 0 : i32, 0 : i32]}
    ttg.local_dealloc %66 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttg.local_dealloc %67 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %68 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %56 : !ttg.memdesc<3x128xf32, #shared1, #smem, mutable>
    ttng.inval_barrier %60 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %61 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %62 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %63 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %64 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %65 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %57 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %58 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %70 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %71 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %72 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %73 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %74 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %75 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.wait_barrier %51, %76#21 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %53 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %52 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %50 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %48 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %47, %76#19 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %46 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %45 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %39, %76#15 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %42 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %39 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %38 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %36 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.wait_barrier %35, %76#13 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %35 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %31 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %28 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %25 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %24 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %23 : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    ttng.inval_barrier %20 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %21 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %22 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %19 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %16 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %17 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttng.inval_barrier %18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 3>
    ttg.local_dealloc %15 : !ttg.memdesc<3xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %14 : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %result_11, %token_12 = ttng.tmem_load %TMEM_ALLOW_QK1_0[%76#5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %result_13, %token_14 = ttng.tmem_load %TMEM_ALLOW_O1_0[%76#7] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    %77 = arith.truncf %result_11 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %78 = ttg.convert_layout %77 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %arg15[%2, %c0_i32], %78 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    %79 = tt.addptr %arg20, %2 : !tt.ptr<f16>, i32
    %80 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked1>
    %81 = tt.splat %79 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %82 = tt.addptr %81, %80 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %83 = arith.truncf %76#0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %84 = ttg.convert_layout %83 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %82, %84 : tensor<128x!tt.ptr<f16>, #blocked1>
    %85 = tt.addptr %arg21, %2 : !tt.ptr<f16>, i32
    %86 = tt.splat %85 : !tt.ptr<f16> -> tensor<128x!tt.ptr<f16>, #blocked1>
    %87 = tt.addptr %86, %80 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %88 = arith.truncf %76#1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %89 = ttg.convert_layout %88 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %87, %89 : tensor<128x!tt.ptr<f16>, #blocked1>
    %90 = arith.truncf %result_13 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %91 = ttg.convert_layout %90 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked2>
    tt.descriptor_store %arg15[%5, %c0_i32], %91 : !tt.tensordesc<tensor<128x128xf16, #shared>>, tensor<128x128xf16, #blocked2>
    %92 = tt.addptr %82, %cst_2 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %93 = arith.truncf %76#2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %94 = ttg.convert_layout %93 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %92, %94 : tensor<128x!tt.ptr<f16>, #blocked1>
    %95 = tt.addptr %87, %cst_2 : tensor<128x!tt.ptr<f16>, #blocked1>, tensor<128xi32, #blocked1>
    %96 = arith.truncf %76#3 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> to tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %97 = ttg.convert_layout %96 : tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf16, #blocked1>
    tt.store %95, %97 : tensor<128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

