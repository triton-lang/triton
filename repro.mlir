#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {
    %cst = arith.constant dense<0> : tensor<64x1xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<1x64xi32, #blocked>
    %cst_1 = arith.constant dense<128> : tensor<128x64xi32, #blocked>
    %false = arith.constant false
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c1_i32 = arith.constant 1 : i32
    %cst_3 = arith.constant dense<128> : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cst_4 = arith.constant dense<128> : tensor<1x64xi32, #blocked>
    %cst_5 = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %cst_6 = arith.constant dense<64> : tensor<128x64xi32, #blocked>
    %cst_7 = arith.constant dense<64> : tensor<1x64xi32, #blocked>
    %cst_8 = arith.constant dense<64> : tensor<64x1xi32, #blocked>
    %true = arith.constant true
    %cst_9 = arith.constant dense<128> : tensor<128x1xi32, #blocked>
    %cst_10 = arith.constant dense<128> : tensor<1x128xi32, #blocked>
    %cst_11 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %0, %c4_i32 : i32
    %2 = arith.muli %1, %c4_i32 : i32
    %3 = arith.subi %c1_i32, %2 : i32
    %4 = arith.minsi %3, %c4_i32 : i32
    %5 = arith.remsi %0, %c4_i32 : i32
    %6 = arith.remsi %5, %4 : i32
    %7 = arith.divsi %5, %4 : i32
    %8 = arith.addi %2, %6 : i32
    %9 = arith.muli %8, %c128_i32 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = tt.splat %9 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %12 = arith.addi %11, %10 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.remsi %12, %cst_2 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %15 = tt.expand_dims %13 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %16 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked>
    %17 = arith.muli %15, %16 : tensor<128x1xi32, #blocked>
    %18 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %19 = tt.addptr %18, %17 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %20 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %21 = tt.broadcast %19 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %22 = tt.broadcast %20 : tensor<1x64xi32, #blocked> -> tensor<128x64xi32, #blocked>
    %23 = tt.addptr %21, %22 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %24 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %25 = arith.muli %7, %c128_i32 : i32
    %26 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %27 = tt.splat %25 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %28 = arith.addi %27, %26 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = arith.remsi %28, %cst_3 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %30 = tt.expand_dims %24 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %31 = tt.splat %arg4 : i32 -> tensor<64x1xi32, #blocked>
    %32 = arith.muli %30, %31 : tensor<64x1xi32, #blocked>
    %33 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %34 = tt.addptr %33, %32 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %35 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %36 = tt.broadcast %34 : tensor<64x1x!tt.ptr<f16>, #blocked> -> tensor<64x128x!tt.ptr<f16>, #blocked>
    %37 = tt.broadcast %35 : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %38 = tt.addptr %36, %37 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %39 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %40 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    %41 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %42 = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<64x128xf16, #shared, #smem, mutable>
    %43 = arith.cmpi slt, %20, %cst_4 : tensor<1x64xi32, #blocked>
    %44 = arith.cmpi slt, %30, %cst_5 : tensor<64x1xi32, #blocked>
    %45 = tt.broadcast %43 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %46 = ttg.async_copy_global_to_local %23, %39 mask %45 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #smem, mutable>
    %47 = tt.broadcast %44 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %48 = ttg.async_copy_global_to_local %38, %41 mask %47 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    cf.cond_br %true, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %49 = tt.addptr %23, %cst_6 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %50 = arith.muli %arg4, %c64_i32 : i32
    %51 = tt.splat %50 : i32 -> tensor<64x128xi32, #blocked>
    %52 = tt.addptr %38, %51 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %53 = arith.cmpi slt, %20, %cst_7 : tensor<1x64xi32, #blocked>
    %54 = arith.cmpi slt, %30, %cst_8 : tensor<64x1xi32, #blocked>
    %55 = tt.broadcast %53 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %56 = ttg.async_copy_global_to_local %49, %40 mask %55 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #smem, mutable>
    %57 = tt.broadcast %54 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %58 = ttg.async_copy_global_to_local %52, %42 mask %57 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    cf.br ^bb3
  ^bb2:  // pred: ^bb0
    cf.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %59 = ttg.async_commit_group
    %60 = ttg.async_wait  {num = 0 : i32}
    %61 = ttng.warp_group_dot %39, %41, %cst_11, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
    cf.cond_br %false, ^bb4, ^bb8
  ^bb4:  // pred: ^bb3
    %62 = tt.addptr %23, %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %63 = arith.muli %arg4, %c128_i32 : i32
    %64 = tt.splat %63 : i32 -> tensor<64x128xi32, #blocked>
    %65 = tt.addptr %38, %64 : tensor<64x128x!tt.ptr<f16>, #blocked>, tensor<64x128xi32, #blocked>
    %66 = arith.cmpi slt, %20, %cst_0 : tensor<1x64xi32, #blocked>
    %67 = arith.cmpi slt, %30, %cst : tensor<64x1xi32, #blocked>
    cf.cond_br %true, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    %68 = tt.broadcast %66 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %69 = ttg.async_copy_global_to_local %62, %39 mask %68 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #smem, mutable>
    %70 = tt.broadcast %67 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %71 = ttg.async_copy_global_to_local %65, %41 mask %70 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    cf.br ^bb7
  ^bb6:  // pred: ^bb4
    %72 = tt.broadcast %66 : tensor<1x64xi1, #blocked> -> tensor<128x64xi1, #blocked>
    %73 = ttg.async_copy_global_to_local %62, %40 mask %72 : tensor<128x64x!tt.ptr<f16>, #blocked> -> <128x64xf16, #shared, #smem, mutable>
    %74 = tt.broadcast %67 : tensor<64x1xi1, #blocked> -> tensor<64x128xi1, #blocked>
    %75 = ttg.async_copy_global_to_local %65, %42 mask %74 : tensor<64x128x!tt.ptr<f16>, #blocked> -> <64x128xf16, #shared, #smem, mutable>
    cf.br ^bb7
  ^bb7:  // 2 preds: ^bb5, ^bb6
    %76 = ttg.async_commit_group
    cf.br ^bb9
  ^bb8:  // pred: ^bb3
    cf.br ^bb9
  ^bb9:  // 2 preds: ^bb7, ^bb8
    cf.cond_br %true, ^bb10, ^bb11
  ^bb10:  // pred: ^bb9
    %77 = ttng.warp_group_dot %40, %42, %61, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
    cf.br ^bb12(%77 : tensor<128x128xf32, #mma>)
  ^bb11:  // pred: ^bb9
    %78 = ttng.warp_group_dot %39, %41, %61, %true {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x128xf16, #shared, #smem, mutable> -> tensor<128x128xf32, #mma>
    cf.br ^bb12(%78 : tensor<128x128xf32, #mma>)
  ^bb12(%79: tensor<128x128xf32, #mma>):  // 2 preds: ^bb10, ^bb11
    cf.br ^bb13
  ^bb13:  // pred: ^bb12
    %80 = ttng.warp_group_dot_wait %79 {pendings = 0 : i32} : tensor<128x128xf32, #mma>
    %81 = ttg.convert_layout %79 {allocation.offset = 0 : i32} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %82 = tt.expand_dims %10 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %83 = tt.splat %9 : i32 -> tensor<128x1xi32, #blocked>
    %84 = arith.addi %83, %82 : tensor<128x1xi32, #blocked>
    %85 = tt.expand_dims %26 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %86 = tt.splat %25 : i32 -> tensor<1x128xi32, #blocked>
    %87 = arith.addi %86, %85 : tensor<1x128xi32, #blocked>
    %88 = arith.cmpi slt, %84, %cst_9 : tensor<128x1xi32, #blocked>
    %89 = arith.cmpi slt, %87, %cst_10 : tensor<1x128xi32, #blocked>
    %90 = tt.broadcast %88 : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %91 = tt.broadcast %89 : tensor<1x128xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %92 = arith.andi %90, %91 : tensor<128x128xi1, #blocked>
    %93 = tt.splat %arg5 : i32 -> tensor<128x1xi32, #blocked>
    %94 = arith.muli %84, %93 : tensor<128x1xi32, #blocked>
    %95 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %96 = tt.addptr %95, %94 : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %97 = tt.broadcast %96 : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %98 = tt.broadcast %87 : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %99 = tt.addptr %97, %98 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %100 = arith.truncf %81 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.store %99, %100, %92 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=87})",
      disable_threading: true,
      verify_each: true
    }
  }
#-}
