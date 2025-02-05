#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [64, 0], [128, 0]], lane = [[0, 8], [0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_blocks_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<64> : tensor<256x64xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %c256_i32 = arith.constant 256 : i32
    %c255_i32 = arith.constant 255 : i32
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %1 = tt.get_program_id x : i32
    %2 = arith.addi %arg4, %c255_i32 : i32
    %3 = arith.divsi %2, %c256_i32 : i32
    %4 = arith.divsi %1, %3 : i32
    %5 = arith.remsi %1, %3 : i32
    %6 = arith.addi %arg3, %c255_i32 : i32
    %7 = arith.divsi %6, %c256_i32 : i32
    %8 = arith.subi %7, %4 : i32
    %9 = arith.minsi %8, %c1_i32 : i32
    %10 = arith.remsi %5, %9 : i32
    %11 = arith.addi %4, %10 : i32
    %12 = arith.muli %11, %c256_i32 : i32
    %13 = tt.splat %12 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %15 = arith.addi %13, %14 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    %17 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked>
    %18 = arith.muli %16, %17 : tensor<256x1xi32, #blocked>
    %19 = tt.broadcast %18 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %20 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %21 = tt.expand_dims %20 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %22 = tt.broadcast %21 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %23 = arith.addi %19, %22 : tensor<256x64xi32, #blocked>
    %24 = tt.addptr %0, %23 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
    %25 = arith.addi %arg5, %c63_i32 : i32
    %26 = arith.divsi %25, %c64_i32 : i32
    %27 = arith.cmpi sgt, %26, %c0_i32 : i32
    %28 = tt.splat %27 : i1 -> tensor<256x64xi1, #blocked>
    %29 = tt.load %24, %28 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64x!tt.ptr<f16>, #blocked>
    %30 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %31 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %32 = tt.expand_dims %31 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<64x1xi32, #blocked1>
    %33 = tt.splat %arg7 : i32 -> tensor<64x1xi32, #blocked1>
    %34 = arith.muli %32, %33 : tensor<64x1xi32, #blocked1>
    %35 = tt.broadcast %34 : tensor<64x1xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %36 = arith.divsi %5, %9 : i32
    %37 = arith.muli %36, %c256_i32 : i32
    %38 = tt.splat %37 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %39 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %40 = arith.addi %38, %39 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %42 = tt.broadcast %41 : tensor<1x256xi32, #blocked1> -> tensor<64x256xi32, #blocked1>
    %43 = arith.addi %35, %42 : tensor<64x256xi32, #blocked1>
    %44 = tt.addptr %30, %43 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
    %45 = tt.splat %27 : i1 -> tensor<64x256xi1, #blocked1>
    %46 = tt.load %44, %45 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
    %47 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %48 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %49 = tt.splat %12 : i32 -> tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %50 = arith.addi %49, %47 : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %51 = tt.splat %37 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %52 = arith.addi %51, %48 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %53 = tt.expand_dims %50 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<256x1xi32, #mma>
    %54 = tt.expand_dims %52 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x256xi32, #mma>
    %55 = arith.muli %arg7, %c64_i32 : i32
    %56 = tt.splat %55 : i32 -> tensor<64x256xi32, #blocked1>
    %57 = ttg.local_alloc  : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %58 = ttg.local_alloc  : () -> !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable>
    %59 = ttg.memdesc_subview %57[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %60 = ttg.convert_layout %29 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #linear>
    ttg.local_store %60, %59 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #linear> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
    %61 = ttg.memdesc_subview %58[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    ttg.local_store %46, %61 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    %62 = arith.subi %26, %c1_i32 : i32
    %63:6 = scf.for %arg9 = %c0_i32 to %62 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %44, %arg12 = %c0_i32, %arg13 = %59, %arg14 = %61, %arg15 = %24) -> (tensor<256x256xf32, #mma>, tensor<64x256x!tt.ptr<f16>, #blocked1>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>, tensor<256x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %84 = tt.addptr %arg15, %cst : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
      %85 = tt.load %84 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64x!tt.ptr<f16>, #blocked>
      %86 = tt.addptr %arg11, %56 : tensor<64x256x!tt.ptr<f16>, #blocked1>, tensor<64x256xi32, #blocked1>
      %87 = ttg.local_load %arg13 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %88 = ttg.local_load %arg14 : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %89 = tt.load %86 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %90 = tt.dot %87, %88, %arg10, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      %91 = arith.addi %arg12, %c1_i32 : i32
      %92 = arith.cmpi slt, %91, %c1_i32 : i32
      %93 = arith.select %92, %91, %c0_i32 : i32
      %94 = ttg.memdesc_subview %57[%93, %c0_i32, %c0_i32] : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %95 = ttg.convert_layout %85 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #linear>
      ttg.local_store %95, %94 {OpIdx = #amdgpu.OpIdx<0>} : tensor<256x64xf16, #linear> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      %96 = ttg.memdesc_subview %58[%93, %c0_i32, %c0_i32] : !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      ttg.local_store %89, %96 {OpIdx = #amdgpu.OpIdx<1>} : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      scf.yield %90, %86, %93, %94, %96, %84 : tensor<256x256xf32, #mma>, tensor<64x256x!tt.ptr<f16>, #blocked1>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>, tensor<256x64x!tt.ptr<f16>, #blocked>
    }
    %64 = arith.cmpi sge, %26, %c1_i32 : i32
    %65 = ttg.local_load %63#3 : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %66 = ttg.local_load %63#4 : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %67 = scf.if %64 -> (tensor<256x256xf32, #mma>) {
      %84 = tt.dot %65, %66, %63#0, inputPrecision = tf32 : tensor<256x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
      scf.yield %84 : tensor<256x256xf32, #mma>
    } else {
      scf.yield %63#0 : tensor<256x256xf32, #mma>
    }
    %68 = arith.select %64, %67, %63#0 : tensor<256x256xf32, #mma>
    ttg.local_dealloc %57 : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %58 : !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable>
    %69 = arith.truncf %68 : tensor<256x256xf32, #mma> to tensor<256x256xf16, #mma>
    %70 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #mma>
    %71 = arith.muli %70, %53 : tensor<256x1xi32, #mma>
    %72 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<256x1x!tt.ptr<f16>, #mma>
    %73 = tt.addptr %72, %71 : tensor<256x1x!tt.ptr<f16>, #mma>, tensor<256x1xi32, #mma>
    %74 = tt.broadcast %73 : tensor<256x1x!tt.ptr<f16>, #mma> -> tensor<256x256x!tt.ptr<f16>, #mma>
    %75 = tt.broadcast %54 : tensor<1x256xi32, #mma> -> tensor<256x256xi32, #mma>
    %76 = tt.addptr %74, %75 : tensor<256x256x!tt.ptr<f16>, #mma>, tensor<256x256xi32, #mma>
    %77 = tt.splat %arg3 : i32 -> tensor<256x1xi32, #mma>
    %78 = arith.cmpi slt, %53, %77 : tensor<256x1xi32, #mma>
    %79 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #mma>
    %80 = arith.cmpi slt, %54, %79 : tensor<1x256xi32, #mma>
    %81 = tt.broadcast %78 : tensor<256x1xi1, #mma> -> tensor<256x256xi1, #mma>
    %82 = tt.broadcast %80 : tensor<1x256xi1, #mma> -> tensor<256x256xi1, #mma>
    %83 = arith.andi %81, %82 : tensor<256x256xi1, #mma>
    tt.store %76, %69, %83 : tensor<256x256x!tt.ptr<f16>, #mma>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(decompose-unsupported-amd-conversions{arch=gfx942}, optimize-amd-lds-usage{lds-limit=0 target-arch=gfx942}, convert-scf-to-cf, convert-index-to-llvm{index-bitwidth=0}, allocate-shared-memory, convert-triton-amdgpu-to-llvm{arch=gfx942 ftz=true}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, convert-cf-to-llvm{index-bitwidth=0}, convert-arith-to-llvm{index-bitwidth=0}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce, enable-line-info, convert-builtin-func-to-llvm{ftz=true})",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
