// RUN: triton-opt --split-input-file --allow-unregistered-dialect --convert-triton-gpu-to-llvm="compute-capability=100 ptx-version=86" --mlir-print-ir-after-all %s | FileCheck %s
// RUN: triton-opt --split-input-file --allow-unregistered-dialect --convert-triton-gpu-to-llvm="compute-capability=100 ptx-version=86" %s | not FileCheck %s --check-prefix=BAD

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 32]], warp = [[16, 0], [32, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 512 : i32, ttg.maxnreg = 168 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 58368 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 12 : i32} {
  tt.func public @attention_persistent_inner_loop_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: f32) attributes {allocation.offset = 58360 : i32, noinline = false, ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 512 : i32} {
    %0 = ub.poison : !ttg.async.token
    %true = arith.constant true
    %c64_i32 = arith.constant 64 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c16_i32 = arith.constant 16 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %c2_i32 = arith.constant 2 : i32
    %c128_i64 = arith.constant 128 : i64
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %1, %arg0, [%c64_i32, %c64_i32], [%c64_i32, %c1024_i32], [%c128_i64], [%c1_i32, %c1_i32] {allocation.offset = 0 : i32, elem_type = 6 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %1 : !tt.ptr<i8>
    %2 = ttng.reinterpret_tensor_descriptor %1 : !tt.ptr<i8> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %3 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32, ttg.global_scratch_memory_offset = 128 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %3, %arg1, [%c64_i32, %c64_i32], [%c64_i32, %c1024_i32], [%c128_i64], [%c1_i32, %c1_i32] {allocation.offset = 0 : i32, elem_type = 6 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %3 : !tt.ptr<i8>
    %4 = ttng.reinterpret_tensor_descriptor %3 : !tt.ptr<i8> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %5 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32, ttg.global_scratch_memory_offset = 256 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %5, %arg2, [%c64_i32, %c64_i32], [%c64_i32, %c1024_i32], [%c128_i64], [%c1_i32, %c1_i32] {allocation.offset = 0 : i32, elem_type = 6 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %5 : !tt.ptr<i8>
    %6 = ttng.reinterpret_tensor_descriptor %5 : !tt.ptr<i8> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %7 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32, ttg.global_scratch_memory_offset = 384 : i32} : !tt.ptr<i8>
    ttng.tensormap_create %7, %arg3, [%c64_i32, %c64_i32], [%c64_i32, %c1024_i32], [%c128_i64], [%c1_i32, %c1_i32] {allocation.offset = 0 : i32, elem_type = 6 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<f16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_fenceproxy_acquire %7 : !tt.ptr<i8>
    %8 = ttng.reinterpret_tensor_descriptor %7 : !tt.ptr<i8> to !tt.tensordesc<tensor<64x64xf16, #shared>>
    %9 = tt.get_program_id x : i32
    %10 = tt.get_num_programs x : i32
    %11 = arith.divsi %c16_i32, %10 : i32
    %12 = arith.remsi %c16_i32, %10 : i32
    %13 = arith.cmpi slt, %9, %12 : i32
    cf.cond_br %13, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %14 = arith.addi %11, %c1_i32 : i32
    cf.br ^bb3(%14 : i32)
  ^bb2:  // pred: ^bb0
    cf.br ^bb3(%11 : i32)
  ^bb3(%15: i32):  // 2 preds: ^bb1, ^bb2
    cf.br ^bb4
  ^bb4:  // pred: ^bb3
    %16 = tt.splat %arg6 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %17 = tt.splat %arg6 : f32 -> tensor<64x64xf32, #linear>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %19 = ttg.local_alloc {allocation.offset = 57344 : i32} : () -> !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>
    %20 = ttg.local_alloc {allocation.offset = 58240 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %21 = ttg.memdesc_index %20[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %21, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %22 = ttg.local_alloc {allocation.offset = 58256 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %23 = ttg.memdesc_index %22[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %23, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %24 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %25 = ttg.local_alloc {allocation.offset = 58112 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %26 = ttg.memdesc_index %25[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %26, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %27 = ttg.memdesc_index %25[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %27, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %28 = ttg.local_alloc {allocation.offset = 58128 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %29 = ttg.memdesc_index %28[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %29, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %30 = ttg.memdesc_index %28[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %30, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %31 = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %32 = ttg.local_alloc {allocation.offset = 58144 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %33 = ttg.memdesc_index %32[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %33, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %34 = ttg.memdesc_index %32[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %34, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %35 = ttg.local_alloc {allocation.offset = 58160 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %36 = ttg.memdesc_index %35[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %36, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %37 = ttg.memdesc_index %35[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %37, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %38 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>
    %39 = ttg.local_alloc {allocation.offset = 58176 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %40 = ttg.memdesc_index %39[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %40, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %41 = ttg.memdesc_index %39[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %41, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %42 = ttg.local_alloc {allocation.offset = 58192 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %43 = ttg.memdesc_index %42[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %43, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %44 = ttg.memdesc_index %42[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %44, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %45 = ttg.local_alloc {allocation.offset = 57600 : i32} : () -> !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>
    %46 = ttg.local_alloc {allocation.offset = 58272 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %47 = ttg.memdesc_index %46[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %47, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %48 = ttg.local_alloc {allocation.offset = 58288 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %49 = ttg.memdesc_index %48[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %49, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %50 = ttg.local_alloc {allocation.offset = 58208 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %51 = ttg.memdesc_index %50[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %51, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %52 = ttg.memdesc_index %50[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %52, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %53 = ttg.local_alloc {allocation.offset = 58224 : i32} : () -> !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    %54 = ttg.memdesc_index %53[%c0_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %54, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %55 = ttg.memdesc_index %53[%c1_i32] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %55, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %56 = ttg.local_alloc {allocation.offset = 58304 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %57 = ttg.memdesc_index %56[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %57, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %58 = ttg.local_alloc {allocation.offset = 58320 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %59 = ttg.memdesc_index %58[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %59, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %60 = ttg.local_alloc {allocation.offset = 58336 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %61 = ttg.memdesc_index %60[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %61, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %62 = ttg.local_alloc {allocation.offset = 58352 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %63 = ttg.memdesc_index %62[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %63, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_1 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 16 : i32} : () -> !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %result_2 = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 16 : i32} : () -> !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>
    ttg.warp_specialize(%50, %28, %24, %35, %31, %result, %53, %32, %42, %38, %58, %result_1, %62, %result_2, %60, %56, %39, %25, %15, %2, %4, %6, %10, %9, %22, %19, %20, %48, %45, %46, %8) attributes {actualRegisters = array<i32: 392, 24, 24, 88, 24>, allocation.offset = 49152 : i32, requestedRegisters = array<i32: 24, 24, 88, 16>, warpGroupStartIds = array<i32: 10, 8, 4, 11>}
    default {

      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"

      cf.br ^bb1(%c0_i32, %9, %c0_i32, %c0_i32, %c1_i32, %c1_i32, %c0_i32 : i32, i32, i32, i32, i32, i32, i32)
    ^bb1(%64: i32, %65: i32, %66: i32, %67: i32, %68: i32, %69: i32, %70: i32):  // 2 preds: ^bb0, ^bb5
      %71 = arith.cmpi slt, %64, %15 : i32
      cf.cond_br %71, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %72 = arith.muli %65, %c64_i32 : i32
      cf.br ^bb3(%c0_i32, %cst_0, %cst, %66, %67, %68, %69, %70 : i32, tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, i32, i32, i32, i32)
    ^bb3(%73: i32, %74: tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, %75: tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, %76: i32, %77: i32, %78: i32, %79: i32, %80: i32):  // 2 preds: ^bb2, ^bb4
      %81 = arith.cmpi slt, %73, %c1024_i32 : i32
      cf.cond_br %81, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %82 = arith.xori %76, %c1_i32 : i32
      ttng.wait_barrier %21, %82 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %83 = ttg.memdesc_index %19[%c0_i32] : !ttg.memdesc<1x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      ttg.local_store %74, %83 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      ttng.arrive_barrier %23, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %84 = arith.addi %78, %c1_i32 : i32
      %85 = arith.cmpi eq, %84, %c2_i32 : i32
      %86 = arith.select %85, %c0_i32, %84 : i32
      %87 = arith.xori %79, %c1_i32 : i32
      %88 = arith.select %85, %87, %79 : i32
      %89 = ttg.memdesc_index %53[%86] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %89, %88 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %90 = ttg.memdesc_index %result[%86] : !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %91 = ttg.memdesc_index %50[%86] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_3 = ttng.tmem_load %90 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear>
      ttng.arrive_barrier %91, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %92 = "tt.reduce"(%result_3) <{axis = 1 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %128 = arith.maxnumf %arg7, %arg8 : f32
        tt.reduce.return %128 : f32
      }) : (tensor<64x64xf32, #linear>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %93 = arith.mulf %92, %16 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %94 = arith.maxnumf %74, %93 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %95 = arith.xori %77, %c1_i32 : i32
      ttng.wait_barrier %47, %95 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %96 = ttg.memdesc_index %45[%c0_i32] : !ttg.memdesc<1x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      ttg.local_store %94, %96 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      ttng.arrive_barrier %49, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %97 = arith.mulf %result_3, %17 : tensor<64x64xf32, #linear>
      %98 = tt.expand_dims %94 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xf32, #linear>
      %99 = tt.broadcast %98 : tensor<64x1xf32, #linear> -> tensor<64x64xf32, #linear>
      %100 = arith.subf %97, %99 : tensor<64x64xf32, #linear>
      %101 = math.exp2 %100 : tensor<64x64xf32, #linear>
      %102 = arith.subf %74, %94 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %103 = math.exp2 %102 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %104 = "tt.reduce"(%101) <{axis = 1 : i32}> ({
      ^bb0(%arg7: f32, %arg8: f32):
        %128 = arith.addf %arg7, %arg8 : f32
        tt.reduce.return %128 : f32
      }) : (tensor<64x64xf32, #linear>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %105 = arith.truncf %101 : tensor<64x64xf32, #linear> to tensor<64x64xf16, #linear>
      %106 = arith.xori %80, %c1_i32 : i32
      ttng.wait_barrier %61, %106 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %107 = ttg.memdesc_index %result_2[%c0_i32] : !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %105, %107, %true : tensor<64x64xf16, #linear> -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
      ttng.arrive_barrier %63, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %108 = arith.mulf %75, %103 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %109 = arith.addf %108, %104 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %110 = arith.addi %73, %c64_i32 : i32
      cf.br ^bb3(%110, %94, %109, %82, %95, %86, %88, %106 : i32, tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, i32, i32, i32, i32)
    ^bb5:  // pred: ^bb3
      %111 = arith.addi %78, %c1_i32 : i32
      %112 = arith.cmpi eq, %111, %c2_i32 : i32
      %113 = arith.select %112, %c0_i32, %111 : i32
      %114 = arith.xori %79, %c1_i32 : i32
      %115 = arith.select %112, %114, %79 : i32
      %116 = ttg.memdesc_index %53[%113] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %116, %115 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %117 = ttg.memdesc_index %50[%113] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %117, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %118 = tt.addptr %arg4, %72 : !tt.ptr<f32>, i32
      %119 = tt.splat %118 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
      %120 = tt.addptr %119, %18 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
      %121 = ttg.convert_layout %75 {allocation.offset = 57856 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64xf32, #blocked>
      tt.store %120, %121 : tensor<64x!tt.ptr<f32>, #blocked>
      %122 = tt.addptr %arg5, %72 : !tt.ptr<f32>, i32
      %123 = tt.splat %122 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked>
      %124 = tt.addptr %123, %18 : tensor<64x!tt.ptr<f32>, #blocked>, tensor<64xi32, #blocked>
      %125 = ttg.convert_layout %74 {allocation.offset = 57856 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64xf32, #blocked>
      tt.store %124, %125 : tensor<64x!tt.ptr<f32>, #blocked>
      %126 = arith.addi %65, %10 : i32
      %127 = arith.addi %64, %c1_i32 : i32
      cf.br ^bb1(%127, %126, %76, %77, %113, %115, %80 : i32, i32, i32, i32, i32, i32, i32)
    ^bb6:  // pred: ^bb1
      ttg.warp_yield
    }
    partition0(%arg7: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg9: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg10: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg12: !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg17: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg19: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg20: !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>, %arg21: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg23: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg24: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg25: i32, %arg26: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg29: i32, %arg30: i32, %arg31: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg37: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(1) {
      %false = arith.constant false
      %true_3 = arith.constant true
      %c64_i32_4 = arith.constant 64 : i32
      %c1024_i32_5 = arith.constant 1024 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c0_i32_7 = arith.constant 0 : i32
      %c2_i32_8 = arith.constant 2 : i32
      %c896_i32 = arith.constant 896 : i32
      cf.br ^bb1(%c0_i32_7, %c1_i32_6, %c1_i32_6, %c1_i32_6, %c1_i32_6, %c1_i32_6, %c1_i32_6, %c1_i32_6, %c0_i32_7, %c1_i32_6, %c1_i32_6 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb1(%64: i32, %65: i32, %66: i32, %67: i32, %68: i32, %69: i32, %70: i32, %71: i32, %72: i32, %73: i32, %74: i32):  // 2 preds: ^bb0, ^bb5
      %75 = arith.cmpi slt, %64, %arg25 : i32
      cf.cond_br %75, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %76 = arith.addi %71, %c1_i32_6 : i32
      %77 = arith.cmpi eq, %76, %c2_i32_8 : i32
      %78 = arith.select %77, %c0_i32_7, %76 : i32
      %79 = arith.xori %72, %c1_i32_6 : i32
      %80 = arith.select %77, %79, %72 : i32
      %81 = ttg.memdesc_index %arg7[%78] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %81, %80 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %82 = arith.addi %65, %c1_i32_6 : i32
      %83 = arith.cmpi eq, %82, %c2_i32_8 : i32
      %84 = arith.select %83, %c0_i32_7, %82 : i32
      %85 = arith.xori %66, %c1_i32_6 : i32
      %86 = arith.select %83, %85, %66 : i32
      %87 = ttg.memdesc_index %arg8[%84] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %87, %86 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %88 = ttg.memdesc_index %arg9[%84] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %89 = arith.addi %67, %c1_i32_6 : i32
      %90 = arith.cmpi eq, %89, %c2_i32_8 : i32
      %91 = arith.select %90, %c0_i32_7, %89 : i32
      %92 = arith.xori %68, %c1_i32_6 : i32
      %93 = arith.select %90, %92, %68 : i32
      %94 = ttg.memdesc_index %arg10[%91] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %94, %93, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %95 = ttg.memdesc_index %arg11[%91] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %96 = ttg.memdesc_trans %95 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %97 = ttg.memdesc_index %arg12[%78] : !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %98 = ttg.memdesc_index %arg13[%78] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %99 = ttg.memdesc_index %arg14[%91] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %88, %96, %97, %false, %true_3, %98[%true_3], %99[%true_3] {is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %100 = arith.addi %78, %c1_i32_6 : i32
      %101 = arith.cmpi eq, %100, %c2_i32_8 : i32
      %102 = arith.select %101, %c0_i32_7, %100 : i32
      %103 = arith.xori %80, %c1_i32_6 : i32
      %104 = arith.select %101, %103, %80 : i32
      %105 = ttg.memdesc_index %arg7[%102] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %105, %104, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %106 = arith.addi %91, %c1_i32_6 : i32
      %107 = arith.cmpi eq, %106, %c2_i32_8 : i32
      %108 = arith.select %107, %c0_i32_7, %106 : i32
      %109 = arith.xori %93, %c1_i32_6 : i32
      %110 = arith.select %107, %109, %93 : i32
      %111 = ttg.memdesc_index %arg10[%108] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %111, %110, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %112 = ttg.memdesc_index %arg11[%108] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %113 = ttg.memdesc_trans %112 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %114 = ttg.memdesc_index %arg12[%102] : !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %115 = ttg.memdesc_index %arg13[%102] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %116 = ttg.memdesc_index %arg14[%108] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %88, %113, %114, %false, %true_3, %115[%true_3], %116[%true_3] {is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %117 = arith.addi %102, %c1_i32_6 : i32
      %118 = arith.cmpi eq, %117, %c2_i32_8 : i32
      %119 = arith.select %118, %c0_i32_7, %117 : i32
      %120 = arith.xori %104, %c1_i32_6 : i32
      %121 = arith.select %118, %120, %104 : i32
      %122 = ttg.memdesc_index %arg7[%119] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %122, %121, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      cf.br ^bb3(%c0_i32_7, %108, %110, %69, %70, %119, %121, %73, %74 : i32, i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb3(%123: i32, %124: i32, %125: i32, %126: i32, %127: i32, %128: i32, %129: i32, %130: i32, %131: i32):  // 2 preds: ^bb2, ^bb4
      %132 = arith.cmpi slt, %123, %c1024_i32_5 : i32
      cf.cond_br %132, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %133 = arith.cmpi slt, %123, %c896_i32 : i32
      %134 = arith.addi %126, %c1_i32_6 : i32
      %135 = arith.cmpi eq, %134, %c2_i32_8 : i32
      %136 = arith.select %135, %c0_i32_7, %134 : i32
      %137 = arith.xori %127, %c1_i32_6 : i32
      %138 = arith.select %135, %137, %127 : i32
      %139 = ttg.memdesc_index %arg15[%136] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %139, %138 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %140 = ttg.memdesc_index %arg16[%136] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %141 = arith.xori %130, %c1_i32_6 : i32
      %142 = ttg.memdesc_index %arg17[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %142, %141 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %143 = ttg.memdesc_index %arg18[%c0_i32_7] : !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %144 = arith.xori %131, %c1_i32_6 : i32
      %145 = ttg.memdesc_index %arg19[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %145, %144 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %146 = ttg.memdesc_index %arg20[%c0_i32_7] : !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>
      %147 = ttg.memdesc_index %arg21[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %148 = ttg.memdesc_index %arg22[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %149 = ttg.memdesc_index %arg23[%136] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %146, %140, %143, %true_3, %true_3, %147[%true_3], %148[%true_3], %149[%true_3] {is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %150 = arith.addi %124, %c1_i32_6 : i32
      %151 = arith.cmpi eq, %150, %c2_i32_8 : i32
      %152 = arith.select %151, %c0_i32_7, %150 : i32
      %153 = arith.xori %125, %c1_i32_6 : i32
      %154 = arith.select %151, %153, %125 : i32
      %155 = ttg.memdesc_index %arg10[%152] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %155, %154, %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %156 = ttg.memdesc_index %arg11[%152] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %157 = ttg.memdesc_trans %156 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>
      %158 = ttg.memdesc_index %arg12[%128] : !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %159 = ttg.memdesc_index %arg13[%128] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %160 = ttg.memdesc_index %arg14[%152] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_mma %88, %157, %158, %false, %133, %159[%133], %160[%133] {is_async, tt.self_latency = 1 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared2, #smem, mutable>, !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %161 = arith.addi %128, %c1_i32_6 : i32
      %162 = arith.cmpi eq, %161, %c2_i32_8 : i32
      %163 = arith.select %162, %c0_i32_7, %161 : i32
      %164 = arith.xori %129, %c1_i32_6 : i32
      %165 = arith.select %162, %164, %129 : i32
      %166 = ttg.memdesc_index %arg7[%163] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %166, %165, %133 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %167 = arith.select %133, %152, %124 : i32
      %168 = arith.select %133, %154, %125 : i32
      %169 = arith.select %133, %163, %128 : i32
      %170 = arith.select %133, %165, %129 : i32
      %171 = arith.addi %123, %c64_i32_4 : i32
      cf.br ^bb3(%171, %167, %168, %136, %138, %169, %170, %141, %144 : i32, i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb5:  // pred: ^bb3

      // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];"
      // CHECK: "@$0 mbarrier.arrive.shared::cta.b64 _, [$1];"

      %172 = ttg.memdesc_index %arg13[%128] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %172, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %173 = ttg.memdesc_index %arg24[%84] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.tc_gen5_commit %173 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %174 = arith.xori %130, %c1_i32_6 : i32
      %175 = ttg.memdesc_index %arg17[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %175, %174 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %176 = ttg.memdesc_index %arg22[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %176, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %177 = arith.addi %64, %c1_i32_6 : i32
      cf.br ^bb1(%177, %84, %86, %124, %125, %126, %127, %128, %129, %174, %131 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb6:  // pred: ^bb1
      ttg.warp_return
    }
    partition1(%arg7: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg9: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg10: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg12: !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg17: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg19: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg20: !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>, %arg21: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg23: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg24: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg25: i32, %arg26: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg29: i32, %arg30: i32, %arg31: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg37: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(2) {
      %true_3 = arith.constant true
      %c64_i32_4 = arith.constant 64 : i32
      %c1024_i32_5 = arith.constant 1024 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c0_i32_7 = arith.constant 0 : i32
      %c2_i32_8 = arith.constant 2 : i32
      %c896_i32 = arith.constant 896 : i32
      %c128_i32 = arith.constant 128 : i32
      cf.br ^bb1(%c0_i32_7, %arg30, %c1_i32_6, %c0_i32_7, %c1_i32_6, %c0_i32_7, %c1_i32_6, %c0_i32_7 : i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb1(%64: i32, %65: i32, %66: i32, %67: i32, %68: i32, %69: i32, %70: i32, %71: i32):  // 2 preds: ^bb0, ^bb5
      %72 = arith.cmpi slt, %64, %arg25 : i32
      cf.cond_br %72, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %73 = arith.muli %65, %c64_i32_4 : i32
      %74 = arith.addi %66, %c1_i32_6 : i32
      %75 = arith.cmpi eq, %74, %c2_i32_8 : i32
      %76 = arith.select %75, %c0_i32_7, %74 : i32
      %77 = arith.xori %67, %c1_i32_6 : i32
      %78 = arith.select %75, %77, %67 : i32
      %79 = ttg.memdesc_index %arg24[%76] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %79, %78 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %80 = ttg.memdesc_index %arg9[%76] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %81 = ttg.memdesc_index %arg8[%76] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %81, 8192, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.async_tma_copy_global_to_local %arg26[%73, %c0_i32_7] %80, %81, %true_3 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %82 = arith.addi %68, %c1_i32_6 : i32
      %83 = arith.cmpi eq, %82, %c2_i32_8 : i32
      %84 = arith.select %83, %c0_i32_7, %82 : i32
      %85 = arith.xori %69, %c1_i32_6 : i32
      %86 = arith.select %83, %85, %69 : i32
      %87 = ttg.memdesc_index %arg14[%84] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %87, %86, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %88 = ttg.memdesc_index %arg11[%84] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %89 = ttg.memdesc_index %arg10[%84] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %89, 8192, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.async_tma_copy_global_to_local %arg27[%c0_i32_7, %c0_i32_7] %88, %89, %true_3 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %90 = arith.addi %84, %c1_i32_6 : i32
      %91 = arith.cmpi eq, %90, %c2_i32_8 : i32
      %92 = arith.select %91, %c0_i32_7, %90 : i32
      %93 = arith.xori %86, %c1_i32_6 : i32
      %94 = arith.select %91, %93, %86 : i32
      %95 = ttg.memdesc_index %arg14[%92] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %95, %94, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %96 = ttg.memdesc_index %arg11[%92] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %97 = ttg.memdesc_index %arg10[%92] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %97, 8192, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.async_tma_copy_global_to_local %arg27[%c64_i32_4, %c0_i32_7] %96, %97, %true_3 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      cf.br ^bb3(%c0_i32_7, %92, %94, %70, %71 : i32, i32, i32, i32, i32)
    ^bb3(%98: i32, %99: i32, %100: i32, %101: i32, %102: i32):  // 2 preds: ^bb2, ^bb4
      %103 = arith.cmpi slt, %98, %c1024_i32_5 : i32
      cf.cond_br %103, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3
      %104 = arith.cmpi slt, %98, %c896_i32 : i32
      %105 = arith.addi %101, %c1_i32_6 : i32
      %106 = arith.cmpi eq, %105, %c2_i32_8 : i32
      %107 = arith.select %106, %c0_i32_7, %105 : i32
      %108 = arith.xori %102, %c1_i32_6 : i32
      %109 = arith.select %106, %108, %102 : i32
      %110 = ttg.memdesc_index %arg23[%107] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %110, %109 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %111 = ttg.memdesc_index %arg16[%107] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %112 = ttg.memdesc_index %arg15[%107] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %112, 8192, %true_3 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.async_tma_copy_global_to_local %arg28[%98, %c0_i32_7] %111, %112, %true_3 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %113 = arith.addi %99, %c1_i32_6 : i32
      %114 = arith.cmpi eq, %113, %c2_i32_8 : i32
      %115 = arith.select %114, %c0_i32_7, %113 : i32
      %116 = arith.xori %100, %c1_i32_6 : i32
      %117 = arith.select %114, %116, %100 : i32
      %118 = ttg.memdesc_index %arg14[%115] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %118, %117, %104 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %119 = ttg.memdesc_index %arg11[%115] : !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %120 = ttg.memdesc_index %arg10[%115] : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %120, 8192, %104 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %121 = arith.addi %98, %c128_i32 : i32
      ttng.async_tma_copy_global_to_local %arg27[%121, %c0_i32_7] %119, %120, %104 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %122 = arith.select %104, %115, %99 : i32
      %123 = arith.select %104, %117, %100 : i32
      %124 = arith.addi %98, %c64_i32_4 : i32
      cf.br ^bb3(%124, %122, %123, %107, %109 : i32, i32, i32, i32, i32)
    ^bb5:  // pred: ^bb3
      %125 = arith.addi %65, %arg29 : i32
      %126 = arith.addi %64, %c1_i32_6 : i32
      cf.br ^bb1(%126, %125, %76, %78, %99, %100, %101, %102 : i32, i32, i32, i32, i32, i32, i32, i32)
    ^bb6:  // pred: ^bb1
      ttg.warp_return
    }
    partition2(%arg7: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg9: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg10: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg12: !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg17: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg19: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg20: !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>, %arg21: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg23: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg24: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg25: i32, %arg26: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg29: i32, %arg30: i32, %arg31: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg37: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(4) {
      %true_3 = arith.constant true
      %c64_i32_4 = arith.constant 64 : i32
      %c1024_i32_5 = arith.constant 1024 : i32
      %c1_i32_6 = arith.constant 1 : i32
      %c0_i32_7 = arith.constant 0 : i32
      %cst_8 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #linear>
      %64 = ttg.local_alloc {allocation.offset = 49152 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      cf.br ^bb1(%c0_i32_7, %arg30, %c1_i32_6, %c1_i32_6, %c0_i32_7 : i32, i32, i32, i32, i32)
    ^bb1(%65: i32, %66: i32, %67: i32, %68: i32, %69: i32):  // 2 preds: ^bb0, ^bb5
      %70 = arith.cmpi slt, %65, %arg25 : i32
      cf.cond_br %70, ^bb2, ^bb6
    ^bb2:  // pred: ^bb1
      %71 = arith.muli %66, %c64_i32_4 : i32
      %72 = arith.xori %69, %c1_i32_6 : i32
      %73 = ttg.memdesc_index %arg22[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %73, %72 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %74 = ttg.memdesc_index %arg18[%c0_i32_7] : !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %cst_8, %74, %true_3 : tensor<64x64xf32, #linear> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      cf.br ^bb3(%c0_i32_7, %67, %68, %72 : i32, i32, i32, i32)
    ^bb3(%75: i32, %76: i32, %77: i32, %78: i32):  // 2 preds: ^bb2, ^bb4
      %79 = arith.cmpi slt, %75, %c1024_i32_5 : i32
      cf.cond_br %79, ^bb4, ^bb5
    ^bb4:  // pred: ^bb3

      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"
      // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"

      %80 = arith.xori %76, %c1_i32_6 : i32
      %81 = ttg.memdesc_index %arg31[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %81, %80 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %82 = ttg.memdesc_index %arg32[%c0_i32_7] : !ttg.memdesc<1x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      %83 = ttg.local_load %82 : !ttg.memdesc<64xf32, #shared1, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %84 = ttg.memdesc_index %arg33[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %84, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %85 = arith.xori %77, %c1_i32_6 : i32
      %86 = ttg.memdesc_index %arg34[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %86, %85 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %87 = ttg.memdesc_index %arg35[%c0_i32_7] : !ttg.memdesc<1x64xf32, #shared1, #smem, mutable> -> !ttg.memdesc<64xf32, #shared1, #smem, mutable>
      %88 = ttg.local_load %87 : !ttg.memdesc<64xf32, #shared1, #smem, mutable> -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %89 = ttg.memdesc_index %arg36[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %89, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %90 = arith.subf %83, %88 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %91 = math.exp2 %90 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %92 = tt.expand_dims %91 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<64x1xf32, #linear>
      %93 = tt.broadcast %92 : tensor<64x1xf32, #linear> -> tensor<64x64xf32, #linear>
      %result_9 = ttng.tmem_load %74 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear>
      %94 = arith.mulf %result_9, %93 : tensor<64x64xf32, #linear>
      ttng.tmem_store %94, %74, %true_3 : tensor<64x64xf32, #linear> -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %95 = ttg.memdesc_index %arg17[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %95, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %96 = arith.xori %78, %c1_i32_6 : i32
      ttng.wait_barrier %73, %96 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %97 = arith.addi %75, %c64_i32_4 : i32
      cf.br ^bb3(%97, %80, %85, %96 : i32, i32, i32, i32)
    ^bb5:  // pred: ^bb3
      %98 = ttg.memdesc_index %arg17[%c0_i32_7] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %result_10 = ttng.tmem_load %74 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32, #linear>
      ttng.arrive_barrier %98, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %99 = arith.truncf %result_10 : tensor<64x64xf32, #linear> to tensor<64x64xf16, #linear>
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.local_store %99, %64 : tensor<64x64xf16, #linear> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttng.fence_async_shared {bCluster = false}
      ttng.async_tma_copy_local_to_global %arg37[%71, %c0_i32_7] %64 : !tt.tensordesc<tensor<64x64xf16, #shared>>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      %100 = arith.addi %66, %arg29 : i32
      %101 = arith.addi %65, %c1_i32_6 : i32
      cf.br ^bb1(%101, %100, %76, %77, %78 : i32, i32, i32, i32, i32)
    ^bb6:  // pred: ^bb1
      ttng.async_tma_store_wait {pendings = 0 : i32}
      ttg.local_dealloc %64 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition3(%arg7: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg8: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg9: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg10: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg11: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg12: !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg13: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg14: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg15: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg16: !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, %arg17: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg18: !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, %arg19: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg20: !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>, %arg21: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg22: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg23: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg24: !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, %arg25: i32, %arg26: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg27: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg28: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg29: i32, %arg30: i32, %arg31: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg32: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg33: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg34: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg35: !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, %arg36: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg37: !tt.tensordesc<tensor<64x64xf16, #shared>>) num_warps(1) {
      ttg.warp_return
    } : (!ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x64x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x64x64xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>, i32, !tt.tensordesc<tensor<64x64xf16, #shared>>, !tt.tensordesc<tensor<64x64xf16, #shared>>, !tt.tensordesc<tensor<64x64xf16, #shared>>, i32, i32, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x64xf32, #shared1, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !tt.tensordesc<tensor<64x64xf16, #shared>>) -> ()
    ttng.inval_barrier %21 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %20 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %23 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %22 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %26 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %27 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %25 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %29 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %30 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %28 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %33 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %34 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %32 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %36 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %37 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %35 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %40 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %41 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %39 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %43 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %44 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %42 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %47 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %46 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %49 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %48 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %51 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %52 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %50 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %54 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %55 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %53 : !ttg.memdesc<2x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %57 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %56 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %59 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %58 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %61 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %60 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %63 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %62 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32, ttg.maxnreg = 256 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 32860 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @auto_simple
  tt.func @auto_simple(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg2: i32) attributes {allocation.offset = 32856 : i32, ttg.global_scratch_memory_alignment = 1 : i32, ttg.global_scratch_memory_size = 0 : i32} {

    // CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;"
    // CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;"
    // CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];"

    // BAD: nvvm.fence.proxy
    // BAD: nvvm.barrier0
    // BAD-NEXT: "mbarrier.arrive.shared::cta.b64 _, [$0];"

    %0 = ub.poison : !ttg.async.token
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %1 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 32832 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %3 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %3, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %4 = ttg.local_alloc {allocation.offset = 32848 : i32} : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %5 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %5, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %6:3 = ttg.warp_specialize(%2, %1, %4, %arg0, %arg2) attributes {actualRegisters = array<i32: 488, 24, 24, 24>, allocation.offset = 32768 : i32, requestedRegisters = array<i32: 24, 24, 16>, warpGroupStartIds = array<i32: 6, 4, 7>}
    default {
      cf.br ^bb1(%c0_i32, %cst, %c0_i32, %c1_i32 : i32, tensor<128x128xf32, #blocked>, i32, i32)
    ^bb1(%9: i32, %10: tensor<128x128xf32, #blocked>, %11: i32, %12: i32):  // 2 preds: ^bb0, ^bb2
      %13 = arith.cmpi slt, %9, %arg2 : i32
      cf.cond_br %13, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %14 = arith.addi %11, %c1_i32 : i32
      %15 = arith.cmpi eq, %14, %c1_i32 : i32
      %16 = arith.select %15, %c0_i32, %14 : i32
      %17 = arith.xori %12, %c1_i32 : i32
      %18 = arith.select %15, %17, %12 : i32
      %19 = ttg.memdesc_index %4[%16] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %19, %18 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %20 = ttg.memdesc_index %1[%16] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %21 = ttg.local_load %20 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
      ttng.fence_async_shared {bCluster = false}
      %22 = ttg.memdesc_index %2[%16] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.arrive_barrier %22, 1 frequency = per_warp : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %23 = arith.extf %21 : tensor<128x128xf16, #blocked> to tensor<128x128xf32, #blocked>
      %24 = arith.addf %23, %23 : tensor<128x128xf32, #blocked>
      "consumer"(%24) : (tensor<128x128xf32, #blocked>) -> ()
      %25 = arith.addi %9, %c1_i32 : i32
      cf.br ^bb1(%25, %24, %16, %18 : i32, tensor<128x128xf32, #blocked>, i32, i32)
    ^bb3:  // pred: ^bb1
      ttg.warp_yield %10, %11, %12 : tensor<128x128xf32, #blocked>, i32, i32
    }
    partition0(%arg3: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg4: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg5: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg6: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg7: i32) num_warps(1) {
      ttg.warp_return
    }
    partition1(%arg3: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg4: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg5: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg6: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg7: i32) num_warps(2) {
      %true = arith.constant true
      %c1_i32_0 = arith.constant 1 : i32
      %c0_i32_1 = arith.constant 0 : i32
      cf.br ^bb1(%c0_i32_1, %c0_i32_1, %c0_i32_1 : i32, i32, i32)
    ^bb1(%9: i32, %10: i32, %11: i32):  // 2 preds: ^bb0, ^bb2
      %12 = arith.cmpi slt, %9, %arg7 : i32
      cf.cond_br %12, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %13 = arith.addi %10, %c1_i32_0 : i32
      %14 = arith.cmpi eq, %13, %c1_i32_0 : i32
      %15 = arith.select %14, %c0_i32_1, %13 : i32
      %16 = arith.xori %11, %c1_i32_0 : i32
      %17 = arith.select %14, %16, %11 : i32
      %18 = ttg.memdesc_index %arg3[%15] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.wait_barrier %18, %17 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      %19 = ttg.memdesc_index %arg4[%15] : !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %20 = ttg.memdesc_index %arg5[%15] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.barrier_expect %20, 32768, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttng.async_tma_copy_global_to_local %arg6[%9, %9] %19, %20, %true : !tt.tensordesc<tensor<128x128xf16, #shared>>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %21 = arith.addi %9, %c1_i32_0 : i32
      cf.br ^bb1(%21, %15, %17 : i32, i32, i32)
    ^bb3:  // pred: ^bb1
      ttg.warp_return
    }
    partition2(%arg3: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg4: !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, %arg5: !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, %arg6: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg7: i32) num_warps(1) {
      ttg.warp_return
    } : (!ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>, !tt.tensordesc<tensor<128x128xf16, #shared>>, i32) -> (tensor<128x128xf32, #blocked>, i32, i32)
    %7 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %7 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %2 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %8 = ttg.memdesc_index %4[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.inval_barrier %8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_dealloc %4 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    "final_use"(%6#0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}
