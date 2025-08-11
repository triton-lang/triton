// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-data-partition=num-warp-groups=3 | FileCheck %s

// CHECK-LABEL: @matmul_persistent_ws_cooperative_kernel
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 0}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_persistent_ws_cooperative_kernel(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %2 = tt.splat %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
      %3 = tt.splat %arg1 {async_task_id = array<i32: 0>} : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
      %4:2 = scf.for %arg7 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        // CHECK: %[[#GA1:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
        // CHECK: %[[#GA2:]] = tt.load {{.*}} : tensor<64x64x!tt.ptr<f16>
        %8 = tt.load %2 {async_task_id = array<i32: 0>} : tensor<128x64x!tt.ptr<f16>, #blocked>
        // CHECK: %[[#LA1:]] = ttg.local_alloc %[[#GA1]]
        // CHECK: %[[#LA2:]] = ttg.local_alloc %[[#GA2]]
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // CHECK: %[[#GB:]] = tt.load {{.*}} : tensor<64x256x!tt.ptr<f16>
        %10 = tt.load %3 {async_task_id = array<i32: 0>} : tensor<64x256x!tt.ptr<f16>, #blocked1>
        // CHECK: %[[#LB:]] = ttg.local_alloc %[[#GB]]
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        // CHECK: %[[#C1:]] = ttng.warp_group_dot %[[#LA1]], %[[#LB]], {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        // CHECK: %[[#C2:]] = ttng.warp_group_dot %[[#LA2]], %[[#LB]], {{.*}} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<64x256xf32, #mma>
        %12 = ttng.warp_group_dot %9, %11, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %13 = arith.addi %arg9, %c64_i32 {async_task_id = array<i32: 0>} : i32
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %12, %13 : tensor<128x256xf32, #mma>, i32
      } {async_task_id = array<i32: 0, 1, 2>}
      %5 = arith.truncf %4#0 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %6 = ttg.convert_layout %5 {async_task_id = array<i32: 1, 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      %7 = tt.splat %arg2 {async_task_id = array<i32: 1, 2>} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
     // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
     // CHECK: tt.store {{.*}} : tensor<64x256x!tt.ptr<f16>, #blocked1>
     tt.store %7, %6 {async_task_id = array<i32: 1, 2>} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: @cross_dim_partition
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 0}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 0}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cross_dim_partition(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg7: f32, %arg8: i32, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32) {
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant {async_task_id = array<i32: 1, 2>} dense<true> : tensor<128x128xi1, #blocked>
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0>} 64 : i32
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = tt.get_program_id y {async_task_id = array<i32: 0, 1, 2>} : i32
    %2 = tt.load %arg1 {async_task_id = array<i32: 0, 1, 2>} : !tt.ptr<i32>
    %3 = arith.extsi %arg8 {async_task_id = array<i32: 0>} : i32 to i64
    ttng.tensormap_create %arg6, %arg0, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg2, [%c64_i32, %c128_i32], [%arg8, %arg9], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg3, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    ttng.tensormap_create %arg6, %arg5, [%c64_i32, %c64_i32], [%arg8, %2], [%3], [%c1_i32, %c1_i32] {async_task_id = array<i32: 0>, elem_type = 1 : i32, fill_mode = 0 : i32, interleave_layout = 0 : i32, swizzle_mode = 3 : i32} : (!tt.ptr<i8>, !tt.ptr<bf16>, i32, i32, i32, i32, i64, i32, i32) -> ()
    %4 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %5 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %6 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    %7 = ttng.reinterpret_tensor_descriptor %arg6 {async_task_id = array<i32: 0>} : !tt.ptr<i8> to !tt.tensordesc<tensor<128x128xbf16>>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    %8 = tt.descriptor_load %4[%0, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<128x128xbf16
    %10 = tt.descriptor_load %5[%1, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x128xbf16, {{.*}} * !ttg.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<64x128xbf16, {{.*}} * !ttg.memdesc<128x128xbf16, {{.*}} -> tensor<64x128xf32, {{.*}}
     %12 = ttng.warp_group_dot %9, %11, %cst {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared, #smem> * !ttg.memdesc<128x128xbf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    %13 = arith.truncf %12 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #mma> to tensor<128x128xbf16, #mma>
    %14 = ttg.local_alloc %13 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #mma>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    // CHECK: tt.descriptor_load {{.*}} -> tensor<64x128xbf16
    %15 = tt.descriptor_load %6[%0, %1] {async_task_id = array<i32: 0>} : !tt.tensordesc<tensor<128x128xbf16>> -> tensor<128x128xbf16, #blocked1>
    %16 = ttg.local_alloc %15 {async_task_id = array<i32: 1, 2>} : (tensor<128x128xbf16, #blocked1>) -> !ttg.memdesc<128x128xbf16, #shared, #smem>
    %17 = ttg.memdesc_trans %16 {async_task_id = array<i32: 1, 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xbf16, #shared, #smem> -> !ttg.memdesc<128x128xbf16, #shared1, #smem>
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<128x64xbf16, {{.*}} * !ttg.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
    // CHECK: ttng.warp_group_dot {{.*}} : !ttg.memdesc<128x64xbf16, {{.*}} * !ttg.memdesc<64x128xbf16, {{.*}} -> tensor<128x128xf32, {{.*}}
    %18 = ttng.warp_group_dot %17, %14, %cst {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x128xbf16, #shared1, #smem> * !ttg.memdesc<128x128xbf16, #shared, #smem> -> tensor<128x128xf32, #mma>
    %19 = ttg.convert_layout %18 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %20 = arith.truncf %19 {async_task_id = array<i32: 1, 2>} : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %21 = tt.splat %arg4 {async_task_id = array<i32: 1, 2>} : !tt.ptr<bf16> -> tensor<1x128x!tt.ptr<bf16>, #blocked>
    %22 = tt.broadcast %21 {async_task_id = array<i32: 1, 2>} : tensor<1x128x!tt.ptr<bf16>, #blocked> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    %23 = tt.atomic_rmw fadd, relaxed, gpu, %22, %20, %cst_0 {async_task_id = array<i32: 1, 2>} : (tensor<128x128x!tt.ptr<bf16>, #blocked>, tensor<128x128xbf16, #blocked>, tensor<128x128xi1, #blocked>) -> tensor<128x128xbf16, #blocked>
    tt.return
  }
}
