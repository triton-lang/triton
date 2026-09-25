// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition=num-buffers=1 -verify-diagnostics=only-expected | FileCheck %s

// CHECK-LABEL: @matmul_kernel_one_consumer
// CHECK: ttg.warp_specialize{{.*}}requestedRegisters = array<i32: 232>
// CHECK: default
// CHECK: scf.for
// CHECK: nvws.producer_acquire
// CHECK: ttg.async_copy_global_to_local
// CHECK: ttg.async_copy_global_to_local
// CHECK: nvws.producer_commit
// CHECK: partition0
// CHECK: nvws.consumer_wait
// CHECK: ttg.local_load
// CHECK: ttg.local_load
// CHECK: nvws.consumer_release
// CHECK: tt.dot


#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_one_consumer(%ptrA: tensor<128x256x!tt.ptr<f16>, #blocked2>, %ptrB: tensor<256x128x!tt.ptr<f16>, #blocked1>, %row: tensor<1x256xi32, #blocked2>, %column: tensor<256x1xi32, #blocked1>, %inc: tensor<256x128xi32, #blocked1>, %store_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant {async_task_id = array<i32: 1>} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 255 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 127 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 1 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 0 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0, 1>} dense<0.000000e+00> : tensor<256x128xf16, #blocked1>
    %cst_1 = arith.constant {async_task_id = array<i32: 0, 1>} dense<0.000000e+00> : tensor<128x256xf16, #blocked2>
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 8 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 128 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1>} 256 : i32
    %cst_2 = arith.constant {async_task_id = array<i32: 0, 1>} dense<256> : tensor<128x256xi32, #blocked2>
    %51 = arith.addi %arg5, %c255_i32 {async_task_id = array<i32: 0, 1>} : i32
    %52 = arith.divsi %51, %c256_i32 {async_task_id = array<i32: 0, 1>} : i32
    %55:3 = scf.for %arg9 = %c0_i32 to %52 step %c1_i32 iter_args(%arg10 = %cst, %arg11 = %ptrA, %arg12 = %ptrB) -> (tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>)  : i32 {
      %74 = arith.muli %arg9, %c256_i32 {async_task_id = array<i32: 0>} : i32
      %75 = arith.subi %arg5, %74 {async_task_id = array<i32: 0>} : i32
      %76 = tt.splat %75 {async_task_id = array<i32: 0>} : i32 -> tensor<1x256xi32, #blocked2>
      %77 = arith.cmpi slt, %row, %76 {async_task_id = array<i32: 0>} : tensor<1x256xi32, #blocked2>
      %78 = tt.broadcast %77 {async_task_id = array<i32: 0>} : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2>
      %79 = tt.load %arg11, %78, %cst_1 {async_task_id = array<i32: 0>} : tensor<128x256x!tt.ptr<f16>, #blocked2>
      %80 = tt.splat %75 {async_task_id = array<i32: 0>} : i32 -> tensor<256x1xi32, #blocked1>
      %81 = arith.cmpi slt, %column, %80 {async_task_id = array<i32: 0>} : tensor<256x1xi32, #blocked1>
      %82 = tt.broadcast %81 {async_task_id = array<i32: 0>} : tensor<256x1xi1, #blocked1> -> tensor<256x128xi1, #blocked1>
      %83 = tt.load %arg12, %82, %cst_0 {async_task_id = array<i32: 0>} : tensor<256x128x!tt.ptr<f16>, #blocked1>
      // 2 loads in partition 0
      %84 = ttg.convert_layout %79 {async_task_id = array<i32: 1>} : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %85 = ttg.convert_layout %83 {async_task_id = array<i32: 1>} : tensor<256x128xf16, #blocked1> -> tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %86 = tt.dot %84, %85, %arg10, inputPrecision = tf32 {async_task_id = array<i32: 1>} : tensor<128x256xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<256x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
      %87 = tt.addptr %arg11, %cst_2 {async_task_id = array<i32: 0>} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
      %88 = tt.addptr %arg12, %inc {async_task_id = array<i32: 0>} : tensor<256x128x!tt.ptr<f16>, #blocked1>, tensor<256x128xi32, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1>} %86, %87, %88 : tensor<128x128xf32, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<256x128x!tt.ptr<f16>, #blocked1>
    } {async_task_id = array<i32: 0, 1>}
    %56 = arith.truncf %55#0 {async_task_id = array<i32: 1>} : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %73 = ttg.convert_layout %56 {async_task_id = array<i32: 1>} : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked1>
    tt.store %store_ptr, %73 {async_task_id = array<i32: 1>} : tensor<128x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----


// CHECK-LABEL: @matmul_kernel_two_consumers
// CHECK: ttg.warp_specialize{{.*}}requestedRegisters = array<i32: 232, 232>
// CHECK: default
// CHECK: scf.for
// CHECK: nvws.producer_acquire
// CHECK: ttg.async_copy_global_to_local
// CHECK: nvws.producer_commit
// CHECK: nvws.producer_acquire
// CHECK: nvws.producer_acquire
// CHECK: ttg.async_copy_global_to_local
// CHECK: nvws.producer_commit
// CHECK: nvws.producer_commit
// CHECK: partition0
// CHECK: scf.for
// CHECK: nvws.consumer_wait
// CHECK: nvws.consumer_wait
// CHECK: ttng.warp_group_dot
// CHECK: nvws.consumer_release
// CHECK: nvws.consumer_release
// CHECK: partition1
// CHECK: scf.for
// CHECK: nvws.consumer_wait
// CHECK: nvws.consumer_wait
// CHECK: ttng.warp_group_dot
// CHECK: nvws.consumer_release
// CHECK: nvws.consumer_release

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_two_consumers(%input_ptr1: tensor<64x64x!tt.ptr<f16>, #blocked>, %input_ptr2: tensor<64x128x!tt.ptr<f16>, #blocked1>, %input_ptr3: tensor<64x64x!tt.ptr<f16>, #blocked>, %row: tensor<1x64xi32, #blocked>, %column: tensor<64x1xi32, #blocked1>, %inc: tensor<64x128xi32, #blocked1>, %store_ptr1: tensor<64x128x!tt.ptr<f16>, #blocked1>, %store_ptr2: tensor<64x128x!tt.ptr<f16>, #blocked1>, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant {async_task_id = array<i32: 0>} dense<64> : tensor<64x64xi32, #blocked>
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c8_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 8 : i32
    %cst_0 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    %cst_1 = arith.constant {async_task_id = array<i32: 0>} dense<0.000000e+00> : tensor<64x128xf16, #blocked1>
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 127 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 63 : i32
    %cst_2 = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %58 = arith.addi %arg5, %c63_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %59 = arith.divsi %58, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %64:5 = scf.for %arg9 = %c0_i32 to %59 step %c1_i32 iter_args(%arg10 = %cst_2, %arg11 = %cst_2, %arg12 = %input_ptr1, %arg13 = %input_ptr2, %arg14 = %input_ptr3) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>)  : i32 {
      %93 = arith.muli %arg9, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %94 = arith.subi %arg5, %93 {async_task_id = array<i32: 0>} : i32
      %95 = tt.splat %94 {async_task_id = array<i32: 0>} : i32 -> tensor<1x64xi32, #blocked>
      %96 = arith.cmpi slt, %row, %95 {async_task_id = array<i32: 0>} : tensor<1x64xi32, #blocked>
      %97 = tt.broadcast %96 {async_task_id = array<i32: 0>} : tensor<1x64xi1, #blocked> -> tensor<64x64xi1, #blocked>
      %98 = tt.load %arg12, %97, %cst_0 {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %99 = ttg.local_alloc %98 {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %100 = tt.splat %94 {async_task_id = array<i32: 0>} : i32 -> tensor<64x1xi32, #blocked1>
      %101 = arith.cmpi slt, %column, %100 {async_task_id = array<i32: 0>} : tensor<64x1xi32, #blocked1>
      %102 = tt.broadcast %101 {async_task_id = array<i32: 0>} : tensor<64x1xi1, #blocked1> -> tensor<64x128xi1, #blocked1>
      %103 = tt.load %arg13, %102, %cst_1 {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %104 = ttg.local_alloc %103 {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %105 = tt.load %arg14, %97, %cst_0 {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %106 = ttg.local_alloc %105 {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %107 = ttng.warp_group_dot %99, %104, %arg10 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %108 = ttng.warp_group_dot %106, %104, %arg11 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %109 = tt.addptr %arg12, %cst {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %110 = tt.addptr %arg14, %cst {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x64xi32, #blocked>
      %111 = tt.addptr %arg13, %inc {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x128xi32, #blocked1>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %107, %108, %109, %111, %110 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>, tensor<64x64x!tt.ptr<f16>, #blocked>, tensor<64x128x!tt.ptr<f16>, #blocked1>, tensor<64x64x!tt.ptr<f16>, #blocked>
    } {async_task_id = array<i32: 0, 1, 2>}
    %65 = arith.truncf %64#0 {async_task_id = array<i32: 1>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %66 = arith.truncf %64#1 {async_task_id = array<i32: 2>} : tensor<64x128xf32, #mma> to tensor<64x128xf16, #mma>
    %91 = ttg.convert_layout %65 {async_task_id = array<i32: 1>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %store_ptr1, %91 {async_task_id = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    %92 = ttg.convert_layout %66 {async_task_id = array<i32: 2>} : tensor<64x128xf16, #mma> -> tensor<64x128xf16, #blocked1>
    tt.store %store_ptr2, %92 {async_task_id = array<i32: 2>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}


// -----

// CHECK-LABEL: @_matmul_layernorm_persistent_one_producer_one_consumer_one_epilog
// CHECK: ttg.warp_specialize{{.*}}requestedRegisters = array<i32: 232, 232>
// CHECK: default
// CHECK: scf.for
// CHECK: scf.for
// CHECK: nvws.producer_acquire
// CHECK: ttng.barrier_expect
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: partition0
// CHECK: scf.for
// CHECK: scf.for
// CHECK: ttng.wait_barrier
// CHECK: ttg.local_load
// CHECK: ttg.local_load
// CHECK: ttng.warp_group_dot
// CHECK: nvws.consumer_release
// CHECK: nvws.producer_acquire
// CHECK: ttg.local_store
// CHECK: nvws.producer_commit
// CHECK: partition1
// CHECK: scf.for
// CHECK: scf.for
// CHECK: nvws.consumer_wait
// CHECK: ttg.local_load
// CHECK: nvws.consumer_release
// CHECK: tt.descriptor_store

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_matmul_layernorm_persistent_one_producer_one_consumer_one_epilog(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x256xf16, #shared>, %arg2: !tt.tensordesc<128x256xf16, #shared>, %arg3: !tt.tensordesc<256xf16, #shared>, %arg4: !tt.tensordesc<256xf16, #shared>, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: f32) {
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 63 : i32
    %c128_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 128 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c132_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 132 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c127_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 127 : i32
    %c256_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 256 : i32
    %c255_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 255 : i32
    %cst = arith.constant {async_task_id = array<i32: 0, 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %cst_0 = arith.constant {async_task_id = array<i32: 2>} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = arith.addi %arg7, %c63_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = arith.divsi %0, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %2 = arith.addi %arg5, %c127_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %3 = arith.divsi %2, %c128_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %4 = arith.addi %arg6, %c255_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %5 = arith.divsi %4, %c256_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %6 = arith.muli %3, %5 {async_task_id = array<i32: 0, 1, 2>} : i32
    %7 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %8 = arith.sitofp %arg6 {async_task_id = array<i32: 2>} : i32 to f32
    %9 = tt.splat %8 {async_task_id = array<i32: 2>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %10 = tt.splat %arg11 {async_task_id = array<i32: 2>} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    scf.for %arg12 = %7 to %6 step %c132_i32  : i32 {
      %11 = arith.muli %arg12, %c128_i32 {async_task_id = array<i32: 0, 2>} : i32
      %true = arith.constant {async_task_id = array<i32: 0, 1, 2>} true
      %false = arith.constant {async_task_id = array<i32: 0, 1, 2>} false
      %12 = scf.for %arg13 = %c0_i32 to %1 step %c1_i32 iter_args(%arg14 = %cst) -> (tensor<128x256xf32, #mma>)  : i32 {
        %45 = arith.muli %arg13, %c64_i32 {async_task_id = array<i32: 0>} : i32
        %46 = tt.descriptor_load %arg0[%11, %45] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked>
        %47 = ttg.local_alloc %46 {async_task_id = array<i32: 1>} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory>
        %48 = tt.descriptor_load %arg1[%45, %c0_i32] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16, #shared> -> tensor<64x256xf16, #blocked1>
        %49 = ttg.local_alloc %48 {async_task_id = array<i32: 1>} : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory>
        %50 = ttng.warp_group_dot %47, %49, %arg14 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x256xf16, #shared, #ttg.shared_memory> -> tensor<128x256xf32, #mma>
        scf.yield {async_task_id = array<i32: 0, 1, 2>} %50 : tensor<128x256xf32, #mma>
      } {async_task_id = array<i32: 0, 1, 2>}
      %13 = "tt.reduce"(%12) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %45 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 2>} : f32
        tt.reduce.return %45 {async_task_id = array<i32: 2>} : f32
      }) {async_task_id = array<i32: 2>} : (tensor<128x256xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %14 = arith.divf %13, %9 {async_task_id = array<i32: 2>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %15 = tt.expand_dims %14 {async_task_id = array<i32: 2>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %16 = tt.broadcast %15 {async_task_id = array<i32: 2>} : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma>
      %17 = arith.subf %12, %16 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma>
      %18 = arith.mulf %17, %17 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma>
      %19 = "tt.reduce"(%18) <{axis = 1 : i32}> ({
      ^bb0(%arg13: f32, %arg14: f32):
        %45 = arith.addf %arg13, %arg14 {async_task_id = array<i32: 2>} : f32
        tt.reduce.return %45 {async_task_id = array<i32: 2>} : f32
      }) {async_task_id = array<i32: 2>} : (tensor<128x256xf32, #mma>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %20 = arith.divf %19, %9 {async_task_id = array<i32: 2>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %21 = arith.addf %20, %10 {async_task_id = array<i32: 2>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %22 = math.sqrt %21 {async_task_id = array<i32: 2>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %23 = arith.divf %cst_0, %22 {async_task_id = array<i32: 2>} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %24 = tt.descriptor_load %arg3[%c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<256xf16, #shared> -> tensor<256xf16, #blocked2>
      %25 = tt.descriptor_load %arg4[%c0_i32] {async_task_id = array<i32: 2>} : !tt.tensordesc<256xf16, #shared> -> tensor<256xf16, #blocked2>
      %26 = tt.expand_dims %23 {async_task_id = array<i32: 2>, axis = 1 : i32} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<128x1xf32, #mma>
      %27 = tt.broadcast %26 {async_task_id = array<i32: 2>} : tensor<128x1xf32, #mma> -> tensor<128x256xf32, #mma>
      %28 = arith.mulf %17, %27 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma>
      %29 = ttg.convert_layout %24 {async_task_id = array<i32: 2>} : tensor<256xf16, #blocked2> -> tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %30 = tt.expand_dims %29 {async_task_id = array<i32: 2>, axis = 0 : i32} : tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xf16, #blocked1>
      %31 = ttg.convert_layout %30 {async_task_id = array<i32: 2>} : tensor<1x256xf16, #blocked1> -> tensor<1x256xf16, #blocked3>
      %32 = arith.extf %31 {async_task_id = array<i32: 2>} : tensor<1x256xf16, #blocked3> to tensor<1x256xf32, #blocked3>
      %33 = ttg.convert_layout %32 {async_task_id = array<i32: 2>} : tensor<1x256xf32, #blocked3> -> tensor<1x256xf32, #mma>
      %34 = tt.broadcast %33 {async_task_id = array<i32: 2>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
      %35 = arith.mulf %28, %34 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma>
      %36 = ttg.convert_layout %25 {async_task_id = array<i32: 2>} : tensor<256xf16, #blocked2> -> tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked1}>>
      %37 = tt.expand_dims %36 {async_task_id = array<i32: 2>, axis = 0 : i32} : tensor<256xf16, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xf16, #blocked1>
      %38 = ttg.convert_layout %37 {async_task_id = array<i32: 2>} : tensor<1x256xf16, #blocked1> -> tensor<1x256xf16, #blocked3>
      %39 = arith.extf %38 {async_task_id = array<i32: 2>} : tensor<1x256xf16, #blocked3> to tensor<1x256xf32, #blocked3>
      %40 = ttg.convert_layout %39 {async_task_id = array<i32: 2>} : tensor<1x256xf32, #blocked3> -> tensor<1x256xf32, #mma>
      %41 = tt.broadcast %40 {async_task_id = array<i32: 2>} : tensor<1x256xf32, #mma> -> tensor<128x256xf32, #mma>
      %42 = arith.addf %35, %41 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma>
      %43 = arith.truncf %42 {async_task_id = array<i32: 2>} : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %44 = ttg.convert_layout %43 {async_task_id = array<i32: 2>} : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%11, %c0_i32], %44 {async_task_id = array<i32: 2>} : !tt.tensordesc<128x256xf16, #shared>, tensor<128x256xf16, #blocked1>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}


// -----

// CHECK-DAG: #[[$SHARED:.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
// CHECK-DAG: #[[$SHARED1:.*]]  = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
// CHECK-LABEL: @_fbgemm_grouped_gemm_fp8_rowwise_ws
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf8E4M3FN, #[[$SHARED1]], #smem, mutable>
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf8E4M3FN, #[[$SHARED1]], #smem, mutable>
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x128xf32, #[[$SHARED]], #smem, mutable>

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_fbgemm_grouped_gemm_fp8_rowwise_ws(%arg0: !tt.ptr<i8, "descriptor"> {tt.nv_tma_desc = 1 : i32}, %arg1: i32, %arg2: !tt.ptr<i8, "descriptor"> {tt.nv_tma_desc = 1 : i32}, %arg3: !tt.ptr<i8, "descriptor"> {tt.nv_tma_desc = 1 : i32}) {
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c2048_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 2048 : i32
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %cst = arith.constant {async_task_id = array<i32: 0, 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %0 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %1 = ttng.reinterpret_tensor_descriptor %arg0 {async_task_id = array<i32: 0>} : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<64x64xf8E4M3FN, #shared>
    %2 = ttng.reinterpret_tensor_descriptor %arg2 {async_task_id = array<i32: 0>} : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<128x64xf8E4M3FN, #shared>
    %3 = ttng.reinterpret_tensor_descriptor %arg3 {async_task_id = array<i32: 0>} : !tt.ptr<i8, "descriptor"> to !tt.tensordesc<128xf32, #shared1>
    scf.for %arg4 = %0 to %arg1 step %c64_i32  : i32 {
      %4 = arith.muli %arg4, %c2048_i32 {async_task_id = array<i32: 0>} : i32
      %5 = scf.for %arg5 = %c0_i32 to %c2048_i32 step %c64_i32 iter_args(%arg6 = %cst) -> (tensor<64x128xf32, #mma>)  : i32 {
        %8 = tt.descriptor_load %1[%4, %arg5] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
        %9 = ttg.local_alloc %8 {async_task_id = array<i32: 1>} : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
        %10 = tt.descriptor_load %2[%4, %arg5] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf8E4M3FN, #shared> -> tensor<128x64xf8E4M3FN, #blocked>
        %11 = ttg.local_alloc %10 {async_task_id = array<i32: 1, 2>} : (tensor<128x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem>
        %12 = ttg.memdesc_trans %11 {async_task_id = array<i32: 1, 2>, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem>
        %13 = ttng.warp_group_dot %9, %12, %arg6 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem> -> tensor<64x128xf32, #mma>
        scf.yield {async_task_id = array<i32: 1, 2>} %13 : tensor<64x128xf32, #mma>
      } {async_task_id = array<i32: 0, 1, 2>}
      %6 = tt.descriptor_load %3[%4] {async_task_id = array<i32: 0>} : !tt.tensordesc<128xf32, #shared1> -> tensor<128xf32, #blocked1>
      %7 = ttg.convert_layout %6 {async_task_id = array<i32: 1, 2>} : tensor<128xf32, #blocked1> -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    } {async_task_id = array<i32: 1, 2>}
    tt.return
  }
}

// -----

// An s8 warp_group_dot takes operand B in a transposed shared layout, which a
// TMA descriptor cannot describe. The buffer created for B must keep the
// transposed encoding its consumer needs instead of one re-derived from the
// producer's register order, and is therefore filled through registers
// (descriptor_load + local_store) while operand A is still filled by TMA.

// CHECK-DAG: #[[$A:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
// CHECK-DAG: #[[$B:.+]] = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
// CHECK-LABEL: @int8_dot_with_transposed_operand_b
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x64x32xi8, #[[$A]], #smem, mutable>
// CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x32x32xi8, #[[$B]], #smem, mutable>
// CHECK: ttng.async_tma_copy_global_to_local {{.*}} -> !ttg.memdesc<64x32xi8, #[[$A]], #smem, mutable>
// CHECK: tt.descriptor_load
// CHECK: ttg.local_store {{.*}} -> !ttg.memdesc<32x32xi8, #[[$B]], #smem, mutable>
// CHECK: ttng.warp_group_dot {{.*}}#[[$A]]{{.*}} * {{.*}}#[[$B]]

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @int8_dot_with_transposed_operand_b(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 32 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 31 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0> : tensor<64x32xi32, #mma>
    %0 = arith.extsi %arg5 {async_task_id = array<i32: 0>} : i32 to i64
    %1 = tt.make_tensor_descriptor %arg1, [%arg3, %arg5], [%0, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <64x32xsi8, #shared>
    %2 = arith.extsi %arg4 {async_task_id = array<i32: 0, 1, 2>} : i32 to i64
    %3 = tt.make_tensor_descriptor %arg2, [%arg5, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <32x32xsi8, #shared>
    %4 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 1, 2>} : <i32>, <64x32xsi32, #shared2>
    %5 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %6 = arith.muli %5, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %7 = tt.get_program_id y {async_task_id = array<i32: 0, 1, 2>} : i32
    %8 = arith.muli %7, %c32_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %9 = arith.addi %arg5, %c31_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %10 = arith.divsi %9, %c32_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %11 = scf.for %arg6 = %c0_i32 to %10 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xi32, #mma>)  : i32 {
      %13 = arith.muli %arg6, %c32_i32 {async_task_id = array<i32: 0>} : i32
      %14 = tt.descriptor_load %1[%6, %13] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x32xsi8, #shared> -> tensor<64x32xi8, #blocked1>
      %15 = ttg.local_alloc %14 {async_task_id = array<i32: 1, 2>} : (tensor<64x32xi8, #blocked1>) -> !ttg.memdesc<64x32xi8, #shared, #smem>
      %16 = tt.descriptor_load %3[%13, %8] {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xsi8, #shared> -> tensor<32x32xi8, #blocked>
      %17 = ttg.local_alloc %16 {async_task_id = array<i32: 1, 2>} : (tensor<32x32xi8, #blocked>) -> !ttg.memdesc<32x32xi8, #shared1, #smem>
      %18 = ttng.warp_group_dot %15, %17, %arg7 {async_task_id = array<i32: 1, 2>} : !ttg.memdesc<64x32xi8, #shared, #smem> * !ttg.memdesc<32x32xi8, #shared1, #smem> -> tensor<64x32xi32, #mma>
      scf.yield {async_task_id = array<i32: 1, 2>} %18 : tensor<64x32xi32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>, tt.num_stages = 3 : i32, tt.warp_specialize}
    %12 = ttg.convert_layout %11 {async_task_id = array<i32: 1, 2>} : tensor<64x32xi32, #mma> -> tensor<64x32xi32, #blocked2>
    tt.descriptor_store %4[%6, %8], %12 {async_task_id = array<i32: 1, 2>} : !tt.tensordesc<64x32xsi32, #shared2>, tensor<64x32xi32, #blocked2>
    tt.return
  }
}

// -----

// A register-staged channel needs its own completion edge. Operand A is filled
// directly by TMA and its fill is signalled by the copy's transaction count,
// but operand B is staged through registers, so nothing signals that its buffer
// is full unless the channel carries a producer_commit with a matching
// consumer_wait. Without them the consumer's warp_group_dot reads B while the
// producer is still storing into it.

// CHECK-LABEL: @int8_staged_operand_b_token_protocol
// The staged channel must be typed as a local-store producer, not as TMA.
// Two consumer task ids, so each group must produce exactly two tokens: the
// TMA group for A first, then the register-staged group for B.
// CHECK-COUNT-2: nvws.create_token {loadType = 2 : i32
// CHECK-COUNT-2: nvws.create_token {loadType = 3 : i32
// Producer: A keeps the direct TMA fill, B is staged and then committed.
// CHECK: nvws.producer_acquire
// CHECK: ttng.async_tma_copy_global_to_local
// CHECK: ttg.local_store
// The staged stores go through the generic proxy and warp_group_dot reads
// through the async proxy, so the fence must publish them before the arrival.
// CHECK: ttng.fence_async_shared
// CHECK: nvws.producer_commit
// Consumer: the dot must not run before the staged buffer is known to be full.
// CHECK: nvws.consumer_wait
// CHECK: ttng.warp_group_dot
// CHECK: nvws.consumer_release

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @int8_staged_operand_b_token_protocol(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c32_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 32 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c31_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 31 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0> : tensor<64x32xi32, #mma>
    %0 = arith.extsi %arg5 {async_task_id = array<i32: 0>} : i32 to i64
    %1 = tt.make_tensor_descriptor %arg1, [%arg3, %arg5], [%0, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <64x32xsi8, #shared>
    %2 = arith.extsi %arg4 {async_task_id = array<i32: 0, 1, 2>} : i32 to i64
    %3 = tt.make_tensor_descriptor %arg2, [%arg5, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <32x32xsi8, #shared>
    %4 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 1, 2>} : <i32>, <64x32xsi32, #shared2>
    %5 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %6 = arith.muli %5, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %7 = tt.get_program_id y {async_task_id = array<i32: 0, 1, 2>} : i32
    %8 = arith.muli %7, %c32_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %9 = arith.addi %arg5, %c31_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %10 = arith.divsi %9, %c32_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %11 = scf.for %arg6 = %c0_i32 to %10 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<64x32xi32, #mma>)  : i32 {
      %13 = arith.muli %arg6, %c32_i32 {async_task_id = array<i32: 0>} : i32
      %14 = tt.descriptor_load %1[%6, %13] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x32xsi8, #shared> -> tensor<64x32xi8, #blocked1>
      %15 = ttg.local_alloc %14 {async_task_id = array<i32: 1, 2>} : (tensor<64x32xi8, #blocked1>) -> !ttg.memdesc<64x32xi8, #shared, #smem>
      %16 = tt.descriptor_load %3[%13, %8] {async_task_id = array<i32: 0>} : !tt.tensordesc<32x32xsi8, #shared> -> tensor<32x32xi8, #blocked>
      %17 = ttg.local_alloc %16 {async_task_id = array<i32: 1, 2>} : (tensor<32x32xi8, #blocked>) -> !ttg.memdesc<32x32xi8, #shared1, #smem>
      %18 = ttng.warp_group_dot %15, %17, %arg7 {async_task_id = array<i32: 1, 2>} : !ttg.memdesc<64x32xi8, #shared, #smem> * !ttg.memdesc<32x32xi8, #shared1, #smem> -> tensor<64x32xi32, #mma>
      scf.yield {async_task_id = array<i32: 1, 2>} %18 : tensor<64x32xi32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>, tt.num_stages = 3 : i32, tt.warp_specialize}
    %12 = ttg.convert_layout %11 {async_task_id = array<i32: 1, 2>} : tensor<64x32xi32, #mma> -> tensor<64x32xi32, #blocked2>
    tt.descriptor_store %4[%6, %8], %12 {async_task_id = array<i32: 1, 2>} : !tt.tensordesc<64x32xsi32, #shared2>, tensor<64x32xi32, #blocked2>
    tt.return
  }
}

// -----

// One descriptor load feeds two WGMMA consumers in opposite operand roles.
// The non-specialized path gives those consumers separate buffers, one for
// each NVMMAShared encoding. Code partitioning cannot merge them into one
// producer buffer until it supports consumer-specific buffers and copies.

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#out_blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared_transposed = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#out_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = false} {
  tt.func public @two_dot_roles_require_different_encodings(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i8>, %arg3: i32, %arg4: i32, %arg5: i32) attributes {noinline = false} {
    %c1_i64 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i64
    %c64_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 64 : i32
    %c0_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %c63_i32 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 63 : i32
    %cst = arith.constant {async_task_id = array<i32: 1, 2>} dense<0> : tensor<64x64xi32, #mma>
    %0 = arith.extsi %arg5 {async_task_id = array<i32: 0>} : i32 to i64
    %1 = tt.make_tensor_descriptor %arg1, [%arg3, %arg5], [%0, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <64x64xsi8, #shared>
    %2 = arith.extsi %arg4 {async_task_id = array<i32: 0, 1, 2>} : i32 to i64
    %3 = tt.make_tensor_descriptor %arg2, [%arg5, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 0>} : <i8>, <64x64xsi8, #shared>
    %4 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%2, %c1_i64] {async_task_id = array<i32: 1, 2>} : <i32>, <64x64xsi32, #out_shared>
    %5 = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
    %6 = arith.muli %5, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %7 = tt.get_program_id y {async_task_id = array<i32: 0, 1, 2>} : i32
    %8 = arith.muli %7, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %9 = arith.addi %arg5, %c63_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %10 = arith.divsi %9, %c64_i32 {async_task_id = array<i32: 0, 1, 2>} : i32
    %11 = scf.for %arg6 = %c0_i32 to %10 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<64x64xi32, #mma>)  : i32 {
      %13 = arith.muli %arg6, %c64_i32 {async_task_id = array<i32: 0>} : i32
      %14 = tt.descriptor_load %1[%6, %13] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xsi8, #shared> -> tensor<64x64xi8, #blocked>
      // expected-error @+1 {{warp specialization cannot share a producer buffer between MMA consumers requiring different NVMMAShared encodings}}
      %15 = ttg.local_alloc %14 {async_task_id = array<i32: 1, 2>} : (tensor<64x64xi8, #blocked>) -> !ttg.memdesc<64x64xi8, #shared, #smem>
      %16 = ttg.local_alloc %14 {async_task_id = array<i32: 1, 2>} : (tensor<64x64xi8, #blocked>) -> !ttg.memdesc<64x64xi8, #shared_transposed, #smem>
      %17 = tt.descriptor_load %3[%13, %8] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xsi8, #shared> -> tensor<64x64xi8, #blocked>
      %18 = ttg.local_alloc %17 {async_task_id = array<i32: 1, 2>} : (tensor<64x64xi8, #blocked>) -> !ttg.memdesc<64x64xi8, #shared_transposed, #smem>
      %19 = ttg.local_alloc %17 {async_task_id = array<i32: 1, 2>} : (tensor<64x64xi8, #blocked>) -> !ttg.memdesc<64x64xi8, #shared, #smem>
      %20 = ttng.warp_group_dot %15, %18, %arg7 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xi8, #shared, #smem> * !ttg.memdesc<64x64xi8, #shared_transposed, #smem> -> tensor<64x64xi32, #mma>
      %21 = ttng.warp_group_dot %19, %16, %20 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xi8, #shared, #smem> * !ttg.memdesc<64x64xi8, #shared_transposed, #smem> -> tensor<64x64xi32, #mma>
      scf.yield {async_task_id = array<i32: 1, 2>} %21 : tensor<64x64xi32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>, tt.num_stages = 3 : i32, tt.warp_specialize}
    %12 = ttg.convert_layout %11 {async_task_id = array<i32: 1, 2>} : tensor<64x64xi32, #mma> -> tensor<64x64xi32, #out_blocked>
    tt.descriptor_store %4[%6, %8], %12 {async_task_id = array<i32: 1, 2>} : !tt.tensordesc<64x64xsi32, #out_shared>, tensor<64x64xi32, #out_blocked>
    tt.return
  }
}
