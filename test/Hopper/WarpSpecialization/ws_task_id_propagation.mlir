// RUN: triton-opt %s -split-input-file --nvgpu-test-taskid-propagate=num-warp-groups=2 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

  // CHECK-LABEL: @matmul_persistent_tma_ws_cooperative_kernel
  // CHECK:       %[[C0:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
  // CHECK-NEXT:  %[[C1:.*]] = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
  // CHECK-NEXT:  %[[C64:.*]] = arith.constant {async_task_id = array<i32: 0>} 64 : i32
  // CHECK-NEXT:  %[[INIT:.*]] = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<128x256xf32, #mma>
  // CHECK-NEXT:  %[[PID:.*]] = tt.get_program_id x {async_task_id = array<i32: 0, 1, 2>} : i32
  // CHECK-NEXT:  %[[NUM:.*]] = tt.get_num_programs x {async_task_id = array<i32: 0, 1, 2>} : i32
  // CHECK-NEXT:  scf.for %[[IV:.*]] = %[[PID]] to %[[UB:.*]] step %[[NUM]]  : i32 {
  // CHECK-NEXT:    %[[FOR:.*]]:2 = scf.for %{{.*}} = %[[C0]] to %{{.*}} step %[[C1]] iter_args(%[[ACC:.*]] = %[[INIT]], %[[OFF:.*]] = %[[C0]])
  // CHECK-NEXT:      %[[LOAD1:.*]] = tt.descriptor_load %[[INPUT1:.*]][%[[IV]], %[[OFF]]] {async_task_id = array<i32: 0>}
  // CHECK-NEXT:      %[[ALLOC1:.*]] = ttg.local_alloc %[[LOAD1]] {async_task_id = array<i32: 1, 2>}
  // CHECK-NEXT:      %[[LOAD2:.*]] = tt.descriptor_load %[[INPUT2:.*]][%[[OFF]], %[[IV]]] {async_task_id = array<i32: 0>}
  // CHECK-NEXT:      %[[ALLOC2:.*]] = ttg.local_alloc %[[LOAD2]] {async_task_id = array<i32: 1, 2>}
  // CHECK-NEXT:      %[[DOT:.*]] = ttng.warp_group_dot %[[ALLOC1]], %[[ALLOC2]], %[[ACC]] {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32}
  // CHECK-NEXT:      %[[ADD:.*]] = arith.addi %[[OFF]], %[[C64]] {async_task_id = array<i32: 0>}
  // CHECK-NEXT:      scf.yield {async_task_id = array<i32: 0, 1, 2>} %[[DOT]], %[[ADD]]
  // CHECK-NEXT:    } {async_task_id = array<i32: 0, 1, 2>}
  // CHECK-NEXT:    arith.truncf %[[FOR]]#0 {async_task_id = array<i32: 1, 2>}
  // CHECK-NEXT:    ttg.convert_layout %{{.*}} {async_task_id = array<i32: 1, 2>}
  // CHECK-NEXT:    tt.descriptor_store %[[OUTPUT:.*]][%[[IV]], %[[IV]]], %{{.*}} {async_task_id = array<i32: 1, 2>}
  // CHECK-NEXT:  } {async_task_id = array<i32: 0, 1, 2>}

  tt.func public @matmul_persistent_tma_ws_cooperative_kernel(%arg0: !tt.tensordesc<128x64xf16>, %arg1: !tt.tensordesc<64x256xf16>, %arg2: !tt.tensordesc<128x256xf16>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    scf.for %arg6 = %0 to %arg3 step %1  : i32 {
      %2:2 = scf.for %arg7 = %c0_i32 to %arg5 step %c1_i32 iter_args(%arg8 = %cst, %arg9 = %c0_i32) -> (tensor<128x256xf32, #mma>, i32)  : i32 {
        %5 = tt.descriptor_load %arg0[%arg6, %arg9] {async_task_id = array<i32: 0>} : !tt.tensordesc<128x64xf16> -> tensor<128x64xf16, #blocked>
        %6 = ttg.local_alloc %5 : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        %7 = tt.descriptor_load %arg1[%arg9, %arg6] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x256xf16> -> tensor<64x256xf16, #blocked1>
        %8 = ttg.local_alloc %7 : (tensor<64x256xf16, #blocked1>) -> !ttg.memdesc<64x256xf16, #shared, #smem>
        %9 = ttng.warp_group_dot %6, %8, %arg8 {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x256xf16, #shared, #smem> -> tensor<128x256xf32, #mma>
        %10 = arith.addi %arg9, %c64_i32 : i32
        scf.yield %9, %10 : tensor<128x256xf32, #mma>, i32
      }
      %3 = arith.truncf %2#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
      %4 = ttg.convert_layout %3 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked1>
      tt.descriptor_store %arg2[%arg6, %arg6], %4 {async_task_id = array<i32: 1, 2>} : !tt.tensordesc<128x256xf16>, tensor<128x256xf16, #blocked1>
    }
    tt.return
  }
}

// -----

// Pre-annotated anchor ops inside a loop must still label their enclosing loop
// even when no unannotated producer ops exist inside the loop body.
// CHECK-LABEL: @anchor_op_labels_parent_loop
// CHECK: scf.for
// CHECK:   tt.descriptor_load {{.*}} {async_task_id = array<i32: 0>}
// CHECK:   ttng.warp_group_dot {{.*}} {async_task_id = array<i32: 1>
// CHECK: } {async_task_id = array<i32: 0, 1>}
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @anchor_op_labels_parent_loop(%arg0: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg1: !tt.tensordesc<64x64xf8E4M3FN, #shared>, %arg2: !tt.tensordesc<64x64xf32, #shared>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c3 = arith.constant 3 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c3 step %c1 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.descriptor_load %arg0[%c0, %iv] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %b = tt.descriptor_load %arg1[%iv, %c0] {async_task_id = array<i32: 0>} : !tt.tensordesc<64x64xf8E4M3FN, #shared> -> tensor<64x64xf8E4M3FN, #blocked>
      %la = ttg.local_alloc %a : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb0 = ttg.local_alloc %b : (tensor<64x64xf8E4M3FN, #blocked>) -> !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>
      %lb = ttg.memdesc_trans %lb0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem>
      %dot = ttng.warp_group_dot %la, %lb, %acc {async_task_id = array<i32: 1>, inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem> * !ttg.memdesc<64x64xf8E4M3FN, #shared1, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    %out = ttg.convert_layout %res : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    tt.descriptor_store %arg2[%c0, %c0], %out {async_task_id = array<i32: 1>} : !tt.tensordesc<64x64xf32, #shared>, tensor<64x64xf32, #blocked>
    tt.return
  }
}
