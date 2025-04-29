// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942' | FileCheck %s

// CHECK-LABEL: @exp_kernel
// CHECK-DAG: [[VALUE_1:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// CHECK-DAG: [[VALUE_2:%.*]] = math.exp2 [[VALUE_1]]
// CHECK-DAG: [[VALUE_3:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// CHECK-DAG: [[VALUE_4:%.*]] = math.exp2 [[VALUE_3]]
// CHECK-DAG: [[VALUE_5:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// CHECK-DAG: [[VALUE_6:%.*]] = math.exp2 [[VALUE_5]]
// CHECK-DAG: [[VALUE_7:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// CHECK-DAG: [[VALUE_8:%.*]] = math.exp2 [[VALUE_7]]
// CHECK-DAG: [[VALUE_9:%.*]] = amdgpu.concat [[VALUE_2]], [[VALUE_4]], [[VALUE_6]], [[VALUE_8]]
// CHECK-DAG: tt.return [[VALUE_9]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @exp_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: mul_kernel
// CHECK-DAG: [[VALUE_1:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// CHECK-DAG: [[VALUE_2:%.*]] = amdgpu.extract_slice {{.*}} [0, 0]
// CHECK-DAG: [[VALUE_3:%.*]] = arith.mulf [[VALUE_1]], [[VALUE_2]]
// CHECK-DAG: [[VALUE_4:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// CHECK-DAG: [[VALUE_5:%.*]] = amdgpu.extract_slice {{.*}} [0, 16]
// CHECK-DAG: [[VALUE_6:%.*]] = arith.mulf [[VALUE_4]], [[VALUE_5]]
// CHECK-DAG: [[VALUE_7:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// CHECK-DAG: [[VALUE_8:%.*]] = amdgpu.extract_slice {{.*}} [64, 0]
// CHECK-DAG: [[VALUE_9:%.*]] = arith.mulf [[VALUE_7]], [[VALUE_8]]
// CHECK-DAG: [[VALUE_10:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// CHECK-DAG: [[VALUE_11:%.*]] = amdgpu.extract_slice {{.*}} [64, 16]
// CHECK-DAG: [[VALUE_12:%.*]] = arith.mulf [[VALUE_10]], [[VALUE_11]]
// CHECK-DAG: [[VALUE_13:%.*]] = amdgpu.concat [[VALUE_3]], [[VALUE_6]], [[VALUE_9]], [[VALUE_12]]
// CHECK-DAG: tt.return [[VALUE_13]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mul_kernel(%arg0: tensor<128x32xf32, #blocked>, %arg1: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: @multiple_operations_kernel

// CHECK-COUNT-4: amdgpu.extract_slice {{.*}}
// CHECK: [[OP1:%.*]] = amdgpu.concat
// CHECK-COUNT-4: amdgpu.extract_slice [[OP1]]
// CHECK: [[OP2:%.*]] = amdgpu.concat
// CHECK-COUNT-4: amdgpu.extract_slice [[OP2]]
// CHECK: [[OP3:%.*]] = amdgpu.concat
// CHECK: tt.return [[OP3]]
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_operations_kernel(%arg0: tensor<128x32xf32, #mma>, %arg1: tensor<128x32xf32, #mma>) -> tensor<128x32xf32, #mma> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #mma>
    %1 = math.exp2 %0 : tensor<128x32xf32, #mma>
    %2 = math.exp2 %1 : tensor<128x32xf32, #mma>
    tt.return %2 : tensor<128x32xf32, #mma>
  }
}

// -----

// CHECK-LABEL: @nested_operations_kernel
// CHECK-COUNT-8: amdgpu.extract_slice
// CHECK: mulf
// CHECK: amdgpu.concat
// CHECK: scf.for
// CHECK-COUNT-4: amdgpu.extract_slice
// CHECK: math.exp2
// CHECK: amdgpu.concat
// CHECK: }
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @nested_operations_kernel(%arg0: tensor<128x32xf32, #blocked>, %arg1: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = arith.mulf %arg0, %arg1 : tensor<128x32xf32, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %1 = scf.for %arg2 = %c0 to %c4 step %c1 iter_args(%arg3 = %0) -> (tensor<128x32xf32, #blocked>) : i32 {
      %2 = math.exp2 %0 : tensor<128x32xf32, #blocked>
      scf.yield %2 : tensor<128x32xf32, #blocked>
    }
    tt.return %1 : tensor<128x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: @peer_operations_kernel
// CHECK: scf.for
// CHECK-COUNT-4: amdgpu.extract_slice
// CHECK: math.exp2
// CHECK: amdgpu.concat
// CHECK: scf.for
// CHECK-NOT: amdgpu.extract_slice
// CHECK: math.exp2
// CHECK-NOT: amdgpu.concat
// CHECK: }
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @peer_operations_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %1 = scf.for %arg1 = %c0 to %c4 step %c1 iter_args(%arg2 = %arg0) -> (tensor<128x32xf32, #blocked>) : i32 {
      amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
      %2 = math.exp2 %arg2 : tensor<128x32xf32, #blocked>
      scf.yield %2 : tensor<128x32xf32, #blocked>
    }
    %3 = scf.for %arg3 = %c0 to %c4 step %c1 iter_args(%arg4 = %1) -> (tensor<128x32xf32, #blocked>) : i32 {
      %4 = math.exp2 %arg4 : tensor<128x32xf32, #blocked>
      scf.yield %4 : tensor<128x32xf32, #blocked>
    }
    tt.return %3 : tensor<128x32xf32, #blocked>
  }
}
