// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: buffer_atomic_cas_i64
  tt.func public @buffer_atomic_cas_i64(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    // CHECK: %[[cas_val:.*]] = llvm.mlir.constant(2 : i64) : i64
    // CHECK: %[[cas_val_cast:.*]] = llvm.bitcast %[[cas_val]] : i64 to i64
    // CHECK: %[[cas_val_insert:.*]] = llvm.insertvalue %[[cas_val_cast]], %{{.*}}[1] : !llvm.struct<(i64, i64)>
    %val = arith.constant dense<2> : tensor<512xi64, #blocked>

    // CHECK: %[[cas_cmp:.*]] = llvm.mlir.constant(0 : i64) : i64
    // CHECK: %[[cas_cmp_cast:.*]] = llvm.bitcast %[[cas_cmp]] : i64 to i64
    // CHECK: %[[cas_cmp_insert:.*]] = llvm.insertvalue %[[cas_cmp_cast]], %{{.*}}[1] : !llvm.struct<(i64, i64)>
    %cmp = arith.constant dense<0> : tensor<512xi64, #blocked>

    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %offsets = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %scalar_ptr = tt.addptr %arg0, %1 : !tt.ptr<i64>, i32

    // CHECK: %[[cas_val_extract:.*]] = llvm.extractvalue %[[cas_val_insert]][0] : !llvm.struct<(i64, i64)>
    // CHECK: %[[cas_cmp_extract:.*]] = llvm.extractvalue %[[cas_cmp_insert]][0] : !llvm.struct<(i64, i64)>
    // CHECK: %[[resource:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK: llvm.fence syncscope("agent") release
    // CHECK: %[[cas_val_insert2:.*]] = llvm.insertelement %[[cas_val_extract]], %{{.*}} : vector<1xi64>
    // CHECK: %[[cas_cmp_insert2:.*]] = llvm.insertelement %[[cas_cmp_extract]], %{{.*}} : vector<1xi64>
    // CHECK: %[[cas_val_cast2:.*]] = llvm.bitcast %[[cas_val_insert2]] : vector<1xi64> to i64
    // CHECK: %[[cas_cmp_cast2:.*]] = llvm.bitcast %[[cas_cmp_insert2]] : vector<1xi64> to i64
    // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %[[cas_val_cast2]], %[[cas_cmp_cast2]], %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i64
    // CHECK: %[[dst:.*]] = rocdl.raw.ptr.buffer.atomic.cmpswap %{{.*}}, %{{.*}}, %[[resource]], %{{.*}}, %{{.*}}, %{{.*}} : i64
    // CHECK: llvm.fence syncscope("agent") acquire
    %4 = amdgpu.buffer_atomic_cas acq_rel, gpu, %cmp, %val, %scalar_ptr[%offsets] : tensor<512xi64, #blocked>

    %5 = tt.addptr %arg1, %1 : !tt.ptr<i64>, i32
    amdgpu.buffer_store %4, %5[%offsets] : tensor<512xi64, #blocked>
    tt.return
  }
}
