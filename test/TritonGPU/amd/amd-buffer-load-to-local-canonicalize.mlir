// RUN: triton-opt %s --canonicalize | FileCheck %s
// RUN: triton-opt %s --canonicalize --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx950 | FileCheck %s --check-prefix=LLVM

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @drop_zero_other(
  // LLVM-LABEL: llvm.func @drop_zero_other(
  // LLVM-NOT: llvm.cond_br
  // LLVM: rocdl.raw.ptr.buffer.load.async.lds
  // LLVM-NOT: llvm.cond_br
  // LLVM-NOT: llvm.store
  tt.func @drop_zero_other(%ptr: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>,
                           %mask: tensor<64xi1, #blocked>,
                           %dest: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: arith.constant dense<0.000000e+00>
    // CHECK: amdg.buffer_load_to_local %{{.*}}[%{{.*}}] mask = %{{.*}} into %{{.*}}
    // CHECK-NOT: other
    %zero = arith.constant dense<0.000000e+00> : tensor<64xf32, #blocked>
    %token = amdg.buffer_load_to_local %ptr[%offsets] mask = %mask other = %zero into %dest : !tt.ptr<f32>[tensor<64xi32, #blocked>] tensor<64xf32, #blocked> -> <64xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: tt.func @keep_nonzero_other(
  // LLVM-LABEL: llvm.func @keep_nonzero_other(
  // LLVM: llvm.cond_br
  // LLVM: rocdl.raw.ptr.buffer.load.async.lds
  // LLVM: llvm.store
  tt.func @keep_nonzero_other(%ptr: !tt.ptr<f32>, %offsets: tensor<64xi32, #blocked>,
                              %mask: tensor<64xi1, #blocked>,
                              %dest: !ttg.memdesc<64xf32, #shared, #smem, mutable>) {
    // CHECK: %[[ONE:.*]] = arith.constant dense<1.000000e+00>
    // CHECK: amdg.buffer_load_to_local %{{.*}}[%{{.*}}] mask = %{{.*}} other = %[[ONE]] into %{{.*}}
    %one = arith.constant dense<1.000000e+00> : tensor<64xf32, #blocked>
    %token = amdg.buffer_load_to_local %ptr[%offsets] mask = %mask other = %one into %dest : !tt.ptr<f32>[tensor<64xi32, #blocked>] tensor<64xf32, #blocked> -> <64xf32, #shared, #smem, mutable>
    tt.return
  }
}
