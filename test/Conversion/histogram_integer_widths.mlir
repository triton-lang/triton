// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --dump-input-context 20

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @histogram_i8
  // CHECK: llvm.zext {{.*}} : i8 to i32
  // CHECK: llvm.icmp "ult"
  // CHECK: llvm.atomicrmw add
  tt.func @histogram_i8(%src: tensor<256xi8, #blocked>, %mask: tensor<256xi1, #blocked>, %out_ptr: tensor<8x!tt.ptr<i32>, #blocked>) {
    %hist = tt.histogram %src, %mask : tensor<256xi8, #blocked> -> tensor<8xi32, #blocked>
    tt.store %out_ptr, %hist : tensor<8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @histogram_i16
  // CHECK: llvm.zext {{.*}} : i16 to i32
  // CHECK: llvm.icmp "ult"
  // CHECK: llvm.atomicrmw add
  tt.func @histogram_i16(%src: tensor<256xi16, #blocked>, %mask: tensor<256xi1, #blocked>, %out_ptr: tensor<8x!tt.ptr<i32>, #blocked>) {
    %hist = tt.histogram %src, %mask : tensor<256xi16, #blocked> -> tensor<8xi32, #blocked>
    tt.store %out_ptr, %hist : tensor<8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @histogram_i64
  // CHECK: llvm.icmp "ult"
  // CHECK: llvm.trunc {{.*}} : i64 to i32
  // CHECK: llvm.atomicrmw add
  tt.func @histogram_i64(%src: tensor<256xi64, #blocked>, %mask: tensor<256xi1, #blocked>, %out_ptr: tensor<8x!tt.ptr<i32>, #blocked>) {
    %hist = tt.histogram %src, %mask : tensor<256xi64, #blocked> -> tensor<8xi32, #blocked>
    tt.store %out_ptr, %hist : tensor<8x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
