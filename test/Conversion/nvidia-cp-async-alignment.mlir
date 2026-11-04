// RUN: split-file %s %t
// RUN: triton-opt %t/valid.mlir --allocate-shared-memory-nv='compute-capability=80' --convert-triton-gpu-to-llvm='compute-capability=80' --convert-nv-gpu-to-llvm -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir | opt -O2 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx80 | FileCheck %t/valid.mlir
// RUN: triton-opt -split-input-file %t/invalid.mlir --allocate-shared-memory-nv --convert-triton-gpu-to-llvm='compute-capability=80' -verify-diagnostics

//--- valid.mlir

#pair = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#align4 = #ttg.shared_linear<{offset = [[1], [2], [4], [8], [16], [32]], block = []}, alignment = 4>
#padded = #ttg.padded_shared<[64:+1] {offset = [[1], [2], [4], [8], [16], [32], [64]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:80"} {
  // CHECK-LABEL: .entry cp_async_weak_alignment
  // CHECK-COUNT-2: cp.async.ca.shared.global {{.*}}, 0x4, 0x4;
  tt.func @cp_async_weak_alignment(%src: tensor<64x!tt.ptr<i32>, #pair> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<2> : tensor<1xi32>}, %dst: !ttg.memdesc<64xi32, #align4, #smem, mutable>) {
    %token = ttg.async_copy_global_to_local %src, %dst : tensor<64x!tt.ptr<i32>, #pair> -> !ttg.memdesc<64xi32, #align4, #smem, mutable>
    tt.return
  }

  // The slice starts at (64 + 1) * 4 bytes.
  // CHECK-LABEL: .entry cp_async_padded_slice
  // CHECK: add.s32 [[DST0:%r[0-9]+]], {{.*}}, 260;
  // CHECK: cp.async.ca.shared.global [ [[DST0]] + 0 ], {{.*}}, 0x4, 0x4;
  // CHECK: add.s32 [[DST1:%r[0-9]+]], {{.*}}, 264;
  // CHECK: cp.async.ca.shared.global [ [[DST1]] + 0 ], {{.*}}, 0x4, 0x4;
  tt.func @cp_async_padded_slice(%src: tensor<64x!tt.ptr<i32>, #pair> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<2> : tensor<1xi32>}) {
    %base = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #padded, #smem, mutable>
    %dst = ttg.memdesc_subslice %base[64] : !ttg.memdesc<128xi32, #padded, #smem, mutable> -> !ttg.memdesc<64xi32, #padded, #smem, mutable, 128>
    %token = ttg.async_copy_global_to_local %src, %dst : tensor<64x!tt.ptr<i32>, #pair> -> !ttg.memdesc<64xi32, #padded, #smem, mutable, 128>
    tt.return
  }

}

//--- invalid.mlir
#pair = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#padded = #ttg.padded_shared<[64:+1] {offset = [[1], [2], [4], [8], [16], [32]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func private @cp_async_underaligned_page(%src: tensor<64x!tt.ptr<f16>, #pair> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<2> : tensor<1xi32>}, %dst: !ttg.memdesc<64xf16, #padded, #ttg.shared_memory, mutable>) {
    // expected-error @+2 {{cp.async requires a 4-, 8-, or 16-byte transfer; source and destination layouts allow 2 bytes}}
    // expected-error @+1 {{failed to legalize operation 'ttg.async_copy_global_to_local'}}
    %token = ttg.async_copy_global_to_local %src, %dst : tensor<64x!tt.ptr<f16>, #pair> -> !ttg.memdesc<64xf16, #padded, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#regs = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.shared_linear<{offset = [[2], [1], [4], [8], [16], [32], [64]], block = []}, alignment = 16>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80"} {
  tt.func private @cp_async_noncontiguous_shared_offsets(%src: tensor<128x!tt.ptr<f16>, #regs> {tt.divisibility = dense<16> : tensor<1xi32>, tt.contiguity = dense<4> : tensor<1xi32>}, %dst: !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>) {
    // expected-error @+2 {{cp.async requires a 4-, 8-, or 16-byte transfer; source and destination layouts allow 2 bytes}}
    // expected-error @+1 {{failed to legalize operation 'ttg.async_copy_global_to_local'}}
    %token = ttg.async_copy_global_to_local %src, %dst : tensor<128x!tt.ptr<f16>, #regs> -> !ttg.memdesc<128xf16, #shared, #ttg.shared_memory, mutable>
    tt.return
  }
}
