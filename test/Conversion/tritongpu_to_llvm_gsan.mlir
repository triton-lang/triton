// RUN: triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --allocate-shared-memory-nv --convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func @load_store
  // CHECK: llvm.call @__triton_gsan_init(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32) -> ()
  // CHECK: nvvm.barrier0
  // CHECK: llvm.store %{{.*}} : i64, !llvm.ptr
  // CHECK: llvm.store %{{.*}} : i8, !llvm.ptr
  // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
  // CHECK-2: ld.global
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
  // CHECK-2: st.global
  tt.func @load_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>,
                      %other: tensor<256xf32, #blocked>, %vals: tensor<256xf32, #blocked>) {
    %loaded = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#shared_f16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @tma_f16_gsan_merge
  tt.func @tma_f16_gsan_merge(%desc: !tt.tensordesc<tensor<32x64xf16, #shared_f16>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x64xf16, #shared_f16, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: llvm.alloca %{{.*}} x !llvm.struct<(array<32 x i64>, array<32 x i8>)>
    // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[BYTES:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %[[COUNT]], %[[BYTES]], %{{.*}}, %{{.*}})
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<tensor<32x64xf16, #shared_f16>>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared_f16, #smem, mutable>
    tt.return
  }
}

// -----

#shared_f16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @tma_f16_gsan_merge_4warps
  tt.func @tma_f16_gsan_merge_4warps(%desc: !tt.tensordesc<tensor<128x64xf16, #shared_f16>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared_f16, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: llvm.alloca %{{.*}} x !llvm.struct<(array<32 x i64>, array<32 x i8>)>
    // CHECK: %[[COUNT_4W:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[BYTES_4W:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %[[COUNT_4W]], %[[BYTES_4W]], %{{.*}}, %{{.*}})
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<tensor<128x64xf16, #shared_f16>>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_f16, #smem, mutable>
    tt.return
  }
}
