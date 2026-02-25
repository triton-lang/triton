// RUN: triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --allocate-shared-memory-nv --convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: llvm.func @load_store
  // CHECK: llvm.call @__triton_gsan_init(%{{.*}}) : (!llvm.ptr) -> ()
  // CHECK: llvm.store %{{.*}} : i64, !llvm.ptr
  // CHECK: llvm.store %{{.*}} : i8, !llvm.ptr
  // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
  // CHECK-2: ld.global
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32) -> ()
  // CHECK-2: st.global
  tt.func @load_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>,
                      %other: tensor<256xf32, #blocked>, %vals: tensor<256xf32, #blocked>) {
    %loaded = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
