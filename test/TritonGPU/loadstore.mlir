// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu=target=cuda:80 -convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

// CHECK-LABEL: @load_store_ops_atomic
tt.func @load_store_ops_atomic(%ptr: !tt.ptr<f32>) {
  // CHECK: llvm.inline_asm {{.*}} "mov.u32 $0, 0x0;\0A\09@$2 ld.relaxed.gpu.global.b32 { $0 }, [ $1 + 0 ];", "=r,l,b"
  %0 = tt.load %ptr memSemantic = relaxed memSyncScope = gpu : !tt.ptr<f32>
  // CHECK: llvm.inline_asm {{.*}} "@$2 st.acquire.cta.global.b32 [ $1 + 0 ], { $0 };", "r,l,b"
  tt.store %ptr, %0 memSemantic = acquire memSyncScope = cta : !tt.ptr<f32>
  tt.return
}
