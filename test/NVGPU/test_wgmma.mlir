// RUN: triton-translate %s | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32,  "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_tma(%opC : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) {
    %buffer = llvm.mlir.null : !llvm.ptr<i64, 3>
    %height = arith.constant 16 : i32
    // CHECK: call i64 @__nv_get_wgmma_desc
    %descA = nvgpu.wgmma_desc_create %buffer, %height {mode = 2 : i32}: (!llvm.ptr<i64, 3>, i32) -> (i64)
    // CHECK: call i64 @__nv_get_wgmma_desc
    %descB = nvgpu.wgmma_desc_create %buffer, %height {mode = 2 : i32}: (!llvm.ptr<i64, 3>, i32) -> (i64)

    // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
    %acc0 = nvgpu.wgmma %descA, %descA, %opC {m=64:i32, n=64:i32, k=16:i32, eltTypeC=7:i32, eltTypeA=4:i32, eltTypeB=4:i32, layoutA=0:i32, layoutB=0:i32} : (i64, i64, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) -> (!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)
    tt.return
  }
} // end module
