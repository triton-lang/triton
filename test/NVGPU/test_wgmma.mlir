// RUN: triton-opt %s -split-input-file --convert-nv-gpu-to-llvm | FileCheck %s
#SHARED = #triton_gpu.shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32,  "triton_gpu.num-ctas" = 2 : i32} {
  tt.func @test_tma(%opC : !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) {
    %buffer = llvm.mlir.null : !llvm.ptr<i64, 3>
    %height = arith.constant 16 : i32
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.shl
    // CHECK: llvm.lshr
    // CHECK: llvm.zext
    // CHECK: llvm.mul
    // CHECK: llvm.lshr
    // CHECK: llvm.shl
    // CHECK: llvm.lshr
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.or
    %descA = nvgpu.wgmma_desc_create %buffer, %height {mode = 2 : i32}: (!llvm.ptr<i64, 3>, i32) -> (i64)
    // CHECK: llvm.ptrtoint
    // CHECK: llvm.shl
    // CHECK: llvm.lshr
    // CHECK: llvm.zext
    // CHECK: llvm.mul
    // CHECK: llvm.lshr
    // CHECK: llvm.shl
    // CHECK: llvm.lshr
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.shl
    // CHECK: llvm.or
    // CHECK: llvm.or
    %descB = nvgpu.wgmma_desc_create %buffer, %height {mode = 2 : i32}: (!llvm.ptr<i64, 3>, i32) -> (i64)

    // CHECK-COUNT-32: llvm.extractvalue
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {$0,$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31}, $64, $65, 1, 1, 1, 0, 1;"
    %acc0 = nvgpu.wgmma %descA, %descB, %opC {m=64:i32, n=64:i32, k=16:i32, eltTypeC=7:i32, eltTypeA=4:i32, eltTypeB=4:i32, layoutA=0:i32, layoutB=0:i32} : (i64, i64, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) -> (!llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>)
    tt.return
  }
} // end module
