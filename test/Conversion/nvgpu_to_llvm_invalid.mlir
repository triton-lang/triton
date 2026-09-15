// RUN: triton-opt %s --convert-nv-gpu-to-llvm -split-input-file -verify-diagnostics

!struct_16xi32 = !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>

// The s8 form of wgmma.mma_async exists only for untransposed operands. This is
// the form triton-lang/triton#11586 reached, and it used to fire an assertion.
llvm.func @wgmma_s8_transposed_b(%desc: i64, %in: !struct_16xi32) {
  %false = llvm.mlir.constant(false) : i1
  // expected-error @+1 {{unsupported wgmma instruction m64n32k32 with A=s8, B=s8, C=s32, layoutA=row, layoutB=row; this element type combination exists only for untransposed operands}}
  %acc0 = nvg.wgmma %desc, %desc, %false {
    eltTypeA = 0 : i32,
    eltTypeB = 0 : i32,
    eltTypeC = 1 : i32,
    layoutA = 0 : i32,
    layoutB = 0 : i32,
    m = 64 : i32,
    n = 32 : i32,
    k = 32 : i32
  } : (i64, i64, i1) -> !struct_16xi32
  llvm.return
}
