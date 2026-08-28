; RUN: triton-llvm-opt -vectorize-extracted-adds %s | FileCheck %s
; RUN: triton-llvm-opt -vectorize-extracted-adds %s | llc -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=fast | FileCheck %s --check-prefix=PTX

; Re-form the full-lane scalar adds so NVPTX sees packed FMA candidates.
define void @full2(<2 x float> %a, <2 x float> %b, ptr %out) {
; PTX-LABEL: full2(
; PTX:       fma.rn.f32x2
; PTX:       ret;
; CHECK-LABEL: define void @full2(
; CHECK:         [[MUL:%.*]] = fmul <2 x float> %a, %b
; CHECK-NEXT:    [[ADD:%.*]] = fadd <2 x float> [[MUL]], splat (float 1.000000e+00)
; CHECK-NEXT:    [[LO:%.*]] = extractelement <2 x float> [[ADD]], i64 0
; CHECK-NEXT:    [[HI:%.*]] = extractelement <2 x float> [[ADD]], i64 1
; CHECK-NEXT:    store float [[LO]], ptr %out, align 4
; CHECK:         store float [[HI]], ptr
  %mul = fmul <2 x float> %a, %b
  %lo = extractelement <2 x float> %mul, i64 0
  %lo.add = fadd float %lo, 1.0
  %hi = extractelement <2 x float> %mul, i64 1
  %hi.add = fadd float 1.0, %hi
  store float %lo.add, ptr %out
  %next = getelementptr float, ptr %out, i64 1
  store float %hi.add, ptr %next
  ret void
}

define void @full8(<8 x float> %a, <8 x float> %b) {
; CHECK-LABEL: define void @full8(
; CHECK:         [[MUL:%.*]] = fmul <8 x float> %a, %b
; CHECK-NEXT:    [[ADD:%.*]] = fadd <8 x float> [[MUL]], zeroinitializer
; CHECK-COUNT-8: extractelement <8 x float> [[ADD]]
; CHECK-NEXT:    ret void
  %mul = fmul <8 x float> %a, %b
  %x0 = extractelement <8 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <8 x float> %mul, i64 1
  %a1 = fadd float %x1, 0.0
  %x2 = extractelement <8 x float> %mul, i64 2
  %a2 = fadd float %x2, 0.0
  %x3 = extractelement <8 x float> %mul, i64 3
  %a3 = fadd float %x3, 0.0
  %x4 = extractelement <8 x float> %mul, i64 4
  %a4 = fadd float %x4, 0.0
  %x5 = extractelement <8 x float> %mul, i64 5
  %a5 = fadd float %x5, 0.0
  %x6 = extractelement <8 x float> %mul, i64 6
  %a6 = fadd float %x6, 0.0
  %x7 = extractelement <8 x float> %mul, i64 7
  %a7 = fadd float %x7, 0.0
  ret void
}

; Keep partial, ambiguous, or non-uniform groups scalar.
define float @missing_lane(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: define float @missing_lane(
; CHECK-NOT:     fadd <2 x float>
; CHECK:         fadd float
  %mul = fmul <2 x float> %a, %b
  %lo = extractelement <2 x float> %mul, i64 0
  %r = fadd float %lo, 0.0
  ret float %r
}

define void @duplicate_lane(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: define void @duplicate_lane(
; CHECK-NOT:     fadd <2 x float>
; CHECK-COUNT-2: fadd float
  %mul = fmul <2 x float> %a, %b
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <2 x float> %mul, i64 0
  %a1 = fadd float %x1, 0.0
  ret void
}

define void @vector_user(<2 x float> %a, <2 x float> %b, ptr %out) {
; CHECK-LABEL: define void @vector_user(
; CHECK-NOT:     fadd <2 x float>
; CHECK-COUNT-2: fadd float
  %mul = fmul <2 x float> %a, %b
  store <2 x float> %mul, ptr %out
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <2 x float> %mul, i64 1
  %a1 = fadd float %x1, 0.0
  ret void
}

define void @different_addends(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: define void @different_addends(
; CHECK-NOT:     fadd <2 x float>
; CHECK-COUNT-2: fadd float
  %mul = fmul <2 x float> %a, %b
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <2 x float> %mul, i64 1
  %a1 = fadd float %x1, 1.0
  ret void
}

define void @different_flags(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: define void @different_flags(
; CHECK-NOT:     fadd <2 x float>
; CHECK:         fadd float
; CHECK:         fadd fast float
  %mul = fmul <2 x float> %a, %b
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <2 x float> %mul, i64 1
  %a1 = fadd fast float %x1, 0.0
  ret void
}

define void @strict(<2 x float> %a, <2 x float> %b) strictfp {
; CHECK-LABEL: define void @strict(
; CHECK-NOT:     fadd <2 x float>
; CHECK-COUNT-2: fadd float
  %mul = fmul <2 x float> %a, %b
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  %x1 = extractelement <2 x float> %mul, i64 1
  %a1 = fadd float %x1, 0.0
  ret void
}

define void @different_blocks(<2 x float> %a, <2 x float> %b) {
; CHECK-LABEL: define void @different_blocks(
; CHECK-NOT:     fadd <2 x float>
; CHECK-COUNT-2: fadd float
entry:
  %mul = fmul <2 x float> %a, %b
  %x0 = extractelement <2 x float> %mul, i64 0
  %a0 = fadd float %x0, 0.0
  br label %next
next:
  %x1 = extractelement <2 x float> %mul, i64 1
  %a1 = fadd float %x1, 0.0
  ret void
}
