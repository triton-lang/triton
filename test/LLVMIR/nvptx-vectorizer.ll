; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=90 %s | FileCheck %s --check-prefixes=CHECK,HOPPER,HOPPER-ONLY
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=100 %s | FileCheck %s --check-prefixes=CHECK,HOPPER,BLACKWELL
; RUN: sed '/^target triple =/d' %s | triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=90 | FileCheck %s --check-prefixes=CHECK,HOPPER,HOPPER-ONLY
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=80 %s | FileCheck %s --check-prefixes=CHECK,AMPERE
; RUN: triton-llvm-opt -nvptx-vectorize -mtriple=amdgcn-amd-amdhsa %s | FileCheck %s --check-prefix=OTHER
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=100 %s | llc -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=off | FileCheck %s --check-prefix=PTX

target datalayout = "e-p:64:64-p1:64:64-p3:32:32-p5:64:64"
target triple = "nvptx64-nvidia-cuda"

; CHECK-LABEL: define void @scalar_memory_unchanged(
; CHECK: load i32
; CHECK: load i32
; CHECK: store i32
; CHECK: store i32
; CHECK-NOT: <2 x i32>
define void @scalar_memory_unchanged(ptr addrspace(1) %dst,
                                      ptr addrspace(1) %src) {
  %src1 = getelementptr i32, ptr addrspace(1) %src, i64 1
  %dst1 = getelementptr i32, ptr addrspace(1) %dst, i64 1
  %first = load i32, ptr addrspace(1) %src, align 8
  %second = load i32, ptr addrspace(1) %src1, align 4
  store i32 %first, ptr addrspace(1) %dst, align 8
  store i32 %second, ptr addrspace(1) %dst1, align 4
  ret void
}

; CHECK-LABEL: define void @scalar_i32_arithmetic(
; CHECK-NOT: add <
; CHECK-COUNT-2: add i32
define void @scalar_i32_arithmetic(ptr addrspace(1) %dst, i32 %a, i32 %b,
                                   i32 %c, i32 %d) {
  %first = add i32 %a, %b
  %second = add i32 %c, %d
  %dst1 = getelementptr i32, ptr addrspace(1) %dst, i64 1
  store i32 %first, ptr addrspace(1) %dst, align 8
  store i32 %second, ptr addrspace(1) %dst1, align 4
  ret void
}

; CHECK-LABEL: define void @packed_half_add(
; CHECK: [[SUM:%.*]] = fadd fast <2 x half> %lhs, %rhs
; CHECK: store <2 x half> [[SUM]], ptr addrspace(1) %dst
; CHECK-NOT: extractelement <2 x half>
; PTX-LABEL: .visible .func packed_half_add(
; PTX: add.f16x2
; OTHER-LABEL: define void @packed_half_add(
; OTHER: fadd fast half
; OTHER: fadd fast half
define void @packed_half_add(ptr addrspace(1) %dst, <2 x half> %lhs,
                             <2 x half> %rhs) {
  %lhs0 = extractelement <2 x half> %lhs, i64 0
  %lhs1 = extractelement <2 x half> %lhs, i64 1
  %rhs0 = extractelement <2 x half> %rhs, i64 0
  %rhs1 = extractelement <2 x half> %rhs, i64 1
  %sum0 = fadd fast half %lhs0, %rhs0
  %sum1 = fadd fast half %lhs1, %rhs1
  %packed0 = insertelement <2 x half> poison, half %sum0, i64 0
  %packed1 = insertelement <2 x half> %packed0, half %sum1, i64 1
  store <2 x half> %packed1, ptr addrspace(1) %dst, align 4
  ret void
}

; CHECK-LABEL: define void @packed_half_wide_input(
; CHECK: [[LEFT:%.*]] = shufflevector <4 x half> %lhs, <4 x half> poison, <2 x i32> <i32 2, i32 3>
; CHECK: [[RIGHT:%.*]] = shufflevector <4 x half> %rhs, <4 x half> poison, <2 x i32> <i32 2, i32 3>
; CHECK: [[SUM:%.*]] = fadd <2 x half> [[LEFT]], [[RIGHT]]
; CHECK: store <2 x half> [[SUM]], ptr addrspace(1) %dst
define void @packed_half_wide_input(ptr addrspace(1) %dst, <4 x half> %lhs,
                                    <4 x half> %rhs) {
  %lhs2 = extractelement <4 x half> %lhs, i64 2
  %lhs3 = extractelement <4 x half> %lhs, i64 3
  %rhs2 = extractelement <4 x half> %rhs, i64 2
  %rhs3 = extractelement <4 x half> %rhs, i64 3
  %sum0 = fadd half %lhs2, %rhs2
  %sum1 = fadd half %lhs3, %rhs3
  %packed0 = insertelement <2 x half> poison, half %sum0, i64 0
  %packed1 = insertelement <2 x half> %packed0, half %sum1, i64 1
  store <2 x half> %packed1, ptr addrspace(1) %dst, align 4
  ret void
}

; HOPPER-LABEL: define void @packed_bfloat_mul(
; HOPPER: [[PRODUCT:%.*]] = fmul <2 x bfloat> %lhs, %rhs
; HOPPER: store <2 x bfloat> [[PRODUCT]], ptr addrspace(1) %dst
; PTX-LABEL: .visible .func packed_bfloat_mul(
; PTX: mul.rn.bf16x2
; AMPERE-LABEL: define void @packed_bfloat_mul(
; AMPERE-NOT: fmul <2 x bfloat>
; AMPERE: fmul bfloat
; AMPERE: fmul bfloat
define void @packed_bfloat_mul(ptr addrspace(1) %dst, <2 x bfloat> %lhs,
                               <2 x bfloat> %rhs) {
  %lhs0 = extractelement <2 x bfloat> %lhs, i64 0
  %lhs1 = extractelement <2 x bfloat> %lhs, i64 1
  %rhs0 = extractelement <2 x bfloat> %rhs, i64 0
  %rhs1 = extractelement <2 x bfloat> %rhs, i64 1
  %product0 = fmul bfloat %lhs0, %rhs0
  %product1 = fmul bfloat %lhs1, %rhs1
  %packed0 = insertelement <2 x bfloat> poison, bfloat %product0, i64 0
  %packed1 = insertelement <2 x bfloat> %packed0, bfloat %product1, i64 1
  store <2 x bfloat> %packed1, ptr addrspace(1) %dst, align 4
  ret void
}

; HOPPER-ONLY-LABEL: define void @packed_f32_from_vectors(
; HOPPER-ONLY-NOT: fadd <2 x float>
; HOPPER-ONLY-COUNT-2: fadd float
; BLACKWELL-LABEL: define void @packed_f32_from_vectors(
; BLACKWELL: [[SUM:%.*]] = fadd <2 x float> %lhs, %rhs
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 0
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 1
define void @packed_f32_from_vectors(<2 x float> %lhs, <2 x float> %rhs) {
  %lhs0 = extractelement <2 x float> %lhs, i64 0
  %lhs1 = extractelement <2 x float> %lhs, i64 1
  %rhs0 = extractelement <2 x float> %rhs, i64 0
  %rhs1 = extractelement <2 x float> %rhs, i64 1
  %sum0 = fadd float %lhs0, %rhs0
  %sum1 = fadd float %lhs1, %rhs1
  call void asm sideeffect "", "f,f"(float %sum0, float %sum1)
  ret void
}

; BLACKWELL-LABEL: define void @unprofitable_f32_packing(
; BLACKWELL-NOT: fadd <2 x float>
; BLACKWELL-COUNT-2: fadd float
; PTX-LABEL: .visible .func unprofitable_f32_packing(
; PTX-NOT: f32x2
; PTX: add.rn.f32
; PTX: add.rn.f32
define void @unprofitable_f32_packing(float %lhs0, float %lhs1, float %rhs0,
                                      float %rhs1) {
  %sum0 = fadd float %lhs0, %rhs0
  %sum1 = fadd float %lhs1, %rhs1
  call void asm sideeffect "", "f,f"(float %sum0, float %sum1)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_register_tuple(
; BLACKWELL: [[SUM:%.*]] = fadd <2 x float>
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 0
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 1
; PTX-LABEL: .visible .func packed_f32_register_tuple(
; PTX: add.rn.f32x2
define void @packed_f32_register_tuple(float %accumulator0,
                                        float %accumulator1) {
  %pair = call { i32, i32 } asm sideeffect
      "mov.u32 $0, 0;\0A\09mov.u32 $1, 0;", "=r,=r"()
  %bits0 = extractvalue { i32, i32 } %pair, 0
  %bits1 = extractvalue { i32, i32 } %pair, 1
  %value0 = bitcast i32 %bits0 to float
  %value1 = bitcast i32 %bits1 to float
  %result0 = fadd float %accumulator0, %value0
  %result1 = fadd float %accumulator1, %value1
  call void asm sideeffect "", "f,f"(float %result0, float %result1)
  ret void
}

; BLACKWELL-LABEL: define void @profitable_f32_scalar_chain(
; BLACKWELL: [[PRODUCT:%.*]] = fmul fast <2 x float>
; BLACKWELL-NOT: extractelement <2 x float> [[PRODUCT]]
; BLACKWELL: [[SUM:%.*]] = fadd <2 x float> [[PRODUCT]], splat (float 1.000000e+00)
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 0
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 1
; PTX-LABEL: .visible .func profitable_f32_scalar_chain(
; PTX: mul.f32x2
; PTX: add.rn.f32x2
define void @profitable_f32_scalar_chain(float %lhs0, float %lhs1,
                                          float %rhs0, float %rhs1) {
  %product0 = fmul fast float %lhs0, %rhs0
  %product1 = fmul fast float %lhs1, %rhs1
  %sum0 = fadd float %product0, 1.0
  %sum1 = fadd float %product1, 1.0
  call void asm sideeffect "", "f,f"(float %sum0, float %sum1)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_shared_producer(
; BLACKWELL: [[PRODUCT:%.*]] = fmul <2 x float>
; BLACKWELL: [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> {{.*}}, <2 x float> [[PRODUCT]], <2 x float> [[PRODUCT]])
; BLACKWELL: [[SQUARE:%.*]] = fmul <2 x float> [[PRODUCT]], [[PRODUCT]]
; PTX-LABEL: .visible .func packed_f32_shared_producer(
; PTX: mul.rn.f32x2
; PTX: fma.rn.f32x2
; PTX: mul.rn.f32x2
define void @packed_f32_shared_producer(float %scale, float %factor,
                                         float %first, float %second) {
  %product0 = fmul float %scale, %first
  %product1 = fmul float %scale, %second
  %fma0 = call float @llvm.fma.f32(float %factor, float %product0,
                                    float %product0)
  %fma1 = call float @llvm.fma.f32(float %factor, float %product1,
                                    float %product1)
  %square0 = fmul float %product0, %product0
  %square1 = fmul float %product1, %product1
  call void asm sideeffect "", "f,f,f,f"(float %fma0, float %fma1,
                                          float %square0, float %square1)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_reduction_chain(
; BLACKWELL: [[FIRST:%.*]] = fadd <2 x float> %lhs, %rhs
; BLACKWELL: [[SECOND:%.*]] = fadd <2 x float> [[FIRST]], %third
; BLACKWELL: [[THIRD:%.*]] = fadd <2 x float> [[SECOND]], %fourth
; BLACKWELL: store <2 x float> [[THIRD]], ptr addrspace(1) %dst
; BLACKWELL-NOT: extractelement <2 x float>
; PTX-LABEL: .visible .func packed_f32_reduction_chain(
; PTX: add.rn.f32x2 [[FIRSTREG:%rd[0-9]+]],
; PTX: add.rn.f32x2 [[SECONDREG:%rd[0-9]+]], [[FIRSTREG]],
; PTX: add.rn.f32x2 {{%rd[0-9]+}}, [[SECONDREG]],
define void @packed_f32_reduction_chain(ptr addrspace(1) %dst,
                                         <2 x float> %lhs,
                                         <2 x float> %rhs,
                                         <2 x float> %third,
                                         <2 x float> %fourth) {
  %lhs0 = extractelement <2 x float> %lhs, i64 0
  %lhs1 = extractelement <2 x float> %lhs, i64 1
  %rhs0 = extractelement <2 x float> %rhs, i64 0
  %rhs1 = extractelement <2 x float> %rhs, i64 1
  %third0 = extractelement <2 x float> %third, i64 0
  %third1 = extractelement <2 x float> %third, i64 1
  %fourth0 = extractelement <2 x float> %fourth, i64 0
  %fourth1 = extractelement <2 x float> %fourth, i64 1
  %first0 = fadd float %lhs0, %rhs0
  %first1 = fadd float %lhs1, %rhs1
  %second0 = fadd float %first0, %third0
  %second1 = fadd float %first1, %third1
  %result0 = fadd float %second0, %fourth0
  %result1 = fadd float %second1, %fourth1
  %packed0 = insertelement <2 x float> poison, float %result0, i64 0
  %packed1 = insertelement <2 x float> %packed0, float %result1, i64 1
  store <2 x float> %packed1, ptr addrspace(1) %dst, align 8
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_loop_reduction(
; BLACKWELL: [[ACC:%.*]] = phi <2 x float> [ [[NEXT:%.*]], %loop ], [ zeroinitializer, %entry ]
; BLACKWELL: [[NEXT]] = fadd <2 x float> [[ACC]], %input
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 0
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 1
; PTX-LABEL: .visible .func packed_f32_loop_reduction(
; PTX: add.rn.f32x2
define void @packed_f32_loop_reduction(<2 x float> %input, i1 %continue) {
entry:
  br label %loop

loop:
  %first = phi float [ %first.next, %loop ], [ 0.0, %entry ]
  %second = phi float [ %second.next, %loop ], [ 0.0, %entry ]
  %input.first = extractelement <2 x float> %input, i64 0
  %input.second = extractelement <2 x float> %input, i64 1
  %first.next = fadd float %first, %input.first
  %second.next = fadd float %second, %input.second
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f"(float %first.next, float %second.next)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_late_operands(
; BLACKWELL: [[SUM:%.*]] = fadd <2 x float> %lhs, %rhs
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 0
; BLACKWELL: extractelement <2 x float> [[SUM]], i64 1
define void @packed_f32_late_operands(<2 x float> %lhs, <2 x float> %rhs) {
  %lhs0 = extractelement <2 x float> %lhs, i64 0
  %rhs0 = extractelement <2 x float> %rhs, i64 0
  %first = fadd float %lhs0, %rhs0
  %lhs1 = extractelement <2 x float> %lhs, i64 1
  %rhs1 = extractelement <2 x float> %rhs, i64 1
  %second = fadd float %lhs1, %rhs1
  call void asm sideeffect "", "f,f"(float %first, float %second)
  ret void
}

declare float @llvm.fma.f32(float, float, float)
declare half @llvm.fma.f16(half, half, half)
declare float @llvm.nvvm.div.full(float, float)

; HOPPER-ONLY-LABEL: define void @packed_f32_fma(
; HOPPER-ONLY-COUNT-2: call float @llvm.fma.f32
; BLACKWELL-LABEL: define void @packed_f32_fma(
; BLACKWELL: [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> %lhs, <2 x float> %rhs, <2 x float> %addend)
; BLACKWELL: extractelement <2 x float> [[FMA]], i64 0
; BLACKWELL: extractelement <2 x float> [[FMA]], i64 1
; PTX-LABEL: .visible .func packed_f32_fma(
; PTX: fma.rn.f32x2
define void @packed_f32_fma(<2 x float> %lhs, <2 x float> %rhs,
                             <2 x float> %addend) {
  %lhs0 = extractelement <2 x float> %lhs, i64 0
  %lhs1 = extractelement <2 x float> %lhs, i64 1
  %rhs0 = extractelement <2 x float> %rhs, i64 0
  %rhs1 = extractelement <2 x float> %rhs, i64 1
  %addend0 = extractelement <2 x float> %addend, i64 0
  %addend1 = extractelement <2 x float> %addend, i64 1
  %result0 = call float @llvm.fma.f32(float %lhs0, float %rhs0,
                                       float %addend0)
  %result1 = call float @llvm.fma.f32(float %lhs1, float %rhs1,
                                       float %addend1)
  call void asm sideeffect "", "f,f"(float %result0, float %result1)
  ret void
}

; CHECK-LABEL: define void @packed_half_fma(
; CHECK: [[FMA:%.*]] = call <2 x half> @llvm.fma.v2f16(<2 x half> %lhs, <2 x half> %rhs, <2 x half> %addend)
; CHECK: store <2 x half> [[FMA]], ptr addrspace(1) %dst
; PTX-LABEL: .visible .func packed_half_fma(
; PTX: fma.rn.f16x2
define void @packed_half_fma(ptr addrspace(1) %dst, <2 x half> %lhs,
                             <2 x half> %rhs, <2 x half> %addend) {
  %lhs0 = extractelement <2 x half> %lhs, i64 0
  %lhs1 = extractelement <2 x half> %lhs, i64 1
  %rhs0 = extractelement <2 x half> %rhs, i64 0
  %rhs1 = extractelement <2 x half> %rhs, i64 1
  %addend0 = extractelement <2 x half> %addend, i64 0
  %addend1 = extractelement <2 x half> %addend, i64 1
  %result0 = call half @llvm.fma.f16(half %lhs0, half %rhs0, half %addend0)
  %result1 = call half @llvm.fma.f16(half %lhs1, half %rhs1, half %addend1)
  %packed0 = insertelement <2 x half> poison, half %result0, i64 0
  %packed1 = insertelement <2 x half> %packed0, half %result1, i64 1
  store <2 x half> %packed1, ptr addrspace(1) %dst, align 4
  ret void
}

; BLACKWELL-LABEL: define void @dependent_f32_arithmetic(
; BLACKWELL-NOT: fadd <2 x float>
; BLACKWELL-COUNT-2: fadd float
define void @dependent_f32_arithmetic(float %lhs, float %rhs) {
  %first = fadd float %lhs, %rhs
  %second = fadd float %first, %rhs
  call void asm sideeffect "", "f,f"(float %first, float %second)
  ret void
}

; BLACKWELL-LABEL: define void @scalar_f32_division_denominators(
; BLACKWELL-NOT: fadd <2 x float>
; BLACKWELL-COUNT-2: fadd float
; BLACKWELL-COUNT-2: call float @llvm.nvvm.div.full
define void @scalar_f32_division_denominators(<2 x float> %numerators,
                                               <2 x float> %denominators) {
  %numerator0 = extractelement <2 x float> %numerators, i64 0
  %numerator1 = extractelement <2 x float> %numerators, i64 1
  %denominator0 = extractelement <2 x float> %denominators, i64 0
  %denominator1 = extractelement <2 x float> %denominators, i64 1
  %first = fadd float %denominator0, 1.0
  %second = fadd float %denominator1, 1.0
  %result0 = call float @llvm.nvvm.div.full(float %numerator0, float %first)
  %result1 = call float @llvm.nvvm.div.full(float %numerator1, float %second)
  call void asm sideeffect "", "f,f"(float %result0, float %result1)
  ret void
}
