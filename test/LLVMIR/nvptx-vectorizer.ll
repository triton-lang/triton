; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=90 %s | FileCheck %s --check-prefixes=CHECK,HOPPER,HOPPER-ONLY
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=100 %s | FileCheck %s --check-prefixes=CHECK,HOPPER,BLACKWELL
; RUN: sed '/^target triple =/d' %s | triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=90 | FileCheck %s --check-prefixes=CHECK,HOPPER,HOPPER-ONLY
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=80 %s | FileCheck %s --check-prefixes=CHECK,AMPERE
; RUN: triton-llvm-opt -nvptx-vectorize -mtriple=amdgcn-amd-amdhsa %s | FileCheck %s --check-prefix=OTHER
; RUN: triton-llvm-opt -nvptx-vectorize -nvptx-compute-capability=100 %s | llc -mtriple=nvptx64 -mcpu=sm_100 -fp-contract=off | FileCheck %s --check-prefix=PTX

target datalayout = "e-p:64:64-p1:64:64-p3:32:32-p5:64:64"
target triple = "nvptx64-nvidia-cuda"

; Adding zero can still canonicalize signed zero, but the scalar form does not
; need to materialize a packed zero operand.
; CHECK-LABEL: define void @scalar_float_add_zero(
; CHECK-NOT: fadd <2 x float>
; CHECK-COUNT-2: fadd float
; CHECK: ret void
define void @scalar_float_add_zero(ptr addrspace(1) %dst, <2 x float> %src) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %x = fadd float %a, 0.0
  %y = fadd float 0.0, %b
  %out0 = insertelement <2 x float> poison, float %x, i64 0
  %out1 = insertelement <2 x float> %out0, float %y, i64 1
  store <2 x float> %out1, ptr addrspace(1) %dst, align 8
  ret void
}

; Narrowing an existing register pair to a packed BF16 result does not need
; extra register moves, so zero canonicalization can stay packed as well.
; BLACKWELL-LABEL: define <2 x bfloat> @packed_zero_add_narrow(
; BLACKWELL: fadd <2 x float> %src, zeroinitializer
; HOPPER-ONLY-LABEL: define <2 x bfloat> @packed_zero_add_narrow(
; HOPPER-ONLY-COUNT-2: fadd float
; PTX-LABEL: packed_zero_add_narrow(
; PTX: add.rn.f32x2
; PTX: cvt.rn.bf16x2.f32
define <2 x bfloat> @packed_zero_add_narrow(<2 x float> %src) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %x = fadd float %a, 0.0
  %y = fadd float %b, 0.0
  %lo = fptrunc float %x to bfloat
  %hi = fptrunc float %y to bfloat
  %out0 = insertelement <2 x bfloat> poison, bfloat %lo, i64 0
  %out1 = insertelement <2 x bfloat> %out0, bfloat %hi, i64 1
  ret <2 x bfloat> %out1
}

; Keep a register pair packed through zero canonicalization and arithmetic.
; BLACKWELL-LABEL: define <2 x float> @packed_zero_add_chain(
; BLACKWELL: fadd <2 x float> %src, zeroinitializer
; BLACKWELL: fmul <2 x float>
; BLACKWELL: ret <2 x float>
; HOPPER-ONLY-LABEL: define <2 x float> @packed_zero_add_chain(
; HOPPER-ONLY-COUNT-2: fadd float
; HOPPER-ONLY-COUNT-2: fmul float
; HOPPER-ONLY: ret <2 x float>
; PTX-LABEL: packed_zero_add_chain(
; PTX: add.rn.f32x2
; PTX: mul.rn.f32x2
define <2 x float> @packed_zero_add_chain(<2 x float> %src, float %scale) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %x = fadd float %a, 0.0
  %y = fadd float %b, 0.0
  %u = fmul float %x, %scale
  %v = fmul float %y, %scale
  %out0 = insertelement <2 x float> poison, float %u, i64 0
  %out1 = insertelement <2 x float> %out0, float %v, i64 1
  ret <2 x float> %out1
}

; Keep negation visible to combines with packed arithmetic.
; BLACKWELL-LABEL: define <2 x float> @packed_negated_float_multiply(
; BLACKWELL: fneg <2 x float> %src
; BLACKWELL: fmul <2 x float>
; BLACKWELL: ret <2 x float>
define <2 x float> @packed_negated_float_multiply(<2 x float> %src,
                                                <2 x float> %scale) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %c = extractelement <2 x float> %scale, i64 0
  %d = extractelement <2 x float> %scale, i64 1
  %na = fneg float %a
  %nb = fneg float %b
  %x = fmul float %na, %c
  %y = fmul float %nb, %d
  %out0 = insertelement <2 x float> poison, float %x, i64 0
  %out1 = insertelement <2 x float> %out0, float %y, i64 1
  ret <2 x float> %out1
}

; Do not duplicate negations that also have scalar users.
; BLACKWELL-LABEL: define <2 x float> @shared_negated_float_multiply(
; BLACKWELL-COUNT-2: fneg float
; BLACKWELL: fmul <2 x float>
; BLACKWELL: ret <2 x float>
define <2 x float> @shared_negated_float_multiply(ptr addrspace(1) %dst,
                                                <2 x float> %src,
                                                <2 x float> %scale) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %c = extractelement <2 x float> %scale, i64 0
  %d = extractelement <2 x float> %scale, i64 1
  %na = fneg float %a
  %nb = fneg float %b
  store volatile float %na, ptr addrspace(1) %dst
  store volatile float %nb, ptr addrspace(1) %dst
  %x = fmul float %na, %c
  %y = fmul float %nb, %d
  %out0 = insertelement <2 x float> poison, float %x, i64 0
  %out1 = insertelement <2 x float> %out0, float %y, i64 1
  ret <2 x float> %out1
}

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

; One packed operation does not justify crossing both scalar boundaries for
; a marginal saving. The broadcast itself remains a free input.
; BLACKWELL-LABEL: define void @marginal_f32_broadcast_multiply(
; BLACKWELL-NOT: fmul <2 x float>
; BLACKWELL-COUNT-2: fmul float
; BLACKWELL: call void asm sideeffect "", "f,f"(float %first.result, float %second.result)
; BLACKWELL: ret void
; PTX-LABEL: marginal_f32_broadcast_multiply(
; PTX-NOT: mul.rn.f32x2
; PTX-COUNT-2: mul.rn.f32{{[ \t]}}
; PTX-NOT: mul.rn.f32x2
; PTX: ret;
define void @marginal_f32_broadcast_multiply(float %scale, float %first,
                                             float %second) {
  %first.result = fmul float %first, %scale
  %second.result = fmul float %second, %scale
  call void asm sideeffect "", "f,f"(float %first.result, float %second.result)
  ret void
}

; The same operation remains profitable when its varying input is packed.
; BLACKWELL-LABEL: define void @packed_input_f32_broadcast_multiply(
; BLACKWELL: [[SCALE:%.*]] = insertelement <2 x float> poison, float %scale, i64 0
; BLACKWELL: [[BROADCAST:%.*]] = insertelement <2 x float> [[SCALE]], float %scale, i64 1
; BLACKWELL: [[RESULT:%.*]] = fmul <2 x float> %src, [[BROADCAST]]
; BLACKWELL: [[LO:%.*]] = extractelement <2 x float> [[RESULT]], i64 0
; BLACKWELL: [[HI:%.*]] = extractelement <2 x float> [[RESULT]], i64 1
; BLACKWELL: call void asm sideeffect "", "f,f"(float [[LO]], float [[HI]])
; BLACKWELL: ret void
; PTX-LABEL: packed_input_f32_broadcast_multiply(
; PTX: mul.rn.f32x2
define void @packed_input_f32_broadcast_multiply(<2 x float> %src, float %scale) {
  %first = extractelement <2 x float> %src, i64 0
  %second = extractelement <2 x float> %src, i64 1
  %first.result = fmul float %first, %scale
  %second.result = fmul float %second, %scale
  call void asm sideeffect "", "f,f"(float %first.result, float %second.result)
  ret void
}

; Likewise, a genuinely packed output avoids the second scalar boundary.
; BLACKWELL-LABEL: define void @packed_output_f32_broadcast_multiply(
; BLACKWELL: [[RESULT:%.*]] = fmul <2 x float>
; BLACKWELL-NOT: extractelement <2 x float> [[RESULT]]
; BLACKWELL: store <2 x float> [[RESULT]], ptr addrspace(1) %dst
; BLACKWELL: ret void
; PTX-LABEL: packed_output_f32_broadcast_multiply(
; PTX: mul.rn.f32x2
define void @packed_output_f32_broadcast_multiply(float %scale, float %first,
                                                  float %second, ptr addrspace(1) %dst) {
  %first.result = fmul float %first, %scale
  %second.result = fmul float %second, %scale
  %out0 = insertelement <2 x float> poison, float %first.result, i64 0
  %out1 = insertelement <2 x float> %out0, float %second.result, i64 1
  store <2 x float> %out1, ptr addrspace(1) %dst, align 8
  ret void
}

; The conservative transition cost also applies to FMA: a temporary packed
; region with paid input packing and scalar users must have more than a marginal
; saving. Preserve explicit scalar FMA, including its single rounding.
; HOPPER-ONLY-LABEL: define void @scalar_f32_broadcast_fma(
; HOPPER-ONLY-COUNT-2: call float @llvm.fma.f32
; BLACKWELL-LABEL: define void @scalar_f32_broadcast_fma(
; BLACKWELL-NEXT: %first.result = call float @llvm.fma.f32(float %scale, float %first, float 0.000000e+00)
; BLACKWELL-NEXT: %second.result = call float @llvm.fma.f32(float %scale, float %second, float 0.000000e+00)
; BLACKWELL-NEXT: call void asm sideeffect "", "f,f"(float %first.result, float %second.result)
; BLACKWELL-NEXT: ret void
; PTX-LABEL: .visible .func scalar_f32_broadcast_fma(
; PTX-NOT: fma.rn.f32x2
; PTX-COUNT-2: fma.rn.f32{{[ \t]}}
; PTX-NOT: fma.rn.f32x2
; PTX: ret;
define void @scalar_f32_broadcast_fma(float %scale, float %first,
                                       float %second) {
  %first.result = call float @llvm.fma.f32(float %scale, float %first,
                                            float 0.0)
  %second.result = call float @llvm.fma.f32(float %scale, float %second,
                                             float 0.0)
  call void asm sideeffect "", "f,f"(float %first.result, float %second.result)
  ret void
}

; One register-tuple operand is free, but the other scalar input pair still
; requires materialization; the scalar output makes this another marginal case.
; BLACKWELL-LABEL: define void @scalar_f32_mixed_register_tuple(
; BLACKWELL-NOT: <2 x float>
; BLACKWELL: %result0 = fadd float %accumulator0, %value0
; BLACKWELL-NEXT: %result1 = fadd float %accumulator1, %value1
; BLACKWELL-NEXT: call void asm sideeffect "", "f,f"(float %result0, float %result1)
; BLACKWELL-NEXT: ret void
; PTX-LABEL: .visible .func scalar_f32_mixed_register_tuple(
; PTX-NOT: add.rn.f32x2
; PTX-COUNT-2: add.rn.f32{{[ \t]}}
; PTX-NOT: add.rn.f32x2
; PTX: ret;
define void @scalar_f32_mixed_register_tuple(float %accumulator0,
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

; Single-root planning is conservative when a marginal producer has multiple
; scalar consumers: it does not assume another root will amortize the packing.
; All original arithmetic, especially the explicit FMAs, remains unchanged.
; BLACKWELL-LABEL: define void @scalar_f32_shared_producer(
; BLACKWELL-NEXT: %product0 = fmul float %scale, %first
; BLACKWELL-NEXT: %product1 = fmul float %scale, %second
; BLACKWELL-NEXT: %fma0 = call float @llvm.fma.f32(float %factor, float %product0, float %product0)
; BLACKWELL-NEXT: %fma1 = call float @llvm.fma.f32(float %factor, float %product1, float %product1)
; BLACKWELL-NEXT: %square0 = fmul float %product0, %product0
; BLACKWELL-NEXT: %square1 = fmul float %product1, %product1
; BLACKWELL-NEXT: call void asm sideeffect "", "f,f,f,f"(float %fma0, float %fma1, float %square0, float %square1)
; BLACKWELL-NEXT: ret void
; PTX-LABEL: .visible .func scalar_f32_shared_producer(
; PTX-NOT: {{(mul|fma)}}.rn.f32x2
; PTX-COUNT-2: fma.rn.f32{{[ \t]}}
; PTX-NOT: {{(mul|fma)}}.rn.f32x2
; PTX: ret;
define void @scalar_f32_shared_producer(float %scale, float %factor,
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

; With an already packed input, the same shared producer and its consumers are
; profitable. Reuse the single packed product in both downstream operations.
; BLACKWELL-LABEL: define void @packed_input_f32_shared_producer(
; BLACKWELL: [[PRODUCT:%.*]] = fmul <2 x float>
; BLACKWELL-NOT: fmul <2 x float>
; BLACKWELL: [[FMA:%.*]] = call <2 x float> @llvm.fma.v2f32(<2 x float> {{.*}}, <2 x float> [[PRODUCT]], <2 x float> [[PRODUCT]])
; BLACKWELL: [[SQUARE:%.*]] = fmul <2 x float> [[PRODUCT]], [[PRODUCT]]
; PTX-LABEL: .visible .func packed_input_f32_shared_producer(
; PTX: mul.rn.f32x2
; PTX: fma.rn.f32x2
; PTX: mul.rn.f32x2
define void @packed_input_f32_shared_producer(float %scale, float %factor,
                                              <2 x float> %input) {
  %first = extractelement <2 x float> %input, i64 0
  %second = extractelement <2 x float> %input, i64 1
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

; BLACKWELL-LABEL: define void @scalar_short_mixed_precision_reduction(
; BLACKWELL-NOT: fadd <2 x float>
; BLACKWELL-COUNT-6: fadd float
; BLACKWELL: fmul <2 x float>
define void @scalar_short_mixed_precision_reduction(ptr addrspace(1) %dst,
                                                      <2 x half> %first,
                                                      <2 x half> %second,
                                                      <2 x half> %third) {
  %first0 = extractelement <2 x half> %first, i64 0
  %first1 = extractelement <2 x half> %first, i64 1
  %first.ext0 = fpext half %first0 to float
  %first.ext1 = fpext half %first1 to float
  %first.sum0 = fadd float %first.ext0, 0.0
  %first.sum1 = fadd float %first.ext1, 0.0
  %second0 = extractelement <2 x half> %second, i64 0
  %second1 = extractelement <2 x half> %second, i64 1
  %second.ext0 = fpext half %second0 to float
  %second.ext1 = fpext half %second1 to float
  %second.sum0 = fadd float %first.sum0, %second.ext0
  %second.sum1 = fadd float %first.sum1, %second.ext1
  %third0 = extractelement <2 x half> %third, i64 0
  %third1 = extractelement <2 x half> %third, i64 1
  %third.ext0 = fpext half %third0 to float
  %third.ext1 = fpext half %third1 to float
  %result0 = fadd float %second.sum0, %third.ext0
  %result1 = fadd float %second.sum1, %third.ext1
  %scaled0 = fmul float %result0, 2.0
  %scaled1 = fmul float %result1, 2.0
  %packed0 = insertelement <2 x float> poison, float %scaled0, i64 0
  %packed1 = insertelement <2 x float> %packed0, float %scaled1, i64 1
  store <2 x float> %packed1, ptr addrspace(1) %dst, align 8
  ret void
}

; BLACKWELL-LABEL: define void @packed_single_mixed_precision_add(
; BLACKWELL: fadd <2 x float>
define void @packed_single_mixed_precision_add(ptr addrspace(1) %dst,
                                                 <2 x half> %input,
                                                 <2 x float> %accumulator) {
  %input0 = extractelement <2 x half> %input, i64 0
  %input1 = extractelement <2 x half> %input, i64 1
  %extended0 = fpext half %input0 to float
  %extended1 = fpext half %input1 to float
  %accumulator0 = extractelement <2 x float> %accumulator, i64 0
  %accumulator1 = extractelement <2 x float> %accumulator, i64 1
  %result0 = fadd float %accumulator0, %extended0
  %result1 = fadd float %accumulator1, %extended1
  %packed0 = insertelement <2 x float> poison, float %result0, i64 0
  %packed1 = insertelement <2 x float> %packed0, float %result1, i64 1
  store <2 x float> %packed1, ptr addrspace(1) %dst, align 8
  ret void
}

; BLACKWELL-LABEL: define void @packed_long_mixed_precision_reduction(
; BLACKWELL-COUNT-2: fadd float
; BLACKWELL-COUNT-8: fadd <2 x float>
define void @packed_long_mixed_precision_reduction(ptr addrspace(1) %dst,
                                                     <2 x half> %first,
                                                     <2 x half> %second,
                                                     <2 x half> %third,
                                                     <2 x half> %fourth,
                                                     <2 x half> %fifth,
                                                     <2 x half> %sixth,
                                                     <2 x half> %seventh,
                                                     <2 x half> %eighth,
                                                     <2 x half> %ninth) {
  %first0 = extractelement <2 x half> %first, i64 0
  %first1 = extractelement <2 x half> %first, i64 1
  %first.ext0 = fpext half %first0 to float
  %first.ext1 = fpext half %first1 to float
  %first.sum0 = fadd float %first.ext0, 0.0
  %first.sum1 = fadd float %first.ext1, 0.0
  %second0 = extractelement <2 x half> %second, i64 0
  %second1 = extractelement <2 x half> %second, i64 1
  %second.ext0 = fpext half %second0 to float
  %second.ext1 = fpext half %second1 to float
  %second.sum0 = fadd float %first.sum0, %second.ext0
  %second.sum1 = fadd float %first.sum1, %second.ext1
  %third0 = extractelement <2 x half> %third, i64 0
  %third1 = extractelement <2 x half> %third, i64 1
  %third.ext0 = fpext half %third0 to float
  %third.ext1 = fpext half %third1 to float
  %third.sum0 = fadd float %second.sum0, %third.ext0
  %third.sum1 = fadd float %second.sum1, %third.ext1
  %fourth0 = extractelement <2 x half> %fourth, i64 0
  %fourth1 = extractelement <2 x half> %fourth, i64 1
  %fourth.ext0 = fpext half %fourth0 to float
  %fourth.ext1 = fpext half %fourth1 to float
  %fourth.sum0 = fadd float %third.sum0, %fourth.ext0
  %fourth.sum1 = fadd float %third.sum1, %fourth.ext1
  %fifth0 = extractelement <2 x half> %fifth, i64 0
  %fifth1 = extractelement <2 x half> %fifth, i64 1
  %fifth.ext0 = fpext half %fifth0 to float
  %fifth.ext1 = fpext half %fifth1 to float
  %fifth.sum0 = fadd float %fourth.sum0, %fifth.ext0
  %fifth.sum1 = fadd float %fourth.sum1, %fifth.ext1
  %sixth0 = extractelement <2 x half> %sixth, i64 0
  %sixth1 = extractelement <2 x half> %sixth, i64 1
  %sixth.ext0 = fpext half %sixth0 to float
  %sixth.ext1 = fpext half %sixth1 to float
  %sixth.sum0 = fadd float %fifth.sum0, %sixth.ext0
  %sixth.sum1 = fadd float %fifth.sum1, %sixth.ext1
  %seventh0 = extractelement <2 x half> %seventh, i64 0
  %seventh1 = extractelement <2 x half> %seventh, i64 1
  %seventh.ext0 = fpext half %seventh0 to float
  %seventh.ext1 = fpext half %seventh1 to float
  %seventh.sum0 = fadd float %sixth.sum0, %seventh.ext0
  %seventh.sum1 = fadd float %sixth.sum1, %seventh.ext1
  %eighth0 = extractelement <2 x half> %eighth, i64 0
  %eighth1 = extractelement <2 x half> %eighth, i64 1
  %eighth.ext0 = fpext half %eighth0 to float
  %eighth.ext1 = fpext half %eighth1 to float
  %eighth.sum0 = fadd float %seventh.sum0, %eighth.ext0
  %eighth.sum1 = fadd float %seventh.sum1, %eighth.ext1
  %ninth0 = extractelement <2 x half> %ninth, i64 0
  %ninth1 = extractelement <2 x half> %ninth, i64 1
  %ninth.ext0 = fpext half %ninth0 to float
  %ninth.ext1 = fpext half %ninth1 to float
  %result0 = fadd float %eighth.sum0, %ninth.ext0
  %result1 = fadd float %eighth.sum1, %ninth.ext1
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

; SIToFP contributions cannot use the separate direct-call accumulator matcher.
; Their paid input pack and scalar exit would make a single packed add marginal
; without the shared DAG's live packed-carry exemption.
; HOPPER-ONLY-LABEL: define void @packed_f32_converted_loop_carry(
; HOPPER-ONLY-COUNT-2: phi float
; HOPPER-ONLY-NOT: fadd <2 x float>
; HOPPER-ONLY-COUNT-2: fadd float
; HOPPER-ONLY: ret void
; BLACKWELL-LABEL: define void @packed_f32_converted_loop_carry(
; BLACKWELL: loop:
; BLACKWELL: [[ACC:%.*]] = phi <2 x float> [ [[NEXT:%.*]], %loop ], [ zeroinitializer, %entry ]
; BLACKWELL-NOT: phi float
; BLACKWELL-DAG: [[FIRST:%.*]] = sitofp i32 %first.input to float
; BLACKWELL-DAG: [[SECOND:%.*]] = sitofp i32 %second.input to float
; BLACKWELL: [[PACK0:%.*]] = insertelement <2 x float> poison, float [[FIRST]], i64 0
; BLACKWELL: [[PACK1:%.*]] = insertelement <2 x float> [[PACK0]], float [[SECOND]], i64 1
; BLACKWELL: [[NEXT]] = fadd <2 x float> [[ACC]], [[PACK1]]
; BLACKWELL: [[LO:%.*]] = extractelement <2 x float> [[NEXT]], i64 0
; BLACKWELL: [[HI:%.*]] = extractelement <2 x float> [[NEXT]], i64 1
; BLACKWELL: br i1 %continue, label %loop, label %exit
; BLACKWELL: exit:
; BLACKWELL: call void asm sideeffect "", "f,f"(float [[LO]], float [[HI]])
; BLACKWELL: ret void
; PTX-LABEL: .visible .func packed_f32_converted_loop_carry(
; PTX: add.rn.f32x2
define void @packed_f32_converted_loop_carry(i32 %first.input,
                                            i32 %second.input, i1 %continue) {
entry:
  br label %loop

loop:
  %first = phi float [ %first.next, %loop ], [ 0.0, %entry ]
  %second = phi float [ %second.next, %loop ], [ 0.0, %entry ]
  %first.contribution = sitofp i32 %first.input to float
  %second.contribution = sitofp i32 %second.input to float
  %first.next = fadd float %first, %first.contribution
  %second.next = fadd float %second, %second.contribution
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f"(float %first.next, float %second.next)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_loaded_loop_carry(
; BLACKWELL: [[INITIAL:%.*]] = load <2 x float>
; BLACKWELL: [[ACC:%.*]] = phi <2 x float> [ [[INITIAL]], %entry ], [ [[NEXT:%.*]], %latch ]
; BLACKWELL-NOT: phi float
; BLACKWELL: latch:
; BLACKWELL: [[NEXT]] = call <2 x float> @llvm.fma.v2f32(<2 x float> [[ACC]],
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 0
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 1
; PTX-LABEL: .visible .func packed_f32_loaded_loop_carry(
; PTX: fma.rn.f32x2
define void @packed_f32_loaded_loop_carry(ptr addrspace(1) %src,
                                           float %scale, i1 %continue) {
entry:
  %initial = load <2 x float>, ptr addrspace(1) %src, align 8
  %initial.first = extractelement <2 x float> %initial, i64 0
  %initial.second = extractelement <2 x float> %initial, i64 1
  br label %loop

loop:
  %first = phi float [ %initial.first, %entry ], [ %first.next, %latch ]
  %second = phi float [ %initial.second, %entry ], [ %second.next, %latch ]
  br label %latch

latch:
  %first.next = call float @llvm.fma.f32(float %first, float %scale, float 1.0)
  %second.next = call float @llvm.fma.f32(float %second, float %scale, float 1.0)
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f"(float %first.next, float %second.next)
  ret void
}

; BLACKWELL-LABEL: define void @packed_f32_register_tuple_loop_carry(
; BLACKWELL: entry:
; BLACKWELL: [[PACK0:%.*]] = insertelement <2 x float> poison, float %initial.first, i64 0
; BLACKWELL: [[INITIAL:%.*]] = insertelement <2 x float> [[PACK0]], float %initial.second, i64 1
; BLACKWELL: br label %loop
; BLACKWELL: loop:
; BLACKWELL: [[ACC:%.*]] = phi <2 x float> [ [[INITIAL]], %entry ], [ [[NEXT:%.*]], %loop ]
; BLACKWELL: [[NEXT]] = fadd <2 x float> [[ACC]], splat (float 1.000000e+00)
define void @packed_f32_register_tuple_loop_carry(ptr addrspace(1) %src,
                                                    i1 %continue) {
entry:
  %initial = call { i32, i32 } asm sideeffect
      "ld.global.v2.b32 { $0, $1 }, [ $2 ];", "=r,=r,l"(ptr addrspace(1) %src)
  %bits.first = extractvalue { i32, i32 } %initial, 0
  %bits.second = extractvalue { i32, i32 } %initial, 1
  %initial.first = bitcast i32 %bits.first to float
  %initial.second = bitcast i32 %bits.second to float
  br label %loop

loop:
  %first = phi float [ %initial.first, %entry ], [ %first.next, %loop ]
  %second = phi float [ %initial.second, %entry ], [ %second.next, %loop ]
  %first.next = fadd float %first, 1.0
  %second.next = fadd float %second, 1.0
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f"(float %first.next, float %second.next)
  ret void
}

; HOPPER-ONLY-LABEL: define void @packed_f32_cross_block_accumulators(
; HOPPER-ONLY: phi float
; HOPPER-ONLY: phi float
; BLACKWELL-LABEL: define void @packed_f32_cross_block_accumulators(
; BLACKWELL: [[ACC:%.*]] = phi <2 x float> [ zeroinitializer, %entry ], [ [[NEXT:%.*]], %latch ]
; BLACKWELL: [[FIRST:%.*]] = call float @llvm.nvvm.div.full(float %first, float %divisor)
; BLACKWELL: call void asm sideeffect "", ""()
; BLACKWELL: [[SECOND:%.*]] = call float @llvm.nvvm.div.full(float %second, float %divisor)
; BLACKWELL: [[PACK0:%.*]] = insertelement <2 x float> poison, float [[FIRST]], i64 0
; BLACKWELL: [[PACK1:%.*]] = insertelement <2 x float> [[PACK0]], float [[SECOND]], i64 1
; BLACKWELL: [[NEXT]] = fadd <2 x float> [[ACC]], [[PACK1]]
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 0
; BLACKWELL: extractelement <2 x float> [[NEXT]], i64 1
; PTX-LABEL: .visible .func packed_f32_cross_block_accumulators(
; PTX: add.rn.f32x2
define void @packed_f32_cross_block_accumulators(float %first, float %second,
                                                  float %divisor, i1 %continue) {
entry:
  br label %loop

loop:
  %first.acc = phi float [ 0.0, %entry ], [ %first.next, %latch ]
  %second.acc = phi float [ 0.0, %entry ], [ %second.next, %latch ]
  br label %latch

latch:
  %first.contribution = call float @llvm.nvvm.div.full(float %first, float %divisor)
  %first.next = fadd float %first.acc, %first.contribution
  call void asm sideeffect "", ""()
  %second.contribution = call float @llvm.nvvm.div.full(float %second, float %divisor)
  %second.next = fadd float %second.acc, %second.contribution
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f"(float %first.next, float %second.next)
  ret void
}

; HOPPER-ONLY-LABEL: define void @packed_f32_interleaved_accumulators(
; HOPPER-ONLY-NOT: fadd <2 x float>
; HOPPER-ONLY-COUNT-4: fadd float
; BLACKWELL-LABEL: define void @packed_f32_interleaved_accumulators(
; BLACKWELL: phi <2 x float>
; BLACKWELL: phi <2 x float>
; BLACKWELL: [[FIRST:%.*]] = call { i32, i32, i32, i32 } asm sideeffect
; BLACKWELL: [[SECOND:%.*]] = call { i32, i32, i32, i32 } asm sideeffect
; BLACKWELL: [[THIRD:%.*]] = call { i32, i32, i32, i32 } asm sideeffect
; BLACKWELL: fadd <2 x float>
; BLACKWELL: [[FOURTH:%.*]] = call { i32, i32, i32, i32 } asm sideeffect
; BLACKWELL: fadd <2 x float>
; PTX-LABEL: .visible .func packed_f32_interleaved_accumulators(
; PTX-COUNT-2: add.rn.f32x2
define void @packed_f32_interleaved_accumulators(ptr addrspace(1) %first,
                                                   ptr addrspace(1) %second,
                                                   ptr addrspace(1) %third,
                                                   ptr addrspace(1) %fourth,
                                                   i1 %continue) {
entry:
  br label %loop

loop:
  %first.acc = phi float [ 0.0, %entry ], [ %first.next, %loop ]
  %second.acc = phi float [ 0.0, %entry ], [ %second.next, %loop ]
  %third.acc = phi float [ 0.0, %entry ], [ %third.next, %loop ]
  %fourth.acc = phi float [ 0.0, %entry ], [ %fourth.next, %loop ]
  %first.load = call { i32, i32, i32, i32 } asm sideeffect
      "ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 ];",
      "=r,=r,=r,=r,l"(ptr addrspace(1) %first)
  %first.bits = extractvalue { i32, i32, i32, i32 } %first.load, 0
  %first.value = bitcast i32 %first.bits to float
  %first.next = fadd float %first.acc, %first.value
  %second.load = call { i32, i32, i32, i32 } asm sideeffect
      "ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 ];",
      "=r,=r,=r,=r,l"(ptr addrspace(1) %second)
  %second.bits = extractvalue { i32, i32, i32, i32 } %second.load, 0
  %second.value = bitcast i32 %second.bits to float
  %second.next = fadd float %second.acc, %second.value
  %third.load = call { i32, i32, i32, i32 } asm sideeffect
      "ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 ];",
      "=r,=r,=r,=r,l"(ptr addrspace(1) %third)
  %third.bits = extractvalue { i32, i32, i32, i32 } %third.load, 0
  %third.value = bitcast i32 %third.bits to float
  %third.next = fadd float %third.acc, %third.value
  %fourth.load = call { i32, i32, i32, i32 } asm sideeffect
      "ld.global.v4.b32 { $0, $1, $2, $3 }, [ $4 ];",
      "=r,=r,=r,=r,l"(ptr addrspace(1) %fourth)
  %fourth.bits = extractvalue { i32, i32, i32, i32 } %fourth.load, 0
  %fourth.value = bitcast i32 %fourth.bits to float
  %fourth.next = fadd float %fourth.acc, %fourth.value
  br i1 %continue, label %loop, label %exit

exit:
  call void asm sideeffect "", "f,f,f,f"(float %first.next, float %second.next,
                                          float %third.next, float %fourth.next)
  ret void
}

; BLACKWELL-LABEL: define void @different_divisor_accumulators(
; BLACKWELL: [[ACC:%.*]] = phi <2 x float>
; BLACKWELL: call float @llvm.nvvm.div.full(float %first, float %first.divisor)
; BLACKWELL: call float @llvm.nvvm.div.full(float %second, float %second.divisor)
; BLACKWELL: fadd <2 x float> [[ACC]],
define void @different_divisor_accumulators(float %first, float %second,
                                             float %first.divisor,
                                             float %second.divisor,
                                             i1 %continue) {
entry:
  br label %loop

loop:
  %first.acc = phi float [ 0.0, %entry ], [ %first.next, %latch ]
  %second.acc = phi float [ 0.0, %entry ], [ %second.next, %latch ]
  br label %latch

latch:
  %first.contribution = call float @llvm.nvvm.div.full(float %first,
                                                       float %first.divisor)
  %first.next = fadd float %first.acc, %first.contribution
  %second.contribution = call float @llvm.nvvm.div.full(float %second,
                                                        float %second.divisor)
  %second.next = fadd float %second.acc, %second.contribution
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

; Keep the absolute-value chain packed for the integer reduction. Reuse the
; scalar conversions kept live by the independent signed BF16 output store.
; BLACKWELL-LABEL: define i16 @packed_bf16_abs_umax(
; BLACKWELL: %b0 = fptrunc float %f0 to bfloat
; BLACKWELL: %b1 = fptrunc float %f1 to bfloat
; BLACKWELL-DAG: [[LOW:%.*]] = insertelement <2 x bfloat> poison, bfloat %b0, i64 0
; BLACKWELL-DAG: [[PAIR:%.*]] = insertelement <2 x bfloat> [[LOW]], bfloat %b1, i64 1
; BLACKWELL-DAG: [[RAW0:%.*]] = bitcast <2 x bfloat> [[PAIR]] to i32
; BLACKWELL-DAG: [[MASK0:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW0]])
; BLACKWELL-DAG: [[ABS0:%.*]] = bitcast i32 [[MASK0]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS0:%.*]] = bitcast <2 x bfloat> [[ABS0]] to <2 x i16>
; BLACKWELL-DAG: [[NARROW1:%.*]] = fptrunc <2 x float> {{%.*}} to <2 x bfloat>
; BLACKWELL-DAG: [[RAW1:%.*]] = bitcast <2 x bfloat> [[NARROW1]] to i32
; BLACKWELL-DAG: [[MASK1:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW1]])
; BLACKWELL-DAG: [[ABS1:%.*]] = bitcast i32 [[MASK1]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS1:%.*]] = bitcast <2 x bfloat> [[ABS1]] to <2 x i16>
; BLACKWELL: %signed0 = insertelement <2 x bfloat> poison, bfloat %b0, i64 0
; BLACKWELL: %signed1 = insertelement <2 x bfloat> %signed0, bfloat %b1, i64 1
; BLACKWELL: store <2 x bfloat> %signed1, ptr addrspace(1) %dst
; BLACKWELL: [[MAX:%.*]] = call <2 x i16> @llvm.umax.v2i16(<2 x i16> [[BITS0]], <2 x i16> [[BITS1]])
; BLACKWELL: [[RESULT:%.*]] = call i16 @llvm.vector.reduce.umax.v2i16(<2 x i16> [[MAX]])
; BLACKWELL: ret i16 [[RESULT]]
; HOPPER-ONLY-LABEL: define i16 @packed_bf16_abs_umax(
; HOPPER-ONLY-COUNT-3: call i16 @llvm.umax.i16
; HOPPER-ONLY: ret i16
; AMPERE-LABEL: define i16 @packed_bf16_abs_umax(
; AMPERE-COUNT-3: call i16 @llvm.umax.i16
; AMPERE: ret i16
; OTHER-LABEL: define i16 @packed_bf16_abs_umax(
; OTHER-COUNT-3: call i16 @llvm.umax.i16
; OTHER: ret i16
; PTX-LABEL: packed_bf16_abs_umax(
; PTX: max.u16x2
define i16 @packed_bf16_abs_umax(<4 x float> %src, ptr addrspace(1) %dst) {
  %f0 = extractelement <4 x float> %src, i64 0
  %f1 = extractelement <4 x float> %src, i64 1
  %f2 = extractelement <4 x float> %src, i64 2
  %f3 = extractelement <4 x float> %src, i64 3
  %b0 = fptrunc float %f0 to bfloat
  %b1 = fptrunc float %f1 to bfloat
  %b2 = fptrunc float %f2 to bfloat
  %b3 = fptrunc float %f3 to bfloat
  %a0 = call bfloat @llvm.fabs.bf16(bfloat %b0)
  %a1 = call bfloat @llvm.fabs.bf16(bfloat %b1)
  %a2 = call bfloat @llvm.fabs.bf16(bfloat %b2)
  %a3 = call bfloat @llvm.fabs.bf16(bfloat %b3)
  %i0 = bitcast bfloat %a0 to i16
  %i1 = bitcast bfloat %a1 to i16
  %i2 = bitcast bfloat %a2 to i16
  %i3 = bitcast bfloat %a3 to i16
  %signed0 = insertelement <2 x bfloat> poison, bfloat %b0, i64 0
  %signed1 = insertelement <2 x bfloat> %signed0, bfloat %b1, i64 1
  store <2 x bfloat> %signed1, ptr addrspace(1) %dst, align 4
  %left = call i16 @llvm.umax.i16(i16 %i0, i16 %i1)
  %right = call i16 @llvm.umax.i16(i16 %i2, i16 %i3)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Pure input packing stays with its dominating definitions. The integer
; reduction remains after the barrier in its own block.
; BLACKWELL-LABEL: define i16 @dominating_bf16_abs_umax(
; BLACKWELL: entry:
; BLACKWELL-DAG: [[RAW0:%.*]] = bitcast <2 x bfloat> {{%.*}} to i32
; BLACKWELL-DAG: [[MASK0:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW0]])
; BLACKWELL-DAG: [[ABS0:%.*]] = bitcast i32 [[MASK0]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS0:%.*]] = bitcast <2 x bfloat> [[ABS0]] to <2 x i16>
; BLACKWELL-DAG: [[RAW1:%.*]] = bitcast <2 x bfloat> {{%.*}} to i32
; BLACKWELL-DAG: [[MASK1:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW1]])
; BLACKWELL-DAG: [[ABS1:%.*]] = bitcast i32 [[MASK1]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS1:%.*]] = bitcast <2 x bfloat> [[ABS1]] to <2 x i16>
; BLACKWELL-NOT: @llvm.umax
; BLACKWELL: br label %reduce
; BLACKWELL: reduce:
; BLACKWELL-NEXT: call void asm sideeffect "bar.sync 0;", ""()
; BLACKWELL-NEXT: [[MAX:%.*]] = call <2 x i16> @llvm.umax.v2i16(<2 x i16> [[BITS0]], <2 x i16> [[BITS1]])
; BLACKWELL-NEXT: [[RESULT:%.*]] = call i16 @llvm.vector.reduce.umax.v2i16(<2 x i16> [[MAX]])
; BLACKWELL-NEXT: ret i16 [[RESULT]]
define i16 @dominating_bf16_abs_umax(<4 x bfloat> %src) {
entry:
  %b0 = extractelement <4 x bfloat> %src, i64 0
  %b1 = extractelement <4 x bfloat> %src, i64 1
  %b2 = extractelement <4 x bfloat> %src, i64 2
  %b3 = extractelement <4 x bfloat> %src, i64 3
  %a0 = call bfloat @llvm.fabs.bf16(bfloat %b0)
  %a1 = call bfloat @llvm.fabs.bf16(bfloat %b1)
  %a2 = call bfloat @llvm.fabs.bf16(bfloat %b2)
  %a3 = call bfloat @llvm.fabs.bf16(bfloat %b3)
  %i0 = bitcast bfloat %a0 to i16
  %i1 = bitcast bfloat %a1 to i16
  %i2 = bitcast bfloat %a2 to i16
  %i3 = bitcast bfloat %a3 to i16
  br label %reduce
reduce:
  call void asm sideeffect "bar.sync 0;", ""()
  %left = call i16 @llvm.umax.i16(i16 %i0, i16 %i1)
  %right = call i16 @llvm.umax.i16(i16 %i2, i16 %i3)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Five raw leaves cannot pay for the odd boundary pack and horizontal result.
; BLACKWELL-LABEL: define i16 @unprofitable_umax_odd_leaf(
; BLACKWELL-NOT: @llvm.vector.reduce.umax
; BLACKWELL-COUNT-4: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @unprofitable_umax_odd_leaf(<5 x i16> %src) {
  %a = extractelement <5 x i16> %src, i64 0
  %b = extractelement <5 x i16> %src, i64 1
  %c = extractelement <5 x i16> %src, i64 2
  %d = extractelement <5 x i16> %src, i64 3
  %e = extractelement <5 x i16> %src, i64 4
  %ab = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  %cd = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %abcd = call i16 @llvm.umax.i16(i16 %ab, i16 %cd)
  %result = call i16 @llvm.umax.i16(i16 %abcd, i16 %e)
  ret i16 %result
}

; A profitable odd leaf count needs the unsigned-max identity, not poison.
; BLACKWELL-LABEL: define i16 @packed_umax_odd_leaf(
; BLACKWELL-DAG: [[TAIL:%.*]] = insertelement <2 x i16> poison, i16 %g, i64 0
; BLACKWELL-DAG: [[PADDED:%.*]] = insertelement <2 x i16> [[TAIL]], i16 0, i64 1
; BLACKWELL-DAG: [[PAIR0:%.*]] = call <2 x i16> @llvm.umax.v2i16
; BLACKWELL-DAG: [[PAIR1:%.*]] = call <2 x i16> @llvm.umax.v2i16(<2 x i16> {{%.*}}, <2 x i16> [[PADDED]])
; BLACKWELL: [[MAX:%.*]] = call <2 x i16> @llvm.umax.v2i16(<2 x i16> [[PAIR0]], <2 x i16> [[PAIR1]])
; BLACKWELL: [[RESULT:%.*]] = call i16 @llvm.vector.reduce.umax.v2i16(<2 x i16> [[MAX]])
; BLACKWELL: ret i16 [[RESULT]]
define i16 @packed_umax_odd_leaf(<7 x i16> %src) {
  %a = extractelement <7 x i16> %src, i64 0
  %b = extractelement <7 x i16> %src, i64 1
  %c = extractelement <7 x i16> %src, i64 2
  %d = extractelement <7 x i16> %src, i64 3
  %e = extractelement <7 x i16> %src, i64 4
  %f = extractelement <7 x i16> %src, i64 5
  %g = extractelement <7 x i16> %src, i64 6
  %ab = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  %cd = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %ef = call i16 @llvm.umax.i16(i16 %e, i16 %f)
  %abcd = call i16 @llvm.umax.i16(i16 %ab, i16 %cd)
  %efg = call i16 @llvm.umax.i16(i16 %ef, i16 %g)
  %result = call i16 @llvm.umax.i16(i16 %abcd, i16 %efg)
  ret i16 %result
}

; Shared partial reductions cannot be consumed as single-use tree nodes.
; BLACKWELL-LABEL: define i16 @shared_umax_partial(
; BLACKWELL-NOT: @llvm.vector.reduce.umax
; BLACKWELL: %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
; BLACKWELL: store i16 %left, ptr addrspace(1) %dst
; BLACKWELL-COUNT-2: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @shared_umax_partial(<4 x i16> %src, ptr addrspace(1) %dst) {
  %a = extractelement <4 x i16> %src, i64 0
  %b = extractelement <4 x i16> %src, i64 1
  %c = extractelement <4 x i16> %src, i64 2
  %d = extractelement <4 x i16> %src, i64 3
  %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  store i16 %left, ptr addrspace(1) %dst, align 2
  %right = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Do not move a partial reduction across a block boundary.
; BLACKWELL-LABEL: define i16 @cross_block_umax(
; BLACKWELL-NOT: @llvm.vector.reduce.umax
; BLACKWELL: %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
; BLACKWELL: br label %next
; BLACKWELL-COUNT-2: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @cross_block_umax(<4 x i16> %src) {
entry:
  %a = extractelement <4 x i16> %src, i64 0
  %b = extractelement <4 x i16> %src, i64 1
  %c = extractelement <4 x i16> %src, i64 2
  %d = extractelement <4 x i16> %src, i64 3
  %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  br label %next
next:
  %right = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; A poison-masking select or freeze is a leaf, not part of the packed chain.
; BLACKWELL-LABEL: define i16 @umax_poison_boundary(
; BLACKWELL: %masked = select i1 %cond, i16 %a, i16 poison
; BLACKWELL: %frozen = freeze i16 %b
; BLACKWELL-NOT: @llvm.vector.reduce.umax
; BLACKWELL-COUNT-3: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @umax_poison_boundary(<4 x i16> %src, i1 %cond) {
  %a = extractelement <4 x i16> %src, i64 0
  %b = extractelement <4 x i16> %src, i64 1
  %c = extractelement <4 x i16> %src, i64 2
  %d = extractelement <4 x i16> %src, i64 3
  %masked = select i1 %cond, i16 %a, i16 poison
  %frozen = freeze i16 %b
  %left = call i16 @llvm.umax.i16(i16 %masked, i16 %frozen)
  %right = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Signed max is not the unsigned bit-pattern maximum used by this rewrite.
; BLACKWELL-LABEL: define i16 @signed_max_unchanged(
; BLACKWELL-NOT: @llvm.vector.reduce
; BLACKWELL-COUNT-3: call i16 @llvm.smax.i16
; BLACKWELL: ret i16 %result
define i16 @signed_max_unchanged(<4 x i16> %src) {
  %a = extractelement <4 x i16> %src, i64 0
  %b = extractelement <4 x i16> %src, i64 1
  %c = extractelement <4 x i16> %src, i64 2
  %d = extractelement <4 x i16> %src, i64 3
  %left = call i16 @llvm.smax.i16(i16 %a, i16 %b)
  %right = call i16 @llvm.smax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.smax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Scalar bitcasts can hide vector-valued sources; do not try to pack those
; sources into an invalid vector of vectors.
; CHECK-LABEL: define i16 @umax_vector_bitcast_leaves(
; CHECK-NOT: @llvm.vector.reduce
; CHECK-COUNT-3: call i16 @llvm.umax.i16
; CHECK: ret i16 %result
define i16 @umax_vector_bitcast_leaves() {
  %a = bitcast <2 x i8> <i8 1, i8 2> to i16
  %b = bitcast <2 x i8> <i8 3, i8 4> to i16
  %c = bitcast <2 x i8> <i8 5, i8 6> to i16
  %d = bitcast <2 x i8> <i8 7, i8 8> to i16
  %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  %right = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; A single packed fabs does not pay for packing unrelated scalar inputs and
; unpacking both results for scalar consumers.
; BLACKWELL-LABEL: define void @unprofitable_scalar_bf16_abs(
; BLACKWELL-NOT: @llvm.fabs.v2bf16
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL-COUNT-2: call bfloat @llvm.fabs.bf16
; BLACKWELL: call void asm sideeffect
; BLACKWELL: ret void
define void @unprofitable_scalar_bf16_abs(bfloat %a, bfloat %b) {
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %a)
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %b)
  %bits0 = bitcast bfloat %abs0 to i16
  %bits1 = bitcast bfloat %abs1 to i16
  call void asm sideeffect "", "h,h"(i16 %bits0, i16 %bits1)
  ret void
}

; Reuse an already packed input and output without requiring a reduction root.
; BLACKWELL-LABEL: define void @packed_bf16_abs_store(
; BLACKWELL: [[RAW:%.*]] = bitcast <2 x bfloat> %src to i32
; BLACKWELL: [[MASK:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW]])
; BLACKWELL: [[ABS:%.*]] = bitcast i32 [[MASK]] to <2 x bfloat>
; BLACKWELL-NOT: extractelement
; BLACKWELL: store <2 x bfloat> [[ABS]], ptr addrspace(1) %dst
; BLACKWELL: ret void
define void @packed_bf16_abs_store(<2 x bfloat> %src, ptr addrspace(1) %dst) {
  %a = extractelement <2 x bfloat> %src, i64 0
  %b = extractelement <2 x bfloat> %src, i64 1
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %a)
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %b)
  %out0 = insertelement <2 x bfloat> poison, bfloat %abs0, i64 0
  %out1 = insertelement <2 x bfloat> %out0, bfloat %abs1, i64 1
  store <2 x bfloat> %out1, ptr addrspace(1) %dst, align 4
  ret void
}

; Conversion and fabs use the same packed DAG when the sink is a plain store.
; BLACKWELL-LABEL: define void @packed_narrow_abs_store(
; BLACKWELL: [[NARROW:%.*]] = fptrunc <2 x float> %src to <2 x bfloat>
; BLACKWELL: [[RAW:%.*]] = bitcast <2 x bfloat> [[NARROW]] to i32
; BLACKWELL: [[MASK:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW]])
; BLACKWELL: [[ABS:%.*]] = bitcast i32 [[MASK]] to <2 x bfloat>
; BLACKWELL-NOT: extractelement
; BLACKWELL: store <2 x bfloat> [[ABS]], ptr addrspace(1) %dst
; BLACKWELL: ret void
define void @packed_narrow_abs_store(<2 x float> %src, ptr addrspace(1) %dst) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %lo = fptrunc float %a to bfloat
  %hi = fptrunc float %b to bfloat
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %lo)
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %hi)
  %out0 = insertelement <2 x bfloat> poison, bfloat %abs0, i64 0
  %out1 = insertelement <2 x bfloat> %out0, bfloat %abs1, i64 1
  store <2 x bfloat> %out1, ptr addrspace(1) %dst, align 4
  ret void
}

; The narrowing pair is a shared diamond node, not two independently credited
; or duplicated producers for the direct and absolute-value paths.
; BLACKWELL-LABEL: define void @packed_shared_narrow_diamond(
; BLACKWELL: [[NARROW:%.*]] = fptrunc <2 x float> %src to <2 x bfloat>
; BLACKWELL-NOT: fptrunc
; BLACKWELL: [[RAW:%.*]] = bitcast <2 x bfloat> [[NARROW]] to i32
; BLACKWELL: [[MASK:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW]])
; BLACKWELL: [[ABS:%.*]] = bitcast i32 [[MASK]] to <2 x bfloat>
; BLACKWELL-NOT: fptrunc
; BLACKWELL-NOT: @llvm.fabs
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL: [[SUM:%.*]] = fadd <2 x bfloat> [[ABS]], [[NARROW]]
; BLACKWELL-NOT: fptrunc
; BLACKWELL-NOT: @llvm.fabs
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL: store <2 x bfloat> [[SUM]], ptr addrspace(1) %dst
; BLACKWELL: ret void
define void @packed_shared_narrow_diamond(<2 x float> %src, ptr addrspace(1) %dst) {
  %a = extractelement <2 x float> %src, i64 0
  %b = extractelement <2 x float> %src, i64 1
  %lo = fptrunc float %a to bfloat
  %hi = fptrunc float %b to bfloat
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %lo)
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %hi)
  %sum0 = fadd bfloat %abs0, %lo
  %sum1 = fadd bfloat %abs1, %hi
  %out0 = insertelement <2 x bfloat> poison, bfloat %sum0, i64 0
  %out1 = insertelement <2 x bfloat> %out0, bfloat %sum1, i64 1
  store <2 x bfloat> %out1, ptr addrspace(1) %dst, align 4
  ret void
}

; Each scalar observer precedes the next volatile input load, so a packed
; result cannot be hoisted to replace the earlier scalar use.
; Duplicating these live producers cannot claim their removal as savings.
; BLACKWELL-LABEL: define i16 @umax_retained_scalar_narrowing(
; BLACKWELL-NOT: fptrunc <2 x float>
; BLACKWELL: %x0 = load volatile float, ptr addrspace(1) %src
; BLACKWELL: %b0 = fptrunc float %x0 to bfloat
; BLACKWELL: call void asm sideeffect "", "h"(i16 %a)
; BLACKWELL: %x1 = load volatile float, ptr addrspace(1) %p1
; BLACKWELL: %b1 = fptrunc float %x1 to bfloat
; BLACKWELL: call void asm sideeffect "", "h"(i16 %b)
; BLACKWELL: %x2 = load volatile float, ptr addrspace(1) %p2
; BLACKWELL: %b2 = fptrunc float %x2 to bfloat
; BLACKWELL: call void asm sideeffect "", "h"(i16 %c)
; BLACKWELL: %x3 = load volatile float, ptr addrspace(1) %p3
; BLACKWELL: %b3 = fptrunc float %x3 to bfloat
; BLACKWELL: call void asm sideeffect "", "h"(i16 %d)
; BLACKWELL-NOT: @llvm.vector.reduce
; BLACKWELL-COUNT-3: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @umax_retained_scalar_narrowing(ptr addrspace(1) %src) {
  %x0 = load volatile float, ptr addrspace(1) %src, align 4
  %b0 = fptrunc float %x0 to bfloat
  %a = bitcast bfloat %b0 to i16
  call void asm sideeffect "", "h"(i16 %a)
  %p1 = getelementptr float, ptr addrspace(1) %src, i64 1
  %x1 = load volatile float, ptr addrspace(1) %p1, align 4
  %b1 = fptrunc float %x1 to bfloat
  %b = bitcast bfloat %b1 to i16
  call void asm sideeffect "", "h"(i16 %b)
  %p2 = getelementptr float, ptr addrspace(1) %src, i64 2
  %x2 = load volatile float, ptr addrspace(1) %p2, align 4
  %b2 = fptrunc float %x2 to bfloat
  %c = bitcast bfloat %b2 to i16
  call void asm sideeffect "", "h"(i16 %c)
  %p3 = getelementptr float, ptr addrspace(1) %src, i64 3
  %x3 = load volatile float, ptr addrspace(1) %p3, align 4
  %b3 = fptrunc float %x3 to bfloat
  %d = bitcast bfloat %b3 to i16
  call void asm sideeffect "", "h"(i16 %d)
  %left = call i16 @llvm.umax.i16(i16 %a, i16 %b)
  %right = call i16 @llvm.umax.i16(i16 %c, i16 %d)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; Even a pure integer reduction must pay for its unpaired scalar inputs and
; horizontal result; discovering a reduction is not a profitability override.
; BLACKWELL-LABEL: define i16 @unprofitable_umax_scalar_inputs(
; BLACKWELL-NOT: @llvm.vector.reduce
; BLACKWELL-COUNT-4: freeze i16
; BLACKWELL-NOT: @llvm.umax.v2i16
; BLACKWELL-COUNT-3: call i16 @llvm.umax.i16
; BLACKWELL: ret i16 %result
define i16 @unprofitable_umax_scalar_inputs(i16 %a, i16 %b, i16 %c, i16 %d) {
  %fa = freeze i16 %a
  %fb = freeze i16 %b
  %fc = freeze i16 %c
  %fd = freeze i16 %d
  %left = call i16 @llvm.umax.i16(i16 %fa, i16 %fb)
  %right = call i16 @llvm.umax.i16(i16 %fc, i16 %fd)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; A long integer reduction can profit by packing the retained scalar values,
; without duplicating their externally observed fabs producers.
; BLACKWELL-LABEL: define i16 @packed_umax_retained_fabs_cut(
; BLACKWELL-NOT: @llvm.fabs.v2bf16
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL: %abs0 = call bfloat @llvm.fabs.bf16(bfloat %x0)
; BLACKWELL-NOT: @llvm.fabs.v2bf16
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL: %abs15 = call bfloat @llvm.fabs.bf16(bfloat %x15)
; BLACKWELL-NOT: @llvm.fabs.v2bf16
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL: call void asm sideeffect "", "h"(i16 %bits15)
; BLACKWELL-NOT: @llvm.fabs.v2bf16
; BLACKWELL-NOT: asm "and.b32
; BLACKWELL-COUNT-7: call <2 x i16> @llvm.umax.v2i16
; BLACKWELL: [[RESULT:%.*]] = call i16 @llvm.vector.reduce.umax.v2i16
; BLACKWELL: ret i16 [[RESULT]]
define i16 @packed_umax_retained_fabs_cut(ptr addrspace(1) %src) {
  %x0 = load volatile bfloat, ptr addrspace(1) %src, align 2
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %x0)
  %bits0 = bitcast bfloat %abs0 to i16
  call void asm sideeffect "", "h"(i16 %bits0)
  %p1 = getelementptr bfloat, ptr addrspace(1) %src, i64 1
  %x1 = load volatile bfloat, ptr addrspace(1) %p1, align 2
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %x1)
  %bits1 = bitcast bfloat %abs1 to i16
  call void asm sideeffect "", "h"(i16 %bits1)
  %p2 = getelementptr bfloat, ptr addrspace(1) %src, i64 2
  %x2 = load volatile bfloat, ptr addrspace(1) %p2, align 2
  %abs2 = call bfloat @llvm.fabs.bf16(bfloat %x2)
  %bits2 = bitcast bfloat %abs2 to i16
  call void asm sideeffect "", "h"(i16 %bits2)
  %p3 = getelementptr bfloat, ptr addrspace(1) %src, i64 3
  %x3 = load volatile bfloat, ptr addrspace(1) %p3, align 2
  %abs3 = call bfloat @llvm.fabs.bf16(bfloat %x3)
  %bits3 = bitcast bfloat %abs3 to i16
  call void asm sideeffect "", "h"(i16 %bits3)
  %p4 = getelementptr bfloat, ptr addrspace(1) %src, i64 4
  %x4 = load volatile bfloat, ptr addrspace(1) %p4, align 2
  %abs4 = call bfloat @llvm.fabs.bf16(bfloat %x4)
  %bits4 = bitcast bfloat %abs4 to i16
  call void asm sideeffect "", "h"(i16 %bits4)
  %p5 = getelementptr bfloat, ptr addrspace(1) %src, i64 5
  %x5 = load volatile bfloat, ptr addrspace(1) %p5, align 2
  %abs5 = call bfloat @llvm.fabs.bf16(bfloat %x5)
  %bits5 = bitcast bfloat %abs5 to i16
  call void asm sideeffect "", "h"(i16 %bits5)
  %p6 = getelementptr bfloat, ptr addrspace(1) %src, i64 6
  %x6 = load volatile bfloat, ptr addrspace(1) %p6, align 2
  %abs6 = call bfloat @llvm.fabs.bf16(bfloat %x6)
  %bits6 = bitcast bfloat %abs6 to i16
  call void asm sideeffect "", "h"(i16 %bits6)
  %p7 = getelementptr bfloat, ptr addrspace(1) %src, i64 7
  %x7 = load volatile bfloat, ptr addrspace(1) %p7, align 2
  %abs7 = call bfloat @llvm.fabs.bf16(bfloat %x7)
  %bits7 = bitcast bfloat %abs7 to i16
  call void asm sideeffect "", "h"(i16 %bits7)
  %p8 = getelementptr bfloat, ptr addrspace(1) %src, i64 8
  %x8 = load volatile bfloat, ptr addrspace(1) %p8, align 2
  %abs8 = call bfloat @llvm.fabs.bf16(bfloat %x8)
  %bits8 = bitcast bfloat %abs8 to i16
  call void asm sideeffect "", "h"(i16 %bits8)
  %p9 = getelementptr bfloat, ptr addrspace(1) %src, i64 9
  %x9 = load volatile bfloat, ptr addrspace(1) %p9, align 2
  %abs9 = call bfloat @llvm.fabs.bf16(bfloat %x9)
  %bits9 = bitcast bfloat %abs9 to i16
  call void asm sideeffect "", "h"(i16 %bits9)
  %p10 = getelementptr bfloat, ptr addrspace(1) %src, i64 10
  %x10 = load volatile bfloat, ptr addrspace(1) %p10, align 2
  %abs10 = call bfloat @llvm.fabs.bf16(bfloat %x10)
  %bits10 = bitcast bfloat %abs10 to i16
  call void asm sideeffect "", "h"(i16 %bits10)
  %p11 = getelementptr bfloat, ptr addrspace(1) %src, i64 11
  %x11 = load volatile bfloat, ptr addrspace(1) %p11, align 2
  %abs11 = call bfloat @llvm.fabs.bf16(bfloat %x11)
  %bits11 = bitcast bfloat %abs11 to i16
  call void asm sideeffect "", "h"(i16 %bits11)
  %p12 = getelementptr bfloat, ptr addrspace(1) %src, i64 12
  %x12 = load volatile bfloat, ptr addrspace(1) %p12, align 2
  %abs12 = call bfloat @llvm.fabs.bf16(bfloat %x12)
  %bits12 = bitcast bfloat %abs12 to i16
  call void asm sideeffect "", "h"(i16 %bits12)
  %p13 = getelementptr bfloat, ptr addrspace(1) %src, i64 13
  %x13 = load volatile bfloat, ptr addrspace(1) %p13, align 2
  %abs13 = call bfloat @llvm.fabs.bf16(bfloat %x13)
  %bits13 = bitcast bfloat %abs13 to i16
  call void asm sideeffect "", "h"(i16 %bits13)
  %p14 = getelementptr bfloat, ptr addrspace(1) %src, i64 14
  %x14 = load volatile bfloat, ptr addrspace(1) %p14, align 2
  %abs14 = call bfloat @llvm.fabs.bf16(bfloat %x14)
  %bits14 = bitcast bfloat %abs14 to i16
  call void asm sideeffect "", "h"(i16 %bits14)
  %p15 = getelementptr bfloat, ptr addrspace(1) %src, i64 15
  %x15 = load volatile bfloat, ptr addrspace(1) %p15, align 2
  %abs15 = call bfloat @llvm.fabs.bf16(bfloat %x15)
  %bits15 = bitcast bfloat %abs15 to i16
  call void asm sideeffect "", "h"(i16 %bits15)
  %max0 = call i16 @llvm.umax.i16(i16 %bits0, i16 %bits1)
  %max1 = call i16 @llvm.umax.i16(i16 %bits2, i16 %bits3)
  %max2 = call i16 @llvm.umax.i16(i16 %bits4, i16 %bits5)
  %max3 = call i16 @llvm.umax.i16(i16 %bits6, i16 %bits7)
  %max4 = call i16 @llvm.umax.i16(i16 %bits8, i16 %bits9)
  %max5 = call i16 @llvm.umax.i16(i16 %bits10, i16 %bits11)
  %max6 = call i16 @llvm.umax.i16(i16 %bits12, i16 %bits13)
  %max7 = call i16 @llvm.umax.i16(i16 %bits14, i16 %bits15)
  %max8 = call i16 @llvm.umax.i16(i16 %max0, i16 %max1)
  %max9 = call i16 @llvm.umax.i16(i16 %max2, i16 %max3)
  %max10 = call i16 @llvm.umax.i16(i16 %max4, i16 %max5)
  %max11 = call i16 @llvm.umax.i16(i16 %max6, i16 %max7)
  %max12 = call i16 @llvm.umax.i16(i16 %max8, i16 %max9)
  %max13 = call i16 @llvm.umax.i16(i16 %max10, i16 %max11)
  %max14 = call i16 @llvm.umax.i16(i16 %max12, i16 %max13)
  ret i16 %max14
}

; PTX packed floating abs may canonicalize NaN payloads. Keep the exact sign
; mask visible through lowering, including a masked scalar input and bitwise max.
; BLACKWELL-LABEL: define i16 @packed_bf16_abs_nan_payload_bits(
; BLACKWELL-DAG: [[RAW0:%.*]] = bitcast <2 x bfloat> {{%.*}} to i32
; BLACKWELL-DAG: [[MASK0:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW0]])
; BLACKWELL-DAG: [[ABS0:%.*]] = bitcast i32 [[MASK0]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS0:%.*]] = bitcast <2 x bfloat> [[ABS0]] to <2 x i16>
; BLACKWELL-DAG: [[RAW1:%.*]] = bitcast <2 x bfloat> {{%.*}} to i32
; BLACKWELL-DAG: [[MASK1:%.*]] = call i32 asm "and.b32 $0, $1, 0x7fff7fff;", "=r,r"(i32 [[RAW1]])
; BLACKWELL-DAG: [[ABS1:%.*]] = bitcast i32 [[MASK1]] to <2 x bfloat>
; BLACKWELL-DAG: [[BITS1:%.*]] = bitcast <2 x bfloat> [[ABS1]] to <2 x i16>
; BLACKWELL: [[MAX:%.*]] = call <2 x i16> @llvm.umax.v2i16(<2 x i16> [[BITS0]], <2 x i16> [[BITS1]])
; BLACKWELL: [[RESULT:%.*]] = call i16 @llvm.vector.reduce.umax.v2i16(<2 x i16> [[MAX]])
; BLACKWELL: ret i16 [[RESULT]]
; PTX-LABEL: packed_bf16_abs_nan_payload_bits(
; PTX-NOT: abs.bf16x2
; PTX: and.b32 {{.*}}0x7fff7fff;
; PTX-NOT: abs.bf16x2
; PTX: and.b32 {{.*}}0x7fff7fff;
; PTX-NOT: abs.bf16x2
; PTX: max.u16x2
; PTX-NOT: abs.bf16x2
; PTX: ret;
define i16 @packed_bf16_abs_nan_payload_bits(bfloat %a, bfloat %b,
                                             bfloat %c, bfloat %d, i1 %active) {
  %masked = select i1 %active, bfloat %d, bfloat 0.0
  %abs0 = call bfloat @llvm.fabs.bf16(bfloat %a)
  %abs1 = call bfloat @llvm.fabs.bf16(bfloat %b)
  %abs2 = call bfloat @llvm.fabs.bf16(bfloat %c)
  %abs3 = call bfloat @llvm.fabs.bf16(bfloat %masked)
  %i0 = bitcast bfloat %abs0 to i16
  %i1 = bitcast bfloat %abs1 to i16
  %i2 = bitcast bfloat %abs2 to i16
  %i3 = bitcast bfloat %abs3 to i16
  %left = call i16 @llvm.umax.i16(i16 %i0, i16 %i1)
  %right = call i16 @llvm.umax.i16(i16 %i2, i16 %i3)
  %result = call i16 @llvm.umax.i16(i16 %left, i16 %right)
  ret i16 %result
}

; A packed sign-bit XOR is not free. Without enough arithmetic savings, leave
; the original scalar negations and their signed-zero/NaN semantics unchanged.
; BLACKWELL-LABEL: define void @unprofitable_bf16_negation_store(
; BLACKWELL-NOT: fneg <2 x bfloat>
; BLACKWELL-NOT: asm "xor.b32
; BLACKWELL-COUNT-2: fneg bfloat
; BLACKWELL-NOT: asm "xor.b32
; BLACKWELL: store <2 x bfloat> %out1, ptr addrspace(1) %dst
; BLACKWELL: ret void
define void @unprofitable_bf16_negation_store(<2 x bfloat> %src, ptr addrspace(1) %dst) {
  %a = extractelement <2 x bfloat> %src, i64 0
  %b = extractelement <2 x bfloat> %src, i64 1
  %neg0 = fneg bfloat %a
  %neg1 = fneg bfloat %b
  %out0 = insertelement <2 x bfloat> poison, bfloat %neg0, i64 0
  %out1 = insertelement <2 x bfloat> %out0, bfloat %neg1, i64 1
  store <2 x bfloat> %out1, ptr addrspace(1) %dst, align 4
  ret void
}

declare bfloat @llvm.fabs.bf16(bfloat)
declare i16 @llvm.umax.i16(i16, i16)
declare i16 @llvm.smax.i16(i16, i16)
