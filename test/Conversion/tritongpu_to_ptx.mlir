// RUN: split-file %s %t
// RUN: triton-opt %t/common.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM90 --dump-input-context=20 %s
// RUN: triton-opt %t/common.mlir --allocate-shared-memory-nv='compute-capability=80 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM80 --dump-input-context=20 %s
// RUN: triton-opt %t/common.mlir --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck --check-prefixes CHECK,SM100 --dump-input-context=20 %s
// RUN: triton-opt %t/common.mlir --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' -cse | FileCheck --check-prefix=VEC80 --dump-input-context=20 %s
// RUN: triton-opt %t/common.mlir --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' -cse | FileCheck --check-prefix=VEC90 --dump-input-context=20 %s
// RUN: triton-opt %t/common.mlir --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' -cse | FileCheck --check-prefix=VEC100 --dump-input-context=20 %s
// RUN: triton-opt %t/fp4.mlir --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100a -mattr=+ptx87 | FileCheck %s --check-prefix=FP4
// RUN: triton-opt %t/f64_bf16.mlir --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefix=F64BF16 --dump-input-context=20 %s
// RUN: triton-opt %t/f64_bf16.mlir --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck --check-prefix=F64BF16 --dump-input-context=20 %s

//--- common.mlir
#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#narrow_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#narrow_dst = #ttg.linear<{register = [], lane = [[2], [1], [4], [8], [16]], warp = [[32]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reciprocal_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: reciprocal_f32(
    // CHECK: div.full.f32
    %one = arith.constant 1.0 : f32
    %result = arith.divf %one, %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @approx_reciprocal_tensor_f32(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %arg: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: approx_reciprocal_tensor_f32(
    // CHECK-COUNT-8: rcp.approx.f32
    %one = arith.constant dense<1.0> : tensor<256xf32, #blocked>
    %result = tt.approx_divf %one, %arg : tensor<256xf32, #blocked>
    tt.store %ptr, %result : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @divide_f32(%ptr: !tt.ptr<f32>, %lhs: f32, %rhs: f32) {
    // CHECK-LABEL: divide_f32(
    // CHECK: div.full.f32
    %result = arith.divf %lhs, %rhs : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @approx_divide_f32(%ptr: !tt.ptr<f32>, %lhs: f32, %rhs: f32) {
    // CHECK-LABEL: approx_divide_f32(
    // CHECK: div.approx.f32
    %result = tt.approx_divf %lhs, %rhs : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @approx_reciprocal_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: approx_reciprocal_f32(
    // CHECK: rcp.approx.f32
    %one = arith.constant 1.0 : f32
    %result = tt.approx_divf %one, %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @reciprocal_f64(%ptr: !tt.ptr<f64>, %arg: f64) {
    // CHECK-LABEL: reciprocal_f64(
    // CHECK: div.rn.f64
    %one = arith.constant 1.0 : f64
    %result = arith.divf %one, %arg : f64
    tt.store %ptr, %result : !tt.ptr<f64>
    tt.return
  }

  tt.func public @precise_reciprocal_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: precise_reciprocal_f32(
    // CHECK: rcp.rn.f32
    %one = arith.constant 1.0 : f32
    %result = tt.precise_divf %one, %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @precise_sqrt_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: precise_sqrt_f32(
    // CHECK: sqrt.rn.f32
    %result = tt.precise_sqrt %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @add_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: add_bf16
    // SM80-COUNT-8: fma.rn.bf16
    // SM90-COUNT-8: add.rn.bf16
    %0 = arith.addf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @sub_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: sub_bf16
    // SM80-COUNT-8: fma.rn.bf16
    // SM90-COUNT-8: sub.rn.bf16
    %0 = arith.subf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @mul_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>, %arg1: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: mul_bf16
    // SM80-COUNT-8: fma.rn.bf16
    // SM90-COUNT-8: mul.rn.bf16
    %0 = arith.mulf %arg0, %arg1 : tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @sitofp_s8_to_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xi8, #blocked>) {
    // CHECK-LABEL: sitofp_s8_to_bf16
    // SM80: cvt.rn.f32.s8
    // CHECK: prmt.b32
    // SM90: sub.rn.bf16x2
    // SM100: sub.rn.bf16x2
    // VEC80-LABEL: llvm.func @sitofp_s8_to_bf16
    // VEC80-NOT: llvm.fsub {{.*}} : vector<4xbf16>
    // VEC90-LABEL: llvm.func @sitofp_s8_to_bf16
    // VEC90: llvm.fsub {{.*}} : vector<4xbf16>
    // VEC100-LABEL: llvm.func @sitofp_s8_to_bf16
    // VEC100: llvm.fsub {{.*}} : vector<4xbf16>
    %0 = arith.sitofp %arg0 : tensor<256xi8, #blocked> to tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_bf16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: extf_bf16
    // CHECK-COUNT-8: cvt.f32.bf16
    %0 = arith.extf %arg0 : tensor<256xbf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_bf16
    // CHECK-COUNT-4: cvt.rn.bf16x2.f32
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xbf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  tt.func public @extf_f16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf16, #blocked>) {
    // CHECK-LABEL: extf_f16
    // CHECK-COUNT-8: cvt.f32.f16
    %0 = arith.extf %arg0 : tensor<256xf16, #blocked> to tensor<256xf32, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @truncf_f16(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg0: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: truncf_f16
    // CHECK-COUNT-4: cvt.rn.f16x2.f32
    %0 = arith.truncf %arg0 : tensor<256xf32, #blocked> to tensor<256xf16, #blocked>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %2 = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %3, %0 : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry fp32_to_bf16_rounding(
  // CHECK-DAG: cvt.rn.bf16.f32
  // CHECK-DAG: cvt.rz.bf16.f32
  tt.func public @fp32_to_bf16_rounding(%rn_out: !tt.ptr<bf16>, %rz_out: !tt.ptr<bf16>, %arg: f32) {
    %rn = tt.fp_to_fp %arg, rounding = rtne : f32 -> bf16
    %rz = tt.fp_to_fp %arg, rounding = rtz : f32 -> bf16
    tt.store %rn_out, %rn : !tt.ptr<bf16>
    tt.store %rz_out, %rz : !tt.ptr<bf16>
    tt.return
  }

  // CHECK-LABEL: .entry fp32_to_fp16_rounding(
  // CHECK-DAG: cvt.rn.f16.f32
  // CHECK-DAG: cvt.rz.f16.f32
  tt.func public @fp32_to_fp16_rounding(%rn_out: !tt.ptr<f16>, %rz_out: !tt.ptr<f16>, %arg: f32) {
    %rn = tt.fp_to_fp %arg, rounding = rtne : f32 -> f16
    %rz = tt.fp_to_fp %arg, rounding = rtz : f32 -> f16
    tt.store %rn_out, %rn : !tt.ptr<f16>
    tt.store %rz_out, %rz : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: .entry precise_divf(
  // CHECK-DAG: div.rn.f32
  // CHECK-DAG: rcp.rn.f32
  tt.func public @precise_divf(%out: !tt.ptr<f32>, %rcp_out: !tt.ptr<f32>, %x: f32, %y: f32) {
    %one = arith.constant 1.0 : f32
    %div = tt.precise_divf %x, %y : f32
    %rcp = tt.precise_divf %one, %y : f32
    tt.store %out, %div : !tt.ptr<f32>
    tt.store %rcp_out, %rcp : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: .entry precise_divf_f64(
  // CHECK-DAG: div.rn.f64
  // CHECK-DAG: rcp.rn.f64
  tt.func public @precise_divf_f64(%out: !tt.ptr<f64>, %rcp_out: !tt.ptr<f64>, %x: f64, %y: f64) {
    %one = arith.constant 1.0 : f64
    %div = tt.precise_divf %x, %y : f64
    %rcp = tt.precise_divf %one, %y : f64
    tt.store %out, %div : !tt.ptr<f64>
    tt.store %rcp_out, %rcp : !tt.ptr<f64>
    tt.return
  }

  // CHECK-LABEL: atomic_poll_relaxed_gpu
  // CHECK: ld.relaxed.gpu.global.b32
  // CHECK-NOT: fence.acquire
  tt.func public @atomic_poll_relaxed_gpu(%ptr: !tt.ptr<i32>, %expected: i32, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll relaxed, gpu, %ptr, %expected {allocation.offset = 0 : i32} : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: atomic_poll_acquire_cta
  // CHECK: ld.relaxed.cta.global.b32
  // SM80: fence.acq_rel.cta
  // SM90: fence.acq_rel.cta
  // SM100: fence.acquire.cta
  tt.func public @atomic_poll_acquire_cta(%ptr: !tt.ptr<i32>, %expected: i32, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll acquire, cta, %ptr, %expected {allocation.offset = 0 : i32} : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: atomic_poll_acquire_gpu
  // CHECK: ld.relaxed.gpu.global.b32
  // SM80: fence.acq_rel.gpu
  // SM90: fence.acq_rel.gpu
  // SM100: fence.acquire.gpu
  tt.func public @atomic_poll_acquire_gpu(%ptr: !tt.ptr<i32>, %expected: i32, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected {allocation.offset = 0 : i32} : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: atomic_poll_acquire_sys
  // CHECK: ld.relaxed.sys.global.b32
  // SM80: fence.acq_rel.sys
  // SM90: fence.acq_rel.sys
  // SM100: fence.acquire.sys
  tt.func public @atomic_poll_acquire_sys(%ptr: !tt.ptr<i32>, %expected: i32, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected {allocation.offset = 0 : i32} : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: reduce_f16_store
  // SM80-NOT: add.rn.f16x2
  // SM90: add.rn.f16x2
  // SM100: add.rn.f16x2
  // VEC80-LABEL: llvm.func @reduce_f16_store
  // VEC80-NOT: llvm.fadd {{.*}} : vector<2xf16>
  // VEC90-LABEL: llvm.func @reduce_f16_store
  // VEC90: llvm.fadd {{.*}} : vector<2xf16>
  // VEC100-LABEL: llvm.func @reduce_f16_store
  // VEC100: llvm.fadd {{.*}} : vector<2xf16>
  tt.func public @reduce_f16_store(%out: !tt.ptr<f16>, %arg0: tensor<1x256xf16, #blocked_reduce>) {
    %r = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f16, %b: f16):
      %sum = arith.addf %a, %b : f16
      tt.reduce.return %sum : f16
    }) {allocation.offset = 0 : i32} : (tensor<1x256xf16, #blocked_reduce>) -> tensor<1xf16, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    %ptr = tt.splat %out : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.store %ptr, %r : tensor<1x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // CHECK-LABEL: reduce_f32_store
  // VEC80-LABEL: llvm.func @reduce_f32_store
  // VEC80-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // VEC90-LABEL: llvm.func @reduce_f32_store
  // VEC90-NOT: llvm.fadd {{.*}} : vector<2xf32>
  // VEC100-LABEL: llvm.func @reduce_f32_store
  // VEC100: llvm.fadd {{.*}} : vector<2xf32>
  tt.func public @reduce_f32_store(%out: !tt.ptr<f32>, %arg0: tensor<1x256xf32, #blocked_reduce>) {
    %r = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %sum = arith.addf %a, %b : f32
      tt.reduce.return %sum : f32
    }) {allocation.offset = 0 : i32} : (tensor<1x256xf32, #blocked_reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    %ptr = tt.splat %out : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.store %ptr, %r : tensor<1x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
    tt.return
  }

  // CHECK-LABEL: .visible .entry shuffle_i1_no_mask(
  tt.func public @shuffle_i1_no_mask(%arg: tensor<64xi1, #narrow_src>, %out: tensor<64x!tt.ptr<i32>, #narrow_dst>) {
    // CHECK: shfl.sync.idx.b32 [[VALUE:%r[0-9]+]],
    // CHECK-NOT: and.b32
    // CHECK: st.global.b32 {{.*}}, { [[VALUE]] };
    %0 = ttg.convert_layout %arg : tensor<64xi1, #narrow_src> -> tensor<64xi1, #narrow_dst>
    %1 = arith.extui %0 : tensor<64xi1, #narrow_dst> to tensor<64xi32, #narrow_dst>
    tt.store %out, %1 : tensor<64x!tt.ptr<i32>, #narrow_dst>
    tt.return
  }
  // CHECK-LABEL: .visible .entry shuffle_i16_no_mask(
  tt.func public @shuffle_i16_no_mask(%arg: tensor<64xi16, #narrow_src>, %out: tensor<64x!tt.ptr<i32>, #narrow_dst>) {
    // CHECK: shfl.sync.idx.b32 [[VALUE:%r[0-9]+]],
    // CHECK-NOT: and.b32
    // CHECK: st.global.b32 {{.*}}, { [[VALUE]] };
    %0 = ttg.convert_layout %arg : tensor<64xi16, #narrow_src> -> tensor<64xi16, #narrow_dst>
    %1 = arith.extui %0 : tensor<64xi16, #narrow_dst> to tensor<64xi32, #narrow_dst>
    tt.store %out, %1 : tensor<64x!tt.ptr<i32>, #narrow_dst>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f32_to_f16_rtne(
  // CHECK: cvt.rn.f16.f32
  tt.func public @fp_trunc_f32_to_f16_rtne(%out: !tt.ptr<f16>, %arg: f32) {
    %result = tt.fp_to_fp %arg, rounding = rtne : f32 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f32_to_f16_rtz(
  // CHECK: cvt.rz.f16.f32
  tt.func public @fp_trunc_f32_to_f16_rtz(%out: !tt.ptr<f16>, %arg: f32) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f32 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f32_to_bf16_rtne(
  // CHECK: cvt.rn.bf16.f32
  tt.func public @fp_trunc_f32_to_bf16_rtne(%out: !tt.ptr<bf16>, %arg: f32) {
    %result = tt.fp_to_fp %arg, rounding = rtne : f32 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f32_to_bf16_rtz(
  // CHECK: cvt.rz.bf16.f32
  tt.func public @fp_trunc_f32_to_bf16_rtz(%out: !tt.ptr<bf16>, %arg: f32) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f32 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f64_to_f32_rtne(
  // CHECK: cvt.rn.f32.f64
  tt.func public @fp_trunc_f64_to_f32_rtne(%out: !tt.ptr<f32>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtne : f64 -> f32
    tt.store %out, %result : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f64_to_f32_rtz(
  // CHECK: cvt.rz.f32.f64
  tt.func public @fp_trunc_f64_to_f32_rtz(%out: !tt.ptr<f32>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f64 -> f32
    tt.store %out, %result : !tt.ptr<f32>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f64_to_f16_rtne(
  // CHECK: cvt.rn.f16.f64
  tt.func public @fp_trunc_f64_to_f16_rtne(%out: !tt.ptr<f16>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtne : f64 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp_trunc_f64_to_f16_rtz(
  // CHECK: cvt.rz.f16.f64
  tt.func public @fp_trunc_f64_to_f16_rtz(%out: !tt.ptr<f16>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f64 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp_convert_f16_to_bf16_default(
  // SM80: cvt.f32.f16
  // SM80: cvt.rn.bf16.f32
  // SM90: cvt.rn.bf16.f16
  // SM100: cvt.rn.bf16.f16
  tt.func public @fp_convert_f16_to_bf16_default(%out: !tt.ptr<bf16>, %arg: f16) {
    %result = tt.fp_to_fp %arg : f16 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }

  // CHECK-LABEL: fp_convert_f16_to_bf16_rtz(
  // SM80: cvt.f32.f16
  // SM80: cvt.rz.bf16.f32
  // SM90: cvt.rz.bf16.f16
  // SM100: cvt.rz.bf16.f16
  tt.func public @fp_convert_f16_to_bf16_rtz(%out: !tt.ptr<bf16>, %arg: f16) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f16 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }

  // CHECK-LABEL: fp_convert_bf16_to_f16_default(
  // SM80: cvt.f32.bf16
  // SM80: cvt.rn.f16.f32
  // SM90: cvt.rn.f16.bf16
  // SM100: cvt.rn.f16.bf16
  tt.func public @fp_convert_bf16_to_f16_default(%out: !tt.ptr<f16>, %arg: bf16) {
    %result = tt.fp_to_fp %arg : bf16 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp_convert_bf16_to_f16_rtz(
  // SM80: cvt.f32.bf16
  // SM80: cvt.rz.f16.f32
  // SM90: cvt.rz.f16.bf16
  // SM100: cvt.rz.f16.bf16
  tt.func public @fp_convert_bf16_to_f16_rtz(%out: !tt.ptr<f16>, %arg: bf16) {
    %result = tt.fp_to_fp %arg, rounding = rtz : bf16 -> f16
    tt.store %out, %result : !tt.ptr<f16>
    tt.return
  }

  // CHECK-LABEL: fp32_to_fp8_rtz_intermediate(
  // CHECK: cvt.rz.f16.f32
  tt.func public @fp32_to_fp8_rtz_intermediate(%out: !tt.ptr<f8E5M2>, %arg: f32) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f32 -> f8E5M2
    tt.store %out, %result : !tt.ptr<f8E5M2>
    tt.return
  }

  // CHECK-LABEL: bf16_to_fp8_rtne_intermediate(
  // SM80: cvt.rn.f16x2.f32
  // SM80: min.f16x2
  // SM80: prmt.b32 {{.*}}, 0x7531;
  // SM90: cvt.rn.satfinite.e5m2x2.f32
  // SM100: cvt.rn.satfinite.e5m2x2.f32
  tt.func public @bf16_to_fp8_rtne_intermediate(%out: tensor<256x!tt.ptr<f8E5M2>, #blocked>, %arg: tensor<256xbf16, #blocked>) {
    %result = tt.fp_to_fp %arg, rounding = rtne : tensor<256xbf16, #blocked> -> tensor<256xf8E5M2, #blocked>
    tt.store %out, %result : tensor<256x!tt.ptr<f8E5M2>, #blocked>
    tt.return
  }
}

//--- f64_bf16.mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32} {
  // F64BF16-LABEL: fp_trunc_f64_to_bf16_rtne(
  // F64BF16: cvt.rn.bf16.f64
  tt.func public @fp_trunc_f64_to_bf16_rtne(%out: !tt.ptr<bf16>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtne : f64 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }

  // F64BF16-LABEL: fp_trunc_f64_to_bf16_rtz(
  // F64BF16: cvt.rz.bf16.f64
  tt.func public @fp_trunc_f64_to_bf16_rtz(%out: !tt.ptr<bf16>, %arg: f64) {
    %result = tt.fp_to_fp %arg, rounding = rtz : f64 -> bf16
    tt.store %out, %result : !tt.ptr<bf16>
    tt.return
  }
}

//--- fp4.mlir
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // FP4-LABEL: .visible .entry extern_fp4_pack(
  // FP4: cvt.rn.satfinite.e2m1x2.f32
  // FP4: st.global.b16
  tt.func public @extern_fp4_pack(%hi: f32, %lo: f32, %out: !tt.ptr<i16>) {
    %false = arith.constant false
    %packed = tt.extern_elementwise %hi, %lo, %false {libname = "", libpath = "", pure = true, symbol = "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"} : (f32, f32, i1) -> i16
    tt.store %out, %packed : !tt.ptr<i16>
    tt.return
  }
}
