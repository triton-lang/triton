// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_90 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM90 --dump-input-context=20 %s
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=80 ptx-version=83' --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_80 -mattr=+ptx83 | FileCheck --check-prefixes CHECK,SM80 --dump-input-context=20 %s
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=100 ptx-version=87' --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' --convert-nv-gpu-to-llvm | mlir-translate --mlir-to-llvmir | opt -O3 -S | llc -mtriple nvptx64-nvidia-cuda -mcpu=sm_100 -mattr=+ptx87 | FileCheck --check-prefixes CHECK,SM100 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=83' -cse | FileCheck --check-prefix=VEC80 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=83' -cse | FileCheck --check-prefix=VEC90 --dump-input-context=20 %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=87' -cse | FileCheck --check-prefix=VEC100 --dump-input-context=20 %s


#blocked = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#packed_fp4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#byte_pairs = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 2], order = [1, 0]}>
#narrow_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#narrow_dst = #ttg.linear<{register = [], lane = [[2], [1], [4], [8], [16]], warp = [[32]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: .entry load_store_byte_pairs(
  // CHECK-NOT: prmt.b32
  // CHECK: ld.global.b16
  // CHECK-NOT: prmt.b32
  // CHECK: st.global.b16
  // CHECK-NOT: prmt.b32
  // CHECK: ret;
  tt.func public @load_store_byte_pairs(%input: !tt.ptr<i8> {tt.divisibility = 2 : i32}, %output: !tt.ptr<i8> {tt.divisibility = 2 : i32}) {
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #byte_pairs>
    %in = tt.splat %input : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>, #byte_pairs>
    %out = tt.splat %output : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>, #byte_pairs>
    %src = tt.addptr %in, %offsets : tensor<128x!tt.ptr<i8>, #byte_pairs>, tensor<128xi32, #byte_pairs>
    %dst = tt.addptr %out, %offsets : tensor<128x!tt.ptr<i8>, #byte_pairs>, tensor<128xi32, #byte_pairs>
    %values = tt.load %src : tensor<128x!tt.ptr<i8>, #byte_pairs>
    tt.store %dst, %values : tensor<128x!tt.ptr<i8>, #byte_pairs>
    tt.return
  }

  // CHECK-LABEL: .entry load_store_byte_pairs_volatile(
  // CHECK-NOT: prmt.b32
  // CHECK: ld.volatile.global.b16
  // CHECK-NOT: prmt.b32
  // CHECK: st.global.b16
  // CHECK-NOT: prmt.b32
  // CHECK: ret;
  tt.func public @load_store_byte_pairs_volatile(%input: !tt.ptr<i8> {tt.divisibility = 2 : i32}, %output: !tt.ptr<i8> {tt.divisibility = 2 : i32}) {
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #byte_pairs>
    %in = tt.splat %input : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>, #byte_pairs>
    %out = tt.splat %output : !tt.ptr<i8> -> tensor<128x!tt.ptr<i8>, #byte_pairs>
    %src = tt.addptr %in, %offsets : tensor<128x!tt.ptr<i8>, #byte_pairs>, tensor<128xi32, #byte_pairs>
    %dst = tt.addptr %out, %offsets : tensor<128x!tt.ptr<i8>, #byte_pairs>, tensor<128xi32, #byte_pairs>
    %values = tt.load %src {isVolatile = true} : tensor<128x!tt.ptr<i8>, #byte_pairs>
    tt.store %dst, %values : tensor<128x!tt.ptr<i8>, #byte_pairs>
    tt.return
  }

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

  tt.func public @sqrt_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: .entry sqrt_f32(
    // CHECK: sqrt.approx.f32 %r{{[0-9]+}}, %r{{[0-9]+}};
    %result = math.sqrt %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @sqrt_f64(%ptr: !tt.ptr<f64>, %arg: f64) {
    // CHECK-LABEL: .entry sqrt_f64(
    // CHECK: sqrt.rn.f64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
    %result = math.sqrt %arg : f64
    tt.store %ptr, %result : !tt.ptr<f64>
    tt.return
  }

  tt.func public @rsqrt_f32(%ptr: !tt.ptr<f32>, %arg: f32) {
    // CHECK-LABEL: .entry rsqrt_f32(
    // CHECK: rsqrt.approx.f32 %r{{[0-9]+}}, %r{{[0-9]+}};
    %result = math.rsqrt %arg : f32
    tt.store %ptr, %result : !tt.ptr<f32>
    tt.return
  }

  tt.func public @rsqrt_f64(%ptr: !tt.ptr<f64>, %arg: f64) {
    // CHECK-LABEL: .entry rsqrt_f64(
    // CHECK: rsqrt.approx.f64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
    %result = math.rsqrt %arg : f64
    tt.store %ptr, %result : !tt.ptr<f64>
    tt.return
  }

  tt.func public @sqrt_tensor_f32(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %arg: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: .entry sqrt_tensor_f32(
    // CHECK-COUNT-8: sqrt.approx.f32 %r{{[0-9]+}}, %r{{[0-9]+}};
    %result = math.sqrt %arg : tensor<256xf32, #blocked>
    tt.store %ptr, %result : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @sqrt_tensor_f64(%ptr: tensor<256x!tt.ptr<f64>, #blocked>, %arg: tensor<256xf64, #blocked>) {
    // CHECK-LABEL: .entry sqrt_tensor_f64(
    // CHECK-COUNT-8: sqrt.rn.f64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
    %result = math.sqrt %arg : tensor<256xf64, #blocked>
    tt.store %ptr, %result : tensor<256x!tt.ptr<f64>, #blocked>
    tt.return
  }

  tt.func public @rsqrt_tensor_f32(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %arg: tensor<256xf32, #blocked>) {
    // CHECK-LABEL: .entry rsqrt_tensor_f32(
    // CHECK-COUNT-8: rsqrt.approx.f32 %r{{[0-9]+}}, %r{{[0-9]+}};
    %result = math.rsqrt %arg : tensor<256xf32, #blocked>
    tt.store %ptr, %result : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  tt.func public @rsqrt_tensor_f64(%ptr: tensor<256x!tt.ptr<f64>, #blocked>, %arg: tensor<256xf64, #blocked>) {
    // CHECK-LABEL: .entry rsqrt_tensor_f64(
    // CHECK-COUNT-8: rsqrt.approx.f64 %rd{{[0-9]+}}, %rd{{[0-9]+}};
    %result = math.rsqrt %arg : tensor<256xf64, #blocked>
    tt.store %ptr, %result : tensor<256x!tt.ptr<f64>, #blocked>
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
    // SM80: cvt.rn.f32.s16
    // SM80: mov.b32 {_,
    // SM80-COUNT-4: mov.b32 {{%r[0-9]+}}, {
    // SM90: prmt.b32
    // SM90: sub.rn.bf16x2
    // SM100: prmt.b32
    // SM100: sub.rn.bf16x2
    // VEC80-LABEL: llvm.func @sitofp_s8_to_bf16
    // VEC80: llvm.sitofp {{.*}} : vector<4xi8> to vector<4xf32>
    // VEC80: llvm.shufflevector {{.*}} [1, 3, 5, 7] : vector<8xi16>
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

  tt.func public @scalar_s8_to_bf16(%ptr: !tt.ptr<bf16>, %input: i8) {
    // CHECK-LABEL: scalar_s8_to_bf16
    // VEC80-LABEL: llvm.func @scalar_s8_to_bf16
    // VEC80: llvm.sitofp {{.*}} : i8 to bf16
    // VEC90-LABEL: llvm.func @scalar_s8_to_bf16
    // VEC90: llvm.sitofp {{.*}} : i8 to f32
    // VEC90: llvm.lshr
    // VEC100-LABEL: llvm.func @scalar_s8_to_bf16
    // VEC100: llvm.sitofp {{.*}} : i8 to f32
    // VEC100: llvm.lshr
    %result = arith.sitofp %input : i8 to bf16
    tt.store %ptr, %result : !tt.ptr<bf16>
    tt.return
  }

  tt.func public @i1_to_bf16(%signed_ptr: !tt.ptr<bf16>, %unsigned_ptr: !tt.ptr<bf16>, %input: i1) {
    // CHECK-LABEL: i1_to_bf16
    // VEC80-LABEL: llvm.func @i1_to_bf16
    // VEC80: llvm.sitofp {{.*}} : i1 to bf16
    // VEC80: llvm.uitofp {{.*}} : i1 to bf16
    // VEC90-LABEL: llvm.func @i1_to_bf16
    // VEC90-DAG: %[[NEG:.*]] = llvm.mlir.constant(-1.000000e+00 : bf16)
    // VEC90-DAG: %[[POS:.*]] = llvm.mlir.constant(1.000000e+00 : bf16)
    // VEC90-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : bf16)
    // VEC90: llvm.select {{.*}}, %[[NEG]], %[[ZERO]] : i1, bf16
    // VEC90: llvm.select {{.*}}, %[[POS]], %[[ZERO]] : i1, bf16
    // VEC100-LABEL: llvm.func @i1_to_bf16
    // VEC100-DAG: %[[NEG:.*]] = llvm.mlir.constant(-1.000000e+00 : bf16)
    // VEC100-DAG: %[[POS:.*]] = llvm.mlir.constant(1.000000e+00 : bf16)
    // VEC100-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0.000000e+00 : bf16)
    // VEC100: llvm.select {{.*}}, %[[NEG]], %[[ZERO]] : i1, bf16
    // VEC100: llvm.select {{.*}}, %[[POS]], %[[ZERO]] : i1, bf16
    %signed = arith.sitofp %input : i1 to bf16
    %unsigned = arith.uitofp %input : i1 to bf16
    tt.store %signed_ptr, %signed : !tt.ptr<bf16>
    tt.store %unsigned_ptr, %unsigned : !tt.ptr<bf16>
    tt.return
  }

  tt.func public @extf_bf16(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg0: tensor<256xbf16, #blocked>) {
    // CHECK-LABEL: extf_bf16
    // SM80-NOT: cvt.f32.bf16
    // SM80: shl.b32
    // SM80-NOT: cvt.f32.bf16
    // SM90-NOT: cvt.f32.bf16
    // SM90: shl.b32
    // SM90-NOT: cvt.f32.bf16
    // SM100-COUNT-8: cvt.f32.bf16
    // VEC80-LABEL: llvm.func @extf_bf16
    // VEC80: %[[BITS:.*]] = llvm.bitcast {{.*}} : bf16 to i16
    // VEC80: %[[WIDE:.*]] = llvm.zext %[[BITS]] : i16 to i32
    // VEC80: llvm.shl %[[WIDE]], {{.*}} : i32
    // VEC80-NOT: llvm.fpext
    // VEC80: llvm.return
    // VEC90-LABEL: llvm.func @extf_bf16
    // VEC90: %[[BITS:.*]] = llvm.bitcast {{.*}} : bf16 to i16
    // VEC90: %[[WIDE:.*]] = llvm.zext %[[BITS]] : i16 to i32
    // VEC90: llvm.shl %[[WIDE]], {{.*}} : i32
    // VEC90-NOT: llvm.fpext
    // VEC90: llvm.return
    // VEC100-LABEL: llvm.func @extf_bf16
    // VEC100: llvm.fpext {{.*}} : bf16 to f32
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
    // CHECK: st.global.b32 {{.*}}, [[VALUE]];
    %0 = ttg.convert_layout %arg : tensor<64xi1, #narrow_src> -> tensor<64xi1, #narrow_dst>
    %1 = arith.extui %0 : tensor<64xi1, #narrow_dst> to tensor<64xi32, #narrow_dst>
    tt.store %out, %1 : tensor<64x!tt.ptr<i32>, #narrow_dst>
    tt.return
  }
  // CHECK-LABEL: .visible .entry shuffle_i16_no_mask(
  tt.func public @shuffle_i16_no_mask(%arg: tensor<64xi16, #narrow_src>, %out: tensor<64x!tt.ptr<i32>, #narrow_dst>) {
    // CHECK: shfl.sync.idx.b32 [[VALUE:%r[0-9]+]],
    // CHECK-NOT: and.b32
    // CHECK: st.global.b32 {{.*}}, [[VALUE]];
    %0 = ttg.convert_layout %arg : tensor<64xi16, #narrow_src> -> tensor<64xi16, #narrow_dst>
    %1 = arith.extui %0 : tensor<64xi16, #narrow_dst> to tensor<64xi32, #narrow_dst>
    tt.store %out, %1 : tensor<64x!tt.ptr<i32>, #narrow_dst>
    tt.return
  }

  // CHECK-LABEL: .entry fp4_to_bf16(
  // CHECK: and.b32 {{.*}}, 2004318071;
  // CHECK: prmt.b32
  // CHECK: prmt.b32
  // CHECK: prmt.b32
  // CHECK: st.global
  // VEC80-LABEL: llvm.func @fp4_to_bf16
  // VEC80: llvm.inline_asm
  // VEC80-SAME: prmt.b32
  // VEC90-LABEL: llvm.func @fp4_to_bf16
  // VEC90: llvm.inline_asm
  // VEC90-SAME: prmt.b32
  // VEC100-LABEL: llvm.func @fp4_to_bf16
  // VEC100: llvm.inline_asm
  // VEC100-SAME: prmt.b32
  tt.func public @fp4_to_bf16(%ptr: tensor<512x!tt.ptr<bf16>, #blocked>, %arg: tensor<256xi8, #packed_fp4>) {
    %result = ttg.fp4_to_fp %arg {axis = 0 : i32} : tensor<256xi8, #packed_fp4> -> tensor<512xbf16, #blocked>
    tt.store %ptr, %result : tensor<512x!tt.ptr<bf16>, #blocked>
    tt.return
  }


  // CHECK-LABEL: .entry fp8e5_to_fp16(
  // SM80-DAG: prmt.b32 {{.*}}, 0, {{.*}}, 0x5040U;
  // SM80-DAG: prmt.b32 {{.*}}, 0, {{.*}}, 0x7060U;
  // SM90: cvt.rn.f16x2.e5m2x2
  // SM100: cvt.rn.f16x2.e5m2x2
  tt.func public @fp8e5_to_fp16(%ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %input: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %input_base = tt.splat %input : !tt.ptr<f8E5M2> -> tensor<256x!tt.ptr<f8E5M2>, #blocked>
    %input_ptrs = tt.addptr %input_base, %offsets : tensor<256x!tt.ptr<f8E5M2>, #blocked>, tensor<256xi32, #blocked>
    %arg = tt.load %input_ptrs : tensor<256x!tt.ptr<f8E5M2>, #blocked>
    %result = tt.fp_to_fp %arg : tensor<256xf8E5M2, #blocked> -> tensor<256xf16, #blocked>
    %base = tt.splat %ptr : !tt.ptr<f16> -> tensor<256x!tt.ptr<f16>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %ptrs, %result : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry packed_arith_f32(
  // SM100: add{{(\.rn)?}}.f32x2
  // SM100: sub{{(\.rn)?}}.f32x2
  // SM100: mul{{(\.rn)?}}.f32x2
  // SM100: fma.rn.f32x2
  tt.func public @packed_arith_f32(%ptr: tensor<256x!tt.ptr<f32>, #blocked>, %a: tensor<256xf32, #blocked>, %b: tensor<256xf32, #blocked>, %c: tensor<256xf32, #blocked>) {
    %add = ttng.packed_arith add %a, %b : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    %sub = ttng.packed_arith sub %add, %b : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    %mul = ttng.packed_arith mul %sub, %b : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    %fma = ttng.packed_arith fma %mul, %b, %c : (tensor<256xf32, #blocked>, tensor<256xf32, #blocked>, tensor<256xf32, #blocked>) -> tensor<256xf32, #blocked>
    tt.store %ptr, %fma : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry packed_arith_f16(
  // CHECK: add{{(\.rn)?}}.f16x2
  // CHECK: sub{{(\.rn)?}}.f16x2
  // CHECK: mul{{(\.rn)?}}.f16x2
  // CHECK: fma.rn.f16x2
  // CHECK: min.f16x2
  // CHECK: max.f16x2
  tt.func public @packed_arith_f16(%ptr: tensor<256x!tt.ptr<f16>, #blocked>, %a: tensor<256xf16, #blocked>, %b: tensor<256xf16, #blocked>, %c: tensor<256xf16, #blocked>) {
    %add = ttng.packed_arith add %a, %b : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    %sub = ttng.packed_arith sub %add, %b : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    %mul = ttng.packed_arith mul %sub, %b : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    %fma = ttng.packed_arith fma %mul, %b, %c : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    %min = ttng.packed_arith min %fma, %b : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    %max = ttng.packed_arith max %min, %c : (tensor<256xf16, #blocked>, tensor<256xf16, #blocked>) -> tensor<256xf16, #blocked>
    tt.store %ptr, %max : tensor<256x!tt.ptr<f16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry packed_arith_bf16(
  // SM90: add{{(\.rn)?}}.bf16x2
  // SM90: sub{{(\.rn)?}}.bf16x2
  // SM90: mul{{(\.rn)?}}.bf16x2
  // SM90: fma.rn.bf16x2
  // SM90: min.bf16x2
  // SM90: max.bf16x2
  // SM100: add{{(\.rn)?}}.bf16x2
  // SM100: sub{{(\.rn)?}}.bf16x2
  // SM100: mul{{(\.rn)?}}.bf16x2
  // SM100: fma.rn.bf16x2
  // SM100: min.bf16x2
  // SM100: max.bf16x2
  tt.func public @packed_arith_bf16(%ptr: tensor<256x!tt.ptr<bf16>, #blocked>, %a: tensor<256xbf16, #blocked>, %b: tensor<256xbf16, #blocked>, %c: tensor<256xbf16, #blocked>) {
    %add = ttng.packed_arith add %a, %b : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    %sub = ttng.packed_arith sub %add, %b : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    %mul = ttng.packed_arith mul %sub, %b : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    %fma = ttng.packed_arith fma %mul, %b, %c : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    %min = ttng.packed_arith min %fma, %b : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    %max = ttng.packed_arith max %min, %c : (tensor<256xbf16, #blocked>, tensor<256xbf16, #blocked>) -> tensor<256xbf16, #blocked>
    tt.store %ptr, %max : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }


  // CHECK-LABEL: .entry fp8e5_to_bf16(
  // SM80: cvt.f32.f16
  // SM80: prmt.b32
  // SM90: cvt{{(\.rn)?}}.bf16.f16
  // SM100: cvt{{(\.rn)?}}.bf16.f16
  // VEC80-LABEL: llvm.func @fp8e5_to_bf16
  // VEC80: llvm.fpext {{.*}} : vector<4xf16> to vector<4xf32>
  // VEC80: llvm.shufflevector
  // VEC90-LABEL: llvm.func @fp8e5_to_bf16
  // VEC90: nvvm.convert.f8x2.to.f16x2
  // VEC100-LABEL: llvm.func @fp8e5_to_bf16
  // VEC100: nvvm.convert.f8x2.to.f16x2
  tt.func public @fp8e5_to_bf16(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg: tensor<256xf8E5M2, #blocked>) {
    %result = tt.fp_to_fp %arg : tensor<256xf8E5M2, #blocked> -> tensor<256xbf16, #blocked>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %ptr : !tt.ptr<bf16> -> tensor<256x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<bf16>, #blocked>, tensor<256xi32, #blocked>
    tt.store %ptrs, %result : tensor<256x!tt.ptr<bf16>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry fp16_to_fp8e5_rtne(
  // SM80: set.nan.f16x2.f16x2
  // SM80: min.f16x2
  // SM80: add.s32
  // SM80: prmt.b32
  // SM90: cvt.rn.satfinite.e5m2x2.f16x2
  // SM100: cvt.rn.satfinite.e5m2x2.f16x2
  // VEC80-LABEL: llvm.func @fp16_to_fp8e5_rtne
  // VEC80: llvm.call_intrinsic "llvm.minimumnum"
  // VEC80: llvm.add
  // VEC80: llvm.intr.copysign
  // VEC80: llvm.shufflevector
  tt.func public @fp16_to_fp8e5_rtne(%ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg: tensor<256xf16, #blocked>) {
    %result = tt.fp_to_fp %arg, rounding = rtne : tensor<256xf16, #blocked> -> tensor<256xf8E5M2, #blocked>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %ptr : !tt.ptr<f8E5M2> -> tensor<256x!tt.ptr<f8E5M2>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f8E5M2>, #blocked>, tensor<256xi32, #blocked>
    tt.store %ptrs, %result : tensor<256x!tt.ptr<f8E5M2>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry fp32_to_fp8e5_rtne(
  // SM80: add.s32 [[ROUND:%r[0-9]+]], [[BITS:%r[0-9]+]], 65535;
  // SM80-NEXT: and.b32 [[JAM:%r[0-9]+]], [[ROUND]], 65536;
  // SM80-NEXT: or.b32 {{%r[0-9]+}}, [[JAM]], [[BITS]];
  // SM80: cvt.rz.f16x2.f32
  // SM80: set.nan.f16x2.f16x2
  // SM80: min.f16x2
  // SM80: prmt.b32
  // SM90: cvt.rn.satfinite.e5m2x2.f32
  // SM100: cvt.rn.satfinite.e5m2x2.f32
  // VEC80-LABEL: llvm.func @fp32_to_fp8e5_rtne
  // VEC80: nvvm.convert.f32x2.to.f16x2 {{.*}}rnd = #nvvm.fp_rnd_mode<rz>
  // VEC80: llvm.call_intrinsic "llvm.minimumnum"
  // VEC80: llvm.intr.copysign
  tt.func public @fp32_to_fp8e5_rtne(%ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg: tensor<256xf32, #blocked>) {
    %result = tt.fp_to_fp %arg, rounding = rtne : tensor<256xf32, #blocked> -> tensor<256xf8E5M2, #blocked>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %ptr : !tt.ptr<f8E5M2> -> tensor<256x!tt.ptr<f8E5M2>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f8E5M2>, #blocked>, tensor<256xi32, #blocked>
    tt.store %ptrs, %result : tensor<256x!tt.ptr<f8E5M2>, #blocked>
    tt.return
  }

  // CHECK-LABEL: .entry fp32_to_fp8e5_rtz(
  // CHECK: cvt.rz.f16.f32
  // SM90: and.b32
  // SM100: and.b32
  // CHECK: prmt.b32 {{.*}}, {{(0x7531U?|29745)}};
  // VEC80-LABEL: llvm.func @fp32_to_fp8e5_rtz
  // VEC80: llvm.shufflevector {{.*}} [1, 3, 5, 7] : vector<8xi8>
  // VEC90-LABEL: llvm.func @fp32_to_fp8e5_rtz
  // VEC90: llvm.inline_asm
  // VEC90-SAME: and.b32
  // VEC90-SAME: prmt.b32
  // VEC100-LABEL: llvm.func @fp32_to_fp8e5_rtz
  // VEC100: llvm.inline_asm
  // VEC100-SAME: and.b32
  // VEC100-SAME: prmt.b32
  tt.func public @fp32_to_fp8e5_rtz(%ptr: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg: tensor<256xf32, #blocked>) {
    %result = tt.fp_to_fp %arg, rounding = rtz : tensor<256xf32, #blocked> -> tensor<256xf8E5M2, #blocked>
    %offsets = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked>
    %base = tt.splat %ptr : !tt.ptr<f8E5M2> -> tensor<256x!tt.ptr<f8E5M2>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<256x!tt.ptr<f8E5M2>, #blocked>, tensor<256xi32, #blocked>
    tt.store %ptrs, %result : tensor<256x!tt.ptr<f8E5M2>, #blocked>
    tt.return
  }
}
