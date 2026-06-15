// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_offsets_only
  // The offsets kwarg bumps global_addr by sum(offset[i]*stride[i]); the address
  // is advanced in place via a <2xi64> view of the descriptor -- a single i64
  // add, no per-dword decode/repack.
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK-DAG: llvm.sext{{.*}}i32 to i64
  // CHECK-DAG: llvm.zext{{.*}}i32 to i64
  // CHECK: llvm.mul{{.*}}: i64
  // The advance reads/writes lane 1 (= group0[2:3], the address) of the <2xi64>
  // view -- pin the lane index to 1 so a wrong-lane (pred/lds) regression fails.
  // CHECK: llvm.bitcast{{.*}}vector<4xi32> to vector<2xi64>
  // CHECK: %[[LANE0:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK: llvm.extractelement %{{.+}}[%[[LANE0]] : i32] : vector<2xi64>
  // CHECK: llvm.add{{.*}}: i64
  // CHECK: %[[LANE1:.+]] = llvm.mlir.constant(1 : i32)
  // CHECK: llvm.insertelement %{{.+}}, %{{.+}}[%[[LANE1]] : i32] : vector<2xi64>
  // CHECK: llvm.bitcast{{.*}}vector<2xi64> to vector<4xi32>
  // The old decode/repack of the address must be gone: no valid-bit mask
  // (0x7FFFFFFF = 2147483647) and no i64<->i32 split (trunc/lshr/shl).
  // CHECK-NOT: 2147483647
  // CHECK-NOT: llvm.trunc
  // CHECK-NOT: llvm.lshr
  // CHECK-NOT: llvm.shl
  tt.func public @update_tensor_descriptor_offsets_only(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dx: i32, %dy: i32)
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = amdg.update_tensor_descriptor %0 add_offsets = [%dx, %dy] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_bounds_only
  // The bounds kwarg writes tensor_dim (lo16/hi16 across group1[1..3]).
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.insertelement{{.*}}vector<8xi32>
  tt.func public @update_tensor_descriptor_bounds_only(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m: i32, %k: i32)
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = amdg.update_tensor_descriptor %0 set_bounds = [%m, %k] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_pred_only
  // The pred kwarg stamps into group0[0].
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.insertelement{{.*}}vector<4xi32>
  tt.func public @update_tensor_descriptor_pred_only(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %p: i32)
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = amdg.update_tensor_descriptor %0 pred = %p : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_prologue
  // The K-loop prologue pattern: position at first tile + set pred.
  // (The LDS destination and TDM barrier are op-level operands on the copy,
  // not descriptor fields.)
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.add{{.*}}: i64
  // CHECK: llvm.insertelement{{.*}}vector<4xi32>
  tt.func public @update_tensor_descriptor_prologue(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dx: i32, %dy: i32, %p: i32) -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = amdg.update_tensor_descriptor %0
            add_offsets = [%dx, %dy]
            pred = %p
            : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_clamp_bounds
  // clamp_bounds bumps global_addr (add_offsets) AND derives the OOB extent:
  // tensor_dim[d] = max(0, decoded_tensor_dim[d] - add_offsets[d]).  The clamp
  // is the shared signed form (max(cur-off, 0), forced to 0 when off<0), one
  // per descriptor dim (inner + outer).
  // CHECK-NOT: amdg.update_tensor_descriptor
  // global_addr bump (add_offsets):
  // CHECK-DAG: llvm.sext{{.*}}i32 to i64
  // CHECK-DAG: llvm.mul{{.*}}: i64
  // tensor_dim clamp, per dim (signed: off<0 check):
  // CHECK-COUNT-2: llvm.icmp "slt"
  // CHECK: llvm.select
  tt.func public @update_tensor_descriptor_clamp_bounds(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %dx: i32, %dy: i32)
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %1 = amdg.update_tensor_descriptor %0 add_offsets = [%dx, %dy] {clamp_bounds} : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

// 3D update lowers (update_tensor_descriptor is no longer 2D-only): add_offsets
// bumps global_addr across all dims; clamp_bounds derives the OOB extent per dim
// (signed clamp, one per dim).
#shared3d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_3d_clamp
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK-DAG: llvm.sext{{.*}}i32 to i64
  // Per-dim signed clamp: icmp slt + select, one per descriptor dim (the select
  // produces the clamped tensor_dim that is stamped back into group1/group2).
  // CHECK-COUNT-3: llvm.icmp "slt"
  // CHECK: llvm.select
  tt.func public @update_tensor_descriptor_3d_clamp(
      %desc: !tt.tensordesc<8x8x32xf16, #shared3d>, %x: i32, %y: i32, %z: i32)
      -> !tt.tensordesc<8x8x32xf16, #shared3d> {
    %0 = amdg.update_tensor_descriptor %desc add_offsets = [%x, %y, %z] {clamp_bounds} : !tt.tensordesc<8x8x32xf16, #shared3d>
    tt.return %0 : !tt.tensordesc<8x8x32xf16, #shared3d>
  }
}
