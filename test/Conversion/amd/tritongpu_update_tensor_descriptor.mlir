// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_offsets_only
  // The offsets kwarg bumps global_addr: offsets are sext'd (signed), stride
  // is zext'd (unsigned), multiplied in i64 (so offsets[0]*stride doesn't
  // overflow i32), added to global_addr, then re-packed into group0[2..3].
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK-DAG: llvm.sext{{.*}}i32 to i64
  // CHECK-DAG: llvm.zext{{.*}}i32 to i64
  // CHECK: llvm.mul{{.*}}: i64
  // CHECK: llvm.add{{.*}}: i64
  // CHECK: llvm.insertelement{{.*}}vector<4xi32>
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
  // CHECK-LABEL: update_tensor_descriptor_dest_only
  // The dest parameter ptrtoints the smem ptr and stamps into group0[1].
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.insertelement{{.*}}vector<4xi32>
  tt.func public @update_tensor_descriptor_dest_only(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32})
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %lds = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %1 = amdg.update_tensor_descriptor %0 dest = %lds : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> : !tt.tensordesc<64x64xf16, #shared>
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
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_barrier_only
  // The barrier kwarg ptrtoints the barrier ptr, shifts and stamps into
  // group1[1] lo-16 plus the enable bit (group1[0] bit 18).
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.insertelement{{.*}}vector<8xi32>
  tt.func public @update_tensor_descriptor_barrier_only(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32})
      -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %1 = amdg.update_tensor_descriptor %0 barrier = %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable> : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#shared_bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: update_tensor_descriptor_prologue
  // The K-loop prologue pattern: position at first tile + wire LDS + barrier + pred.
  // CHECK-NOT: amdg.update_tensor_descriptor
  // CHECK: llvm.ptrtoint
  // CHECK: llvm.add{{.*}}: i64
  // CHECK: llvm.insertelement{{.*}}vector<4xi32>
  // CHECK: llvm.insertelement{{.*}}vector<8xi32>
  tt.func public @update_tensor_descriptor_prologue(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %dx: i32, %dy: i32, %p: i32) -> !tt.tensordesc<64x64xf16, #shared> {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #shared>
    %lds = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
    %1 = amdg.update_tensor_descriptor %0
            add_offsets = [%dx, %dy]
            dest = %lds : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
            pred = %p
            barrier = %bar : !ttg.memdesc<1xi64, #shared_bar, #smem, mutable>
            : !tt.tensordesc<64x64xf16, #shared>
    tt.return %1 : !tt.tensordesc<64x64xf16, #shared>
  }
}
