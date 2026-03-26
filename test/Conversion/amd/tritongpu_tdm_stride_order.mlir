// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 --verify-diagnostics | FileCheck %s

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_load_runtime_strides(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride0: i64, %stride1: i64) {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    // expected-error @+2 {{requires at least one dimension to have stride 1}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%stride0, %stride1] : <f16>, <64x64xf16, #shared>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_1x1_tensor(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %stride0: i64, %stride1: i64) {
    %c_shape = arith.constant 1 : i32
    %c_stride1 = arith.constant 1 : i64
    // expected-error @+2 {{requires at least one dimension to have stride 1}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%stride1, %stride1] : <f16>, <64x64xf16, #shared>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_wrong_stride_order(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %runtime_stride: i64) {
    %c_shape = arith.constant 128 : i32
    %c_stride1 = arith.constant 1 : i64
    // expected-error @+2 {{requires shared order [rank-2, rank-1, rank-3, rank-4, ..., 0] because dim[rank-2] has stride 1}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride1, %runtime_stride] : <f16>, <64x64xf16, #shared>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [0, 1], shape = [64, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_wrong_smem_order(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %runtime_stride: i64) {
    %c_shape = arith.constant 128 : i32
    %c_stride1 = arith.constant 1 : i64
    // expected-error @+2 {{requires shared order [rank-1, rank-2, ..., 0]}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%runtime_stride, %c_stride1] : <f16>, <64x64xf16, #shared>
    tt.return
  }
}

// -----

// Positive test case for 1x1x1 tensor
#shared = #ttg.padded_shared<[32:+4] {order = [0, 1, 2], shape = [1, 1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_1x1x1
  tt.func public @tdm_1x1x1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c_stride1 = arith.constant 1 : i64
    %c_shape = arith.constant 1 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape, %c_shape], [%c_stride1, %c_stride1, %c_stride1] : <f16>, <1x1x1xf16, #shared>
    tt.return
  }
}

// -----

// Positive test case for Xx1x1 tensor
#shared = #ttg.padded_shared<[32:+4] {order = [0, 1, 2], shape = [1, 1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_Xx1x1
  tt.func public @tdm_Xx1x1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %runtime_stride: i64) {
    %c_stride1 = arith.constant 1 : i64
    %c_shape = arith.constant 1 : i32
    %c_shape2 = arith.constant 128 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape2, %c_shape, %c_shape], [%runtime_stride, %c_stride1, %c_stride1] : <f16>, <1x1x1xf16, #shared>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [0, 1, 2], shape = [1, 1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_1xix1(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %runtime_stride: i64) {
    %c_stride1 = arith.constant 1 : i64
    %c_shape = arith.constant 1 : i32
    %c_shape2 = arith.constant 128 : i32
    // expected-error @+2 {{requires all stride 1 dimensions to be consecutive starting from the last dimension}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape2, %c_shape], [%c_stride1, %runtime_stride, %c_stride1] : <f16>, <1x1x1xf16, #shared>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [0, 1, 2], shape = [1, 1, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tdm_1x1xi(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %runtime_stride: i64) {
    %c_stride1 = arith.constant 1 : i64
    %c_shape = arith.constant 1 : i32
    %c_shape2 = arith.constant 128 : i32
    // expected-error @+2 {{requires all stride 1 dimensions to be consecutive starting from the last dimension}}
    // expected-error @+1 {{failed to legalize operation}}
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape, %c_shape2], [%c_stride1, %c_stride1, %runtime_stride] : <f16>, <1x1x1xf16, #shared>
    tt.return
  }
}
