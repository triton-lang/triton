// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' -cse | FileCheck %s --check-prefix=TERNARY

#linear = #ttg.linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_packed_reduce = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @reduce_linear_layout
tt.func private @reduce_linear_layout(%arg0: tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> {
  // CHECK-NEXT: [[SRC0:%.*]] = extractvalue {{.*}} %0, 0
  // CHECK-NEXT: [[SRC1:%.*]] = extractvalue {{.*}} %0, 1
  // CHECK-NEXT: [[SRC2:%.*]] = extractvalue {{.*}} %0, 2
  // CHECK-NEXT: [[SRC3:%.*]] = extractvalue {{.*}} %0, 3

  // The layout looks lke
  // [[  T0:0,  T32:0,   T0:1,  T32:1, ...
  // [   T4:0,  T36:0,   T4:1,  T36:1, ...
  // [   T0:2,  T32:2,   T0:3,  T32:3, ...
  // [   T4:2,  T36:2,   T4:3,  T36:3,
  // ...
  //
  // A reduction along axis=0 consists of adding registers (0, 2) and (1, 3)
  // before shuffling.
  //
  // Columns along axis=0 are contained within a warp, so reduction arcoss warps
  // is not needed.

  // Reduce within threads
  // CHECK: [[SUM0:%.*]] = add i32 [[SRC0]], [[SRC2]]
  // CHECK-NEXT: [[SUM1:%.*]] = add i32 [[SRC1]], [[SRC3]]

  // Reduce within warp.
  // CHECK-NEXT: [[W0:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SUM0]], i32 16, i32 31)
  // CHECK-NEXT: [[WSUM0:%.*]] = add i32 [[W0]], [[SUM0]]
  // CHECK-NEXT: [[W1:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM0]], i32 8, i32 31)
  // CHECK-NEXT: [[WSUM1:%.*]] = add i32 [[WSUM0]], [[W1]]
  // CHECK-NEXT: [[W2:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM1]], i32 4, i32 31)
  // CHECK-NEXT: [[WSUM2:%.*]] = add i32 [[WSUM1]], [[W2]]
  // CHECK-NEXT: [[W3:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM2]], i32 2, i32 31)
  // CHECK-NEXT: [[WSUM3:%.*]] = add i32 [[WSUM2]], [[W3]]

  // CHECK-NEXT: [[W4:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[SUM1]], i32 16, i32 31)
  // CHECK-NEXT: [[WSUM4:%.*]] = add i32 [[W4]], [[SUM1]]
  // CHECK-NEXT: [[W5:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM4]], i32 8, i32 31)
  // CHECK-NEXT: [[WSUM5:%.*]] = add i32 [[WSUM4]], [[W5]]
  // CHECK-NEXT: [[W6:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM5]], i32 4, i32 31)
  // CHECK-NEXT: [[WSUM6:%.*]] = add i32 [[WSUM5]], [[W6]]
  // CHECK-NEXT: [[W7:%.*]] = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 [[WSUM6]], i32 2, i32 31)
  // CHECK-NEXT: [[WSUM7:%.*]] = add i32 [[WSUM6]], [[W7]]

  // CHECK-NEXT: [[DST0:%.*]] = insertvalue { i32, i32 } undef, i32 [[WSUM3]], 0
  // CHECK-NEXT: [[DST1:%.*]] = insertvalue { i32, i32 } [[DST0]], i32 [[WSUM7]], 1

  %0 = "tt.reduce"(%arg0) ({
  ^bb0(%arg1: i32, %arg2: i32):
    %1 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %1 : i32
  }) {axis = 0 : i32} : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>

  // CHECK-NEXT: ret { i32, i32 } [[DST1]]
  tt.return %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
}

tt.func @anchor(%ptr: !llvm.ptr, %arg0: tensor<32x16xi32, #linear>) {
  %0 = tt.call @reduce_linear_layout(%arg0) : (tensor<32x16xi32, #linear>) -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
  %1 = builtin.unrealized_conversion_cast %0 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> to !llvm.struct<(i32, i32)>
  llvm.store volatile %1, %ptr : !llvm.struct<(i32, i32)>, !llvm.ptr
  tt.return
}

// TERNARY-LABEL: @reduce_maximum_f32
// TERNARY: %[[MAXIMUM_A:.*]] = llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: %[[MAXIMUM_B:.*]] = llvm.intr.maximum(%[[MAXIMUM_A]], %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: llvm.intr.maximum(%[[MAXIMUM_B]], %{{.*}}) : (f32, f32) -> f32
tt.func public @reduce_maximum_f32(%arg0: tensor<128x4xf32, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %maximum = arith.maximumf %a, %b : f32
    tt.reduce.return %maximum : f32
  }) : (tensor<128x4xf32, #blocked_reduce>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_minimum_f32
// TERNARY: %[[MINIMUM_A:.*]] = llvm.intr.minimum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: %[[MINIMUM_B:.*]] = llvm.intr.minimum(%[[MINIMUM_A]], %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: llvm.intr.minimum(%[[MINIMUM_B]], %{{.*}}) : (f32, f32) -> f32
tt.func public @reduce_minimum_f32(%arg0: tensor<128x4xf32, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %minimum = arith.minimumf %a, %b : f32
    tt.reduce.return %minimum : f32
  }) : (tensor<128x4xf32, #blocked_reduce>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_maxnum_f32
// TERNARY: %[[MAXNUM_A:.*]] = llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: %[[MAXNUM_B:.*]] = llvm.intr.maxnum(%[[MAXNUM_A]], %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: llvm.intr.maxnum(%[[MAXNUM_B]], %{{.*}}) : (f32, f32) -> f32
tt.func public @reduce_maxnum_f32(%arg0: tensor<128x4xf32, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %maximum = arith.maxnumf %a, %b : f32
    tt.reduce.return %maximum : f32
  }) : (tensor<128x4xf32, #blocked_reduce>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_minnum_f32
// TERNARY: %[[MINNUM_A:.*]] = llvm.intr.minnum(%{{.*}}, %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: %[[MINNUM_B:.*]] = llvm.intr.minnum(%[[MINNUM_A]], %{{.*}}) : (f32, f32) -> f32
// TERNARY-NEXT: llvm.intr.minnum(%[[MINNUM_B]], %{{.*}}) : (f32, f32) -> f32
tt.func public @reduce_minnum_f32(%arg0: tensor<128x4xf32, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %minimum = arith.minnumf %a, %b : f32
    tt.reduce.return %minimum : f32
  }) : (tensor<128x4xf32, #blocked_reduce>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_maxsi_i32
// TERNARY: %[[SMAX_A:.*]] = llvm.intr.smax(%{{.*}}, %{{.*}}) : (i32, i32) -> i32
// TERNARY-NEXT: %[[SMAX_B:.*]] = llvm.intr.smax(%[[SMAX_A]], %{{.*}}) : (i32, i32) -> i32
// TERNARY-NEXT: llvm.intr.smax(%[[SMAX_B]], %{{.*}}) : (i32, i32) -> i32
tt.func public @reduce_maxsi_i32(%arg0: tensor<128x4xi32, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %maximum = arith.maxsi %a, %b : i32
    tt.reduce.return %maximum : i32
  }) : (tensor<128x4xi32, #blocked_reduce>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_minui_i16
// TERNARY: %[[PACKED_UMIN_A:.*]] = llvm.intr.umin(%{{.*}}, %{{.*}}) : (vector<2xi16>, vector<2xi16>) -> vector<2xi16>
// TERNARY-NEXT: %[[PACKED_UMIN_B:.*]] = llvm.intr.umin(%[[PACKED_UMIN_A]], %{{.*}}) : (vector<2xi16>, vector<2xi16>) -> vector<2xi16>
// TERNARY-NEXT: llvm.intr.umin(%[[PACKED_UMIN_B]], %{{.*}}) : (vector<2xi16>, vector<2xi16>) -> vector<2xi16>
tt.func public @reduce_minui_i16(%arg0: tensor<128x8xi16, #blocked_packed_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i16, %b: i16):
    %minimum = arith.minui %a, %b : i16
    tt.reduce.return %minimum : i16
  }) : (tensor<128x8xi16, #blocked_packed_reduce>) -> tensor<128xi16, #ttg.slice<{dim = 1, parent = #blocked_packed_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_maxnum_f16
// TERNARY: %[[PACKED_MAXNUM_A:.*]] = llvm.intr.maxnum(%{{.*}}, %{{.*}}) : (vector<2xf16>, vector<2xf16>) -> vector<2xf16>
// TERNARY-NEXT: %[[PACKED_MAXNUM_B:.*]] = llvm.intr.maxnum(%[[PACKED_MAXNUM_A]], %{{.*}}) : (vector<2xf16>, vector<2xf16>) -> vector<2xf16>
// TERNARY-NEXT: llvm.intr.maxnum(%[[PACKED_MAXNUM_B]], %{{.*}}) : (vector<2xf16>, vector<2xf16>) -> vector<2xf16>
tt.func public @reduce_maxnum_f16(%arg0: tensor<128x8xf16, #blocked_packed_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f16, %b: f16):
    %maximum = arith.maxnumf %a, %b : f16
    tt.reduce.return %maximum : f16
  }) : (tensor<128x8xf16, #blocked_packed_reduce>) -> tensor<128xf16, #ttg.slice<{dim = 1, parent = #blocked_packed_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_minimum_bf16
// TERNARY: %[[PACKED_MINIMUM_A:.*]] = llvm.intr.minimum(%{{.*}}, %{{.*}}) : (vector<2xbf16>, vector<2xbf16>) -> vector<2xbf16>
// TERNARY-NEXT: %[[PACKED_MINIMUM_B:.*]] = llvm.intr.minimum(%[[PACKED_MINIMUM_A]], %{{.*}}) : (vector<2xbf16>, vector<2xbf16>) -> vector<2xbf16>
// TERNARY-NEXT: llvm.intr.minimum(%[[PACKED_MINIMUM_B]], %{{.*}}) : (vector<2xbf16>, vector<2xbf16>) -> vector<2xbf16>
tt.func public @reduce_minimum_bf16(%arg0: tensor<128x8xbf16, #blocked_packed_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: bf16, %b: bf16):
    %minimum = arith.minimumf %a, %b : bf16
    tt.reduce.return %minimum : bf16
  }) : (tensor<128x8xbf16, #blocked_packed_reduce>) -> tensor<128xbf16, #ttg.slice<{dim = 1, parent = #blocked_packed_reduce}>>
  tt.return
}

// TERNARY-LABEL: @reduce_maximum_f64
// TERNARY: %[[F64_LEFT:.*]] = llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
// TERNARY-NEXT: %[[F64_RIGHT:.*]] = llvm.intr.maximum(%{{.*}}, %{{.*}}) : (f64, f64) -> f64
// TERNARY-NEXT: llvm.intr.maximum(%[[F64_LEFT]], %[[F64_RIGHT]]) : (f64, f64) -> f64
tt.func public @reduce_maximum_f64(%arg0: tensor<128x4xf64, #blocked_reduce>) {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f64, %b: f64):
    %maximum = arith.maximumf %a, %b : f64
    tt.reduce.return %maximum : f64
  }) : (tensor<128x4xf64, #blocked_reduce>) -> tensor<128xf64, #ttg.slice<{dim = 1, parent = #blocked_reduce}>>
  tt.return
}

}
