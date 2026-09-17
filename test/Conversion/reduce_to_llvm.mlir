// RUN: triton-opt %s --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm | mlir-translate -mlir-to-llvmir | opt -S -O1 | FileCheck %s
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=100 ptx-version=88' -cse | FileCheck %s --check-prefixes=TERNARY,REDUX,FASTMINMAX
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=80 ptx-version=80' -cse | FileCheck %s --check-prefixes=PRE100,REDUX,SLOWMINMAX
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' -cse | FileCheck %s --check-prefixes=PRE100,REDUX,SLOWMINMAX
// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=120 ptx-version=88' -cse | FileCheck %s --check-prefixes=REDUX,FASTMINMAX

#linear = #ttg.linear<{register = [[0, 2], [2, 0]], lane = [[0, 8], [8, 0], [1, 0], [4, 0], [16, 0]], warp = [[0, 1], [0, 4]], block = []}>
#blocked_reduce = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_packed_reduce = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_warp_reduce = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

#even_odd = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
#halves = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>

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

  // Reduce the even and odd lanes separately with converged full-warp redux.
  // CHECK-NEXT: [[TID:%.*]] = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  // CHECK-NEXT: [[GROUP_BIT:%.*]] = and i32 [[TID]], 1
  // CHECK-NEXT: [[GROUP:%.*]] = icmp eq i32 [[GROUP_BIT]], 0
  // CHECK-NEXT: [[V0:%.*]] = select i1 [[GROUP]], i32 [[SUM0]], i32 0
  // CHECK-NEXT: [[R0:%.*]] = tail call i32 @llvm.nvvm.redux.sync.add(i32 [[V0]], i32 -1)
  // CHECK-NEXT: [[V1:%.*]] = select i1 [[GROUP]], i32 0, i32 [[SUM0]]
  // CHECK-NEXT: [[R1:%.*]] = tail call i32 @llvm.nvvm.redux.sync.add(i32 [[V1]], i32 -1)
  // CHECK-NEXT: [[WSUM0:%.*]] = select i1 [[GROUP]], i32 [[R0]], i32 [[R1]]
  // CHECK-NEXT: [[V2:%.*]] = select i1 [[GROUP]], i32 [[SUM1]], i32 0
  // CHECK-NEXT: [[R2:%.*]] = tail call i32 @llvm.nvvm.redux.sync.add(i32 [[V2]], i32 -1)
  // CHECK-NEXT: [[V3:%.*]] = select i1 [[GROUP]], i32 0, i32 [[SUM1]]
  // CHECK-NEXT: [[R3:%.*]] = tail call i32 @llvm.nvvm.redux.sync.add(i32 [[V3]], i32 -1)
  // CHECK-NEXT: [[WSUM1:%.*]] = select i1 [[GROUP]], i32 [[R2]], i32 [[R3]]

  // CHECK-NEXT: [[DST0:%.*]] = insertvalue { i32, i32 } undef, i32 [[WSUM0]], 0
  // CHECK-NEXT: [[DST1:%.*]] = insertvalue { i32, i32 } [[DST0]], i32 [[WSUM1]], 1

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

// TERNARY-LABEL: @reduce_wide_minmax
// TERNARY: %[[MIN_HI:.*]] = nvvm.redux.sync min %{{.*}}, %{{.*}}
// TERNARY: %[[MIN_EQ:.*]] = llvm.icmp "eq" %{{.*}}, %[[MIN_HI]] : i32
// TERNARY: %[[MIN_LO:.*]] = llvm.select %[[MIN_EQ]], %{{.*}}, %{{.*}} : i1, i32
// TERNARY: nvvm.redux.sync umin %[[MIN_LO]], %{{.*}}
// TERNARY: %[[MAX_HI:.*]] = nvvm.redux.sync umax %{{.*}}, %{{.*}}
// TERNARY: %[[MAX_EQ:.*]] = llvm.icmp "eq" %{{.*}}, %[[MAX_HI]] : i32
// TERNARY: %[[MAX_LO:.*]] = llvm.select %[[MAX_EQ]], %{{.*}}, %{{.*}} : i1, i32
// TERNARY: nvvm.redux.sync umax %[[MAX_LO]], %{{.*}}
tt.func private @reduce_wide_minmax(%arg0: tensor<4x32xi64, #blocked_warp_reduce>) -> (tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>, tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>) {
  %minimum = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %c = arith.minsi %a, %b : i64
    tt.reduce.return %c : i64
  }) : (tensor<4x32xi64, #blocked_warp_reduce>) -> tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  %maximum = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %c = arith.maxui %a, %b : i64
    tt.reduce.return %c : i64
  }) : (tensor<4x32xi64, #blocked_warp_reduce>) -> tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %minimum, %maximum : tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>, tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// CHECK-LABEL: @reduce_wide_bitwise
// CHECK-COUNT-2: call i32 @llvm.nvvm.redux.sync.and
// CHECK-COUNT-2: call i32 @llvm.nvvm.redux.sync.or
// CHECK-COUNT-2: call i32 @llvm.nvvm.redux.sync.xor
// TERNARY-LABEL: @reduce_wide_bitwise
// TERNARY-DAG: %[[SHIFT:.*]] = llvm.mlir.constant(32 : i64)
// TERNARY-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// TERNARY: %[[SHIFTED:.*]] = llvm.lshr %[[INPUT:.*]], %[[SHIFT]] : i64
// TERNARY: %[[HI:.*]] = llvm.trunc %[[SHIFTED]] : i64 to i32
// TERNARY: nvvm.redux.sync and %[[HI]], %[[MASK]]
// TERNARY: %[[LO:.*]] = llvm.trunc %[[INPUT]] : i64 to i32
// TERNARY: nvvm.redux.sync and %[[LO]], %[[MASK]]
// TERNARY: nvvm.redux.sync or %[[HI]], %[[MASK]]
// TERNARY: nvvm.redux.sync or %[[LO]], %[[MASK]]
// TERNARY: %[[XOR_HI:.*]] = nvvm.redux.sync xor %[[HI]], %[[MASK]]
// TERNARY: %[[XOR_LO:.*]] = nvvm.redux.sync xor %[[LO]], %[[MASK]]
// TERNARY-DAG: %[[HI64:.*]] = llvm.zext %[[XOR_HI]] : i32 to i64
// TERNARY-DAG: %[[LO64:.*]] = llvm.zext %[[XOR_LO]] : i32 to i64
// TERNARY-DAG: %[[RESULT_HI:.*]] = llvm.shl %[[HI64]], %[[SHIFT]] : i64
// TERNARY: llvm.or %[[RESULT_HI]], %[[LO64]] : i64
tt.func @reduce_wide_bitwise(
    %arg0: tensor<4x32xi64, #blocked_warp_reduce>,
    %and_ptr: tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>,
    %or_ptr: tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>,
    %xor_ptr: tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>) {
  %and = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %c = arith.andi %a, %b : i64
    tt.reduce.return %c : i64
  }) : (tensor<4x32xi64, #blocked_warp_reduce>) -> tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  %or = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %c = arith.ori %a, %b : i64
    tt.reduce.return %c : i64
  }) : (tensor<4x32xi64, #blocked_warp_reduce>) -> tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  %xor = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i64, %b: i64):
    %c = arith.xori %a, %b : i64
    tt.reduce.return %c : i64
  }) : (tensor<4x32xi64, #blocked_warp_reduce>) -> tensor<4xi64, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.store %and_ptr, %and : tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.store %or_ptr, %or : tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.store %xor_ptr, %xor : tensor<4x!tt.ptr<i64>, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return
}

// PRE100-LABEL: @reduce_even_odd_minnum
// PRE100-NOT: nvvm.redux
// PRE100: llvm.return
// TERNARY-LABEL: @reduce_even_odd_minnum
// TERNARY-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// TERNARY-DAG: %[[NAN:.*]] = llvm.mlir.constant(0x7FC00000 : f32)
// TERNARY-DAG: %[[BIT:.*]] = llvm.mlir.constant(1 : i32)
// TERNARY: llvm.and %{{.*}}, %[[BIT]]
// TERNARY: %[[GROUP:.*]] = llvm.icmp "eq"
// TERNARY: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[NAN]]
// TERNARY: %[[R0:.*]] = nvvm.redux.sync fmin %[[FIRST]], %[[MASK]]
// TERNARY: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[NAN]], %[[INPUT]]
// TERNARY: %[[R1:.*]] = nvvm.redux.sync fmin %[[SECOND]], %[[MASK]]
// TERNARY: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// TERNARY-NOT: nvvm.shfl
// TERNARY: llvm.return
tt.func private @reduce_even_odd_minnum(%arg0: tensor<8x16xf32, #even_odd>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #even_odd}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.minnumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<8x16xf32, #even_odd>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #even_odd}>>
  tt.return %0 : tensor<8xf32, #ttg.slice<{dim = 1, parent = #even_odd}>>
}

// PRE100-LABEL: @reduce_halves_maximum
// PRE100-NOT: nvvm.redux
// PRE100: llvm.return
// TERNARY-LABEL: @reduce_halves_maximum
// TERNARY-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// TERNARY-DAG: %[[INF:.*]] = llvm.mlir.constant(0xFF800000 : f32)
// TERNARY-DAG: %[[BIT:.*]] = llvm.mlir.constant(16 : i32)
// TERNARY: llvm.and %{{.*}}, %[[BIT]]
// TERNARY: %[[GROUP:.*]] = llvm.icmp "eq"
// TERNARY: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[INF]]
// TERNARY: %[[R0:.*]] = nvvm.redux.sync fmax %[[FIRST]], %[[MASK]] {{.*}}nan
// TERNARY: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[INF]], %[[INPUT]]
// TERNARY: %[[R1:.*]] = nvvm.redux.sync fmax %[[SECOND]], %[[MASK]] {{.*}}nan
// TERNARY: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// TERNARY-NOT: nvvm.shfl
// TERNARY: llvm.return
tt.func private @reduce_halves_maximum(%arg0: tensor<8x16xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.maximumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<8x16xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// REDUX-LABEL: @reduce_even_odd_sum
// REDUX-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// REDUX: %[[GROUP:.*]] = llvm.icmp "eq"
// REDUX: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[ZERO]]
// REDUX: %[[R0:.*]] = nvvm.redux.sync add %[[FIRST]], %[[MASK]]
// REDUX: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[ZERO]], %[[INPUT]]
// REDUX: %[[R1:.*]] = nvvm.redux.sync add %[[SECOND]], %[[MASK]]
// REDUX: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_even_odd_sum(%arg0: tensor<8x16xi32, #even_odd>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #even_odd}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x16xi32, #even_odd>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #even_odd}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #even_odd}>>
}

// REDUX-LABEL: @reduce_halves_and
// REDUX-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX: %[[GROUP:.*]] = llvm.icmp "eq"
// REDUX: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[MASK]]
// REDUX: %[[R0:.*]] = nvvm.redux.sync and %[[FIRST]], %[[MASK]]
// REDUX: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[MASK]], %[[INPUT]]
// REDUX: %[[R1:.*]] = nvvm.redux.sync and %[[SECOND]], %[[MASK]]
// REDUX: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_halves_and(%arg0: tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.andi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// REDUX-LABEL: @reduce_halves_or
// REDUX-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// REDUX: %[[GROUP:.*]] = llvm.icmp "eq"
// REDUX: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[ZERO]]
// REDUX: %[[R0:.*]] = nvvm.redux.sync or %[[FIRST]], %[[MASK]]
// REDUX: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[ZERO]], %[[INPUT]]
// REDUX: %[[R1:.*]] = nvvm.redux.sync or %[[SECOND]], %[[MASK]]
// REDUX: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_halves_or(%arg0: tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.ori %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// REDUX-LABEL: @reduce_halves_xor
// REDUX-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// REDUX: %[[GROUP:.*]] = llvm.icmp "eq"
// REDUX: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[ZERO]]
// REDUX: %[[R0:.*]] = nvvm.redux.sync xor %[[FIRST]], %[[MASK]]
// REDUX: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[ZERO]], %[[INPUT]]
// REDUX: %[[R1:.*]] = nvvm.redux.sync xor %[[SECOND]], %[[MASK]]
// REDUX: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_halves_xor(%arg0: tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.xori %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// PRE100-LABEL: @reduce_broadcast_maximum
// PRE100-NOT: nvvm.redux
// PRE100: llvm.return
// TERNARY-LABEL: @reduce_broadcast_maximum
// TERNARY: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// TERNARY-NOT: llvm.select
// TERNARY: nvvm.redux.sync fmax %{{.*}}, %[[MASK]] {{.*}}nan
// TERNARY-NOT: nvvm.redux
// TERNARY-NOT: nvvm.shfl
// TERNARY-NOT: llvm.select
// TERNARY: llvm.return
tt.func private @reduce_broadcast_maximum(%arg0: tensor<4x2xf32, #blocked_warp_reduce>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.maximumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<4x2xf32, #blocked_warp_reduce>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %0 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// REDUX-LABEL: @reduce_broadcast_sum
// REDUX-DAG: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-DAG: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// REDUX-DAG: %[[BROADCAST:.*]] = llvm.mlir.constant(28 : i32)
// REDUX: %[[BITS:.*]] = llvm.and %{{.*}}, %[[BROADCAST]]
// REDUX: %[[UNIQUE:.*]] = llvm.icmp "eq" %[[BITS]], %[[ZERO]]
// REDUX: %[[VALUE:.*]] = llvm.select %[[UNIQUE]], %{{.*}}, %[[ZERO]]
// REDUX: nvvm.redux.sync add %[[VALUE]], %[[MASK]]
// REDUX-NOT: nvvm.redux
// REDUX-NOT: nvvm.shfl
// REDUX-NOT: llvm.select
// REDUX: llvm.return
tt.func private @reduce_broadcast_sum(%arg0: tensor<4x4xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.addi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<4x4xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// REDUX-LABEL: @reduce_broadcast_halves_xor
// REDUX-NOT: nvvm.redux
// REDUX: nvvm.shfl.sync
// REDUX-NOT: nvvm.redux
// REDUX: llvm.return
tt.func private @reduce_broadcast_halves_xor(%arg0: tensor<8x8xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.xori %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x8xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// PRE100-LABEL: @reduce_broadcast_halves_maximum
// PRE100-NOT: nvvm.redux
// PRE100: llvm.return
// TERNARY-LABEL: @reduce_broadcast_halves_maximum
// TERNARY-NOT: nvvm.shfl
// TERNARY-COUNT-2: nvvm.redux.sync fmax
// TERNARY-NOT: nvvm.redux
// TERNARY-NOT: nvvm.shfl
// TERNARY: llvm.return
tt.func private @reduce_broadcast_halves_maximum(%arg0: tensor<8x8xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.maximumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<8x8xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// PRE100-LABEL: @reduce_broadcast_halves_maximum_small
// PRE100-NOT: nvvm.redux
// PRE100: llvm.return
// TERNARY-LABEL: @reduce_broadcast_halves_maximum_small
// TERNARY-NOT: nvvm.redux
// TERNARY: nvvm.shfl.sync
// TERNARY-NOT: nvvm.redux
// TERNARY: llvm.return
tt.func private @reduce_broadcast_halves_maximum_small(%arg0: tensor<8x4xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: f32, %b: f32):
    %c = arith.maximumf %a, %b : f32
    tt.reduce.return %c : f32
  }) : (tensor<8x4xf32, #halves>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xf32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// REDUX-LABEL: @reduce_broadcast_xor_small
// REDUX-NOT: nvvm.redux
// REDUX: nvvm.shfl.sync
// REDUX-NOT: nvvm.redux
// REDUX: llvm.return
tt.func private @reduce_broadcast_xor_small(%arg0: tensor<4x2xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.xori %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<4x2xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// Four broadcast values meet the generic one-redux threshold.
// REDUX-LABEL: @reduce_broadcast_maxsi
// REDUX: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-NOT: nvvm.shfl
// REDUX-NOT: llvm.select
// REDUX: nvvm.redux.sync max %{{.*}}, %[[MASK]]
// REDUX-NOT: nvvm.redux
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_broadcast_maxsi(%arg0: tensor<4x4xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.maxsi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<4x4xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// Two values are profitable only with the SM100+ min/max threshold.
// SLOWMINMAX-LABEL: @reduce_broadcast_maxsi_small
// SLOWMINMAX-NOT: nvvm.redux
// SLOWMINMAX: nvvm.shfl.sync
// SLOWMINMAX-NOT: nvvm.redux
// SLOWMINMAX: llvm.return
// FASTMINMAX-LABEL: @reduce_broadcast_maxsi_small
// FASTMINMAX: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// FASTMINMAX-NOT: nvvm.shfl
// FASTMINMAX-NOT: llvm.select
// FASTMINMAX: nvvm.redux.sync max %{{.*}}, %[[MASK]]
// FASTMINMAX-NOT: nvvm.redux
// FASTMINMAX-NOT: nvvm.shfl
// FASTMINMAX: llvm.return
tt.func private @reduce_broadcast_maxsi_small(%arg0: tensor<4x2xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.maxsi %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<4x2xi32, #blocked_warp_reduce>) -> tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
  tt.return %0 : tensor<4xi32, #ttg.slice<{dim = 1, parent = #blocked_warp_reduce}>>
}

// Sixteen values per group meet the generic two-redux threshold.
// REDUX-LABEL: @reduce_halves_minui
// REDUX: %[[MASK:.*]] = llvm.mlir.constant(-1 : i32)
// REDUX-NOT: nvvm.shfl
// REDUX: %[[GROUP:.*]] = llvm.icmp "eq"
// REDUX: %[[FIRST:.*]] = llvm.select %[[GROUP]], %[[INPUT:.*]], %[[MASK]]
// REDUX: %[[R0:.*]] = nvvm.redux.sync umin %[[FIRST]], %[[MASK]]
// REDUX: %[[SECOND:.*]] = llvm.select %[[GROUP]], %[[MASK]], %[[INPUT]]
// REDUX: %[[R1:.*]] = nvvm.redux.sync umin %[[SECOND]], %[[MASK]]
// REDUX: llvm.select %[[GROUP]], %[[R0]], %[[R1]]
// REDUX-NOT: nvvm.redux
// REDUX-NOT: nvvm.shfl
// REDUX: llvm.return
tt.func private @reduce_halves_minui(%arg0: tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.minui %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x16xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// Eight values per group are profitable only with SM100+ min/max.
// SLOWMINMAX-LABEL: @reduce_broadcast_halves_minui
// SLOWMINMAX-NOT: nvvm.redux
// SLOWMINMAX: nvvm.shfl.sync
// SLOWMINMAX-NOT: nvvm.redux
// SLOWMINMAX: llvm.return
// FASTMINMAX-LABEL: @reduce_broadcast_halves_minui
// FASTMINMAX-NOT: nvvm.shfl
// FASTMINMAX-COUNT-2: nvvm.redux.sync umin
// FASTMINMAX-NOT: nvvm.redux
// FASTMINMAX-NOT: nvvm.shfl
// FASTMINMAX: llvm.return
tt.func private @reduce_broadcast_halves_minui(%arg0: tensor<8x8xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.minui %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x8xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

// Four values per group remain below both two-redux thresholds.
// REDUX-LABEL: @reduce_broadcast_halves_minui_small
// REDUX-NOT: nvvm.redux
// REDUX: nvvm.shfl.sync
// REDUX-NOT: nvvm.redux
// REDUX: llvm.return
tt.func private @reduce_broadcast_halves_minui_small(%arg0: tensor<8x4xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>> {
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %c = arith.minui %a, %b : i32
    tt.reduce.return %c : i32
  }) : (tensor<8x4xi32, #halves>) -> tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
  tt.return %0 : tensor<8xi32, #ttg.slice<{dim = 1, parent = #halves}>>
}

}
