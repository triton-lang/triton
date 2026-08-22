// RUN: triton-opt %s -split-input-file -tritongpu-optimize-thread-locality -canonicalize | FileCheck %s

// CHECK-LABEL: negative_zero_accumulator
// CHECK: %[[INIT_ARG:.*]] = arith.constant dense<0.000000e+00>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[INIT_ARG]]) -> {{.*}}
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[FACTORED:.*]] = tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{32x2x32x2xf32.*}}>
// CHECK-NEXT: %[[TRANSPOSED:.*]] = tt.trans %[[FACTORED]] {order = array<i32: 0, 2, 1, 3>}
// CHECK-NEXT: tt.reshape %[[TRANSPOSED]] efficient_layout : {{.*}} -> tensor<{{32x32x4xf32.*}}>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.addf
// CHECK: arith.addf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: tt.store {{%.*}}, %[[CVT_OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @negative_zero_accumulator(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<-0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.addf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: positive_zero_accumulator
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00>
// CHECK-NEXT: %[[CST1:.*]] = arith.constant dense<0.000000e+00>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST1]]) -> {{.*}}
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.addf
// CHECK: arith.addf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: arith.addf %[[CVT_OUTPUT]], %[[CST]]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @positive_zero_accumulator(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.addf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: @row_major_register_grouping
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[FACTORED:.*]] = tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{8x2x32x4xf32.*}}>
// CHECK-NEXT: %[[TRANSPOSED:.*]] = tt.trans %[[FACTORED]] {order = array<i32: 0, 2, 1, 3>}
// CHECK-NEXT: %[[RESHAPED:.*]] = tt.reshape %[[TRANSPOSED]] efficient_layout : {{.*}} -> tensor<{{8x32x8xf32.*}}>
// CHECK: %[[REDUCED:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: scf.yield
// CHECK: "tt.reduce"({{%.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @row_major_register_grouping(%input: tensor<8x256x!tt.ptr<f32>, #blocked>, %output: tensor<8x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<8xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<8xf32, #slice>) : i32 {
      %values = tt.load %input : tensor<8x256x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%values) <{axis = 1 : i32}> ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        tt.reduce.return %sum : f32
      }) : (tensor<8x256xf32, #blocked>) -> tensor<8xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<8xf32, #slice>
      scf.yield %updated : tensor<8xf32, #slice>
    }
    tt.store %output, %result : tensor<8x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// CHECK-LABEL: @preserve_rows_when_reordering_registers
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[FACTORED:.*]] = tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{256x2x2x1xf32.*}}>
// CHECK-NEXT: %[[TRANSPOSED:.*]] = tt.trans %[[FACTORED]] {order = array<i32: 0, 2, 1, 3>}
// CHECK-NEXT: %[[RESHAPED:.*]] = tt.reshape %[[TRANSPOSED]] efficient_layout : {{.*}} -> tensor<{{256x2x2xf32.*}}>
// CHECK-NEXT: %[[REORDERED:.*]] = ttg.convert_layout %[[RESHAPED]] : tensor<{{256x2x2xf32.*}}> -> tensor<{{256x2x2xf32.*}}>
// CHECK-NEXT: "tt.reduce"(%[[REORDERED]]) <{axis = 2 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [0, 1]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @preserve_rows_when_reordering_registers(%input: tensor<256x4x!tt.ptr<f32>, #blocked>, %output: tensor<256x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<256xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<256xf32, #slice>) : i32 {
      %values = tt.load %input : tensor<256x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%values) <{axis = 1 : i32}> ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        tt.reduce.return %sum : f32
      }) : (tensor<256x4xf32, #blocked>) -> tensor<256xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<256xf32, #slice>
      scf.yield %updated : tensor<256xf32, #slice>
    }
    tt.store %output, %result : tensor<256x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The yielded accumulator belongs to a different loop than the one enclosing the reduce.
// CHECK-LABEL: @reduce_outside_accumulator_loop
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reduce_outside_accumulator_loop(%input: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    scf.for %index = %start to %count step %step : i32 {
      %values = tt.load %input : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%values) <{axis = 1 : i32}> ({
      ^bb0(%left: f32, %right: f32):
        %sum = arith.addf %left, %right : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %result = scf.for %inner = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
        %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
        scf.yield %updated : tensor<32xf32, #slice>
      }
      tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: slice_layout
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK-NEXT: scf.yield
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[LOOP_OUTPUT]]
#blocked3d = #ttg.blocked<{sizePerThread = [1, 4, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#slice2d = #ttg.slice<{dim = 2, parent = #blocked3d}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @slice_layout(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #slice2d> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #slice2d}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #slice2d}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #slice2d}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #slice2d}>> -> tensor<1x128xi32, #slice2d>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #slice2d> -> tensor<32x128xi32, #slice2d>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #slice2d>, tensor<32x128xi32, #slice2d>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #slice2d>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #slice2d>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>>
      %35 = arith.addf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #slice2d}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: mma_layout
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
// CHECK: arith.addf
// CHECK: arith.addf
// CHECK-NEXT: scf.yield
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[LOOP_OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mma_layout(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #mma> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #mma}>> -> tensor<1x128xi32, #mma>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #mma> -> tensor<32x128xi32, #mma>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #mma>, tensor<32x128xi32, #mma>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #mma>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #mma>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %35 = arith.addf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: max_reduce
// CHECK: %[[INIT_ARG:.*]] = arith.constant dense<0xFF800000>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[INIT_ARG]]) -> {{.*}}
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{32x2x32x2xf32.*}}>
// CHECK: tt.reshape {{%.*}} efficient_layout : {{.*}} -> tensor<{{32x32x4xf32.*}}>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.maximumf
// CHECK: arith.maximumf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.maximumf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: tt.store {{%.*}}, %[[CVT_OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @max_reduce(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0xFF800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.maximumf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.maximumf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: max_reduce_zero_int_accumulator
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00>
// CHECK-NEXT: %[[CST1:.*]] = arith.constant dense<0xFF800000>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST1]]) -> {{.*}}
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.maximumf
// CHECK: arith.maximumf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.maximumf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: arith.maximumf %[[CVT_OUTPUT]], %[[CST]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @max_reduce_zero_int_accumulator(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.maximumf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.maximumf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: min_reduce
// CHECK: %[[CST:.*]] = arith.constant dense<0x7F800000>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST]]) -> {{.*}}
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{32x2x32x2xf32.*}}>
// CHECK: tt.reshape {{%.*}} efficient_layout : {{.*}} -> tensor<{{32x32x4xf32.*}}>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.minimumf
// CHECK: arith.minimumf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.minimumf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: tt.store {{%.*}}, %[[CVT_OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @min_reduce(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0x7F800000> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.minimumf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.minimumf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: min_reduce_zero_int_accumulator
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00>
// CHECK-NEXT: %[[CST1:.*]] = arith.constant dense<0x7F800000>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST1]]) -> {{.*}}
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.minimumf
// CHECK: arith.minimumf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.minimumf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: arith.minimumf %[[CVT_OUTPUT]], %[[CST]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @min_reduce_zero_int_accumulator(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.minimumf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.minimumf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: mul_reduce
// CHECK: %[[CST:.*]] = arith.constant dense<1.000000e+00>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST]]) -> {{.*}}
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: tt.reshape %[[LOAD]] : {{.*}} -> tensor<{{32x2x32x2xf32.*}}>
// CHECK: tt.reshape {{%.*}} efficient_layout : {{.*}} -> tensor<{{32x32x4xf32.*}}>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.mulf
// CHECK: arith.mulf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.mulf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: tt.store {{%.*}}, %[[CVT_OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mul_reduce(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<1.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.mulf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.mulf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: mul_reduce_zero_int_accumulator
// CHECK: %[[CST:.*]] = arith.constant dense
// CHECK-NEXT: %[[CST1:.*]] = arith.constant dense<1.000000e+00>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[CST1]]) -> {{.*}}
// CHECK: tt.load
// CHECK: tt.reshape
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.mulf
// CHECK: arith.mulf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.mulf
// CHECK: %[[CVT_OUTPUT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: arith.mulf %[[CVT_OUTPUT]], %[[CST]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mul_reduce_zero_int_accumulator(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.mulf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.mulf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}


// -----

// The reduce operand comes from an elementwise op rather than a load
// CHECK-LABEL: elementwise_operand
// CHECK: %[[ORIG_INIT:.*]] = arith.constant dense<0.000000e+00>
// CHECK: %[[INIT_ARG:.*]] = arith.constant dense<0xFF800000>
// CHECK: %[[LOOP_OUTPUT:.*]] = scf.for {{.*}} iter_args(%[[FOR_ARG:.*]] = %[[INIT_ARG]]) -> {{.*}}
// CHECK: %[[LOAD:.*]] = tt.load
// CHECK: %[[MULF:.*]] = arith.mulf %[[LOAD]], %[[LOAD]]
// CHECK: tt.reshape %[[MULF]] : {{.*}} -> tensor<{{32x2x32x2xf32.*}}>
// CHECK: tt.reshape {{%.*}} efficient_layout : {{.*}} -> tensor<{{32x32x4xf32.*}}>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{%.*}}) <{axis = 2 : i32}>
// CHECK: arith.maximumf
// CHECK: arith.maximumf %[[FOR_ARG]], %[[REDUCE]]
// CHECK-NEXT: scf.yield
// CHECK: %[[FINAL_REDUCE:.*]] = "tt.reduce"(%[[LOOP_OUTPUT]]) <{axis = 1 : i32}>
// CHECK: arith.maximumf
// CHECK: %[[CVT:.*]] = ttg.convert_layout %[[FINAL_REDUCE]]
// CHECK: %[[WITH_INIT:.*]] = arith.maximumf %[[CVT]], %[[ORIG_INIT]]
// CHECK: %[[OUTPUT:.*]] = ttg.convert_layout %[[WITH_INIT]]
// CHECK: tt.store {{%.*}}, %[[OUTPUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @elementwise_operand(
    %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32},
    %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32},
    %18: tensor<32x128x!tt.ptr<f32>, #blocked> {tt.divisibility = 16 : i32},
    %11: i32 {tt.divisibility = 16 : i32},
    %25: tensor<32x!tt.ptr<f32>, #blocked1> {tt.divisibility = 16 : i32}
    ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c128_i32 = arith.constant 128 : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs y : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 : tensor<32x128x!tt.ptr<f32>, #blocked>
      %333 = arith.mulf %33, %33: tensor<32x128xf32, #blocked>
      %34 = "tt.reduce"(%333) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.maximumf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.maximumf %arg4, %34 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    }
    %26 = ttg.convert_layout %19 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32xf32, #blocked1>
    tt.store %25, %26 : tensor<32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[$BLOCK0:.+]] = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCK1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCK2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
// CHECK-LABEL: optimize_view_layout
// CHECK: %[[R:.+]] = tt.reshape {{.*}} allow_reorder efficient_layout : tensor<8x128xf32, #[[$BLOCK0]]> -> tensor<64x16xf32, #[[$BLOCK2]]>
// CHECK: %[[C:.+]] = ttg.convert_layout %[[R]] : tensor<64x16xf32, #[[$BLOCK2]]> -> tensor<64x16xf32, #[[$BLOCK1]]>
// CHECK:  "tt.reduce"(%[[C]])
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @optimize_view_layout(%arg0: tensor<8x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>> {
    %0 = tt.reshape %arg0 allow_reorder : tensor<8x128xf32, #blocked> -> tensor<64x16xf32, #blocked1>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.maximumf %arg1, %arg2 : f32
      tt.reduce.return %2 : f32
    }) : (tensor<64x16xf32, #blocked1>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    tt.return %1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked1}>>
  }
}

// -----


// CHECK-DAG: #[[$BLOCK0:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
// CHECK-DAG: #[[$BLOCK1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 1], order = [0, 1]}>
// CHECK-LABEL: optimize_view_layout_same_shape
// CHECK: %[[R:.+]] = tt.reshape {{.*}} allow_reorder efficient_layout : tensor<64x16xf32, #[[$BLOCK0]]> -> tensor<64x16xf32, #[[$BLOCK1]]>
// CHECK: %[[C:.+]] = ttg.convert_layout %[[R]] : tensor<64x16xf32, #[[$BLOCK1]]> -> tensor<64x16xf32, #[[$BLOCK0]]>
// CHECK:  "tt.reduce"(%[[C]])
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @optimize_view_layout_same_shape(%arg0: tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = tt.reshape %arg0 allow_reorder : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%arg1: f32, %arg2: f32):
      %2 = arith.maximumf %arg1, %arg2 : f32
      tt.reduce.return %2 : f32
    }) : (tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func public @reduce_for_arg(%arg: tensor<64x128xf32, #blocked>, %arg1: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64x128xf32, #blocked>
    %64:1 = scf.for %arg22 = %c0_i32 to %c4096_i32 step %c128_i32 iter_args(%arg29 = %arg) -> (tensor<64x128xf32, #blocked>)  : i32 {
      %129 = "tt.reduce"(%arg29) <{axis = 1 : i32}> ({
      ^bb0(%arg31: f32, %arg32: f32):
        %160 = arith.maxnumf %arg31, %arg32 : f32
        tt.reduce.return %160 : f32
      }) : (tensor<64x128xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %75 = ttg.convert_layout %129 : tensor<64xf32, #slice> -> tensor<64xf32, #blocked1>
      %79 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
      %80 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>, #blocked1>
      %81 = tt.addptr %80, %79 : tensor<64x!tt.ptr<f32>, #blocked1>, tensor<64xi32, #blocked1>
      tt.store %81, %75 : tensor<64x!tt.ptr<f32>, #blocked1>
      %141 = arith.addf %arg29, %cst_1 : tensor<64x128xf32, #blocked>
      scf.yield %141 : tensor<64x128xf32, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_square_axis_0
tt.func @set_warp_shuffle_layout_square_axis_0(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
  // CHECK-NEXT: [[SRC:%.*]] = ttg.convert_layout %arg0
  // CHECK-NEXT: [[IDX:%.*]] = ttg.convert_layout %arg1
  // CHECK-NEXT: [[OUT:%.*]] = tt.gather [[SRC]][[[IDX]]] {axis = 0 : i32, efficient_layout} : (tensor<64x64xf32, [[LAYOUT]]>, tensor<64x64xi32, [[LAYOUT]]>) -> tensor<64x64xf32, [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
  // CHECK-NEXT: [[RES:%.*]] = ttg.convert_layout [[OUT]]
  // CHECK-NEXT: return [[RES]]
  tt.return %0 : tensor<64x64xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_square_axis_1
tt.func @set_warp_shuffle_layout_square_axis_1(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
  // CHECK: tt.gather {{.*}} (tensor<64x64xf32, [[LAYOUT]]>, tensor<64x64xi32, [[LAYOUT]]>) -> tensor<64x64xf32, [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
  tt.return %0 : tensor<64x64xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_warp_broadcast
tt.func @set_warp_shuffle_layout_warp_broadcast(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
  tt.return %0 : tensor<64x1xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_3d_warp
tt.func @set_warp_shuffle_layout_3d_warp(%arg0: tensor<32x2x32xf32, #blocked>, %arg1: tensor<32x2x2xi32, #blocked>) -> tensor<32x2x2xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
    %0 = tt.gather %arg0[%arg1] {axis = 2 : i32} : (tensor<32x2x32xf32, #blocked>, tensor<32x2x2xi32, #blocked>) -> tensor<32x2x2xf32, #blocked>
    tt.return %0 : tensor<32x2x2xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_3d_warp_thread_split
tt.func @set_warp_shuffle_layout_3d_warp_thread_split(%arg0: tensor<32x4x16xf32, #blocked>, %arg1: tensor<32x4x2xi32, #blocked>) -> tensor<32x4x2xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
    %0 = tt.gather %arg0[%arg1] {axis = 2 : i32} : (tensor<32x4x16xf32, #blocked>, tensor<32x4x2xi32, #blocked>) -> tensor<32x4x2xf32, #blocked>
    tt.return %0 : tensor<32x4x2xf32, #blocked>
}

}


// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_thread_broadcast
tt.func @set_warp_shuffle_layout_thread_broadcast(%arg0: tensor<16x64xf32, #blocked>, %arg1: tensor<16x1xi32, #blocked>) -> tensor<16x1xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<16x64xf32, #blocked>, tensor<16x1xi32, #blocked>) -> tensor<16x1xf32, #blocked>
  tt.return %0 : tensor<16x1xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_large_source
tt.func @set_warp_shuffle_layout_large_source(%arg0: tensor<256x256xf32, #blocked>, %arg1: tensor<256x8xi32, #blocked>) -> tensor<256x8xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<256x256xf32, #blocked>, tensor<256x8xi32, #blocked>) -> tensor<256x8xf32, #blocked>
  tt.return %0 : tensor<256x8xf32, #blocked>
}

}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: skip_optimize_on_1d_tensor
tt.func @skip_optimize_on_1d_tensor(%arg0: tensor<256xf32, #blocked>, %arg1: tensor<8xi32, #blocked>) -> tensor<8xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<256xf32, #blocked>, tensor<8xi32, #blocked>) -> tensor<8xf32, #blocked>
  tt.return %0 : tensor<8xf32, #blocked>
}

}

// -----

// A multi-operand reduce must be left untouched
// CHECK-LABEL: @two_operand_reduce
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}, {{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_operand_reduce(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %tags: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %t = tt.load %tags : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced:2 = "tt.reduce"(%v, %t) <{axis = 1 : i32}> ({
      ^bb0(%a0: f32, %a1: f32, %b0: f32, %b1: f32):
        %sum = arith.addf %a0, %b0 : f32
        tt.reduce.return %sum, %a1 : f32, f32
      }) : (tensor<32x4xf32, #blocked>, tensor<32x4xf32, #blocked>) -> (tensor<32xf32, #slice>, tensor<32xf32, #slice>)
      %updated = arith.addf %accumulator, %reduced#0 : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// A degenerate combiner that could change the result if the optimization is applied
// CHECK-LABEL: @self_add_combiner
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @self_add_combiner(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %a : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The accumulator update op does not match the reduce's combiner
// CHECK-LABEL: @update_combiner_mismatch
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_combiner_mismatch(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0xFF800000> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.maximumf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The accumulator's final value is consumed in two places
// CHECK-LABEL: @loop_result_two_uses
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_result_two_uses(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %out1: tensor<32x!tt.ptr<f32>, #slice>, %out2: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %out1, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.store %out2, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The accumulator has a second use inside the loop
// CHECK-LABEL: @accumulator_second_use
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @accumulator_second_use(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %trace: tensor<32x!tt.ptr<f32>, #slice>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      tt.store %trace, %accumulator : tensor<32x!tt.ptr<f32>, #slice>
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The update's other operand is a rescaled accumulator rather than the iter_arg
// CHECK-LABEL: @rescaled_accumulator
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @rescaled_accumulator(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %alpha = arith.constant dense<5.000000e-01> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %scaled = arith.mulf %accumulator, %alpha : tensor<32xf32, #slice>
      %updated = arith.addf %scaled, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// Two independent accumulators in the same loop are both optimized
// CHECK-LABEL: @two_independent_accumulators
// CHECK: %[[RES:.*]]:2 = scf.for {{.*}} iter_args(%[[SUM:.*]] = %{{.*}}, %[[MAX:.*]] = %{{.*}})
// CHECK: %[[A_RED:.*]] = "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK: %[[SUM_UPD:.*]] = arith.addf %[[SUM]], %[[A_RED]]
// CHECK: %[[B_RED:.*]] = "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK: %[[MAX_UPD:.*]] = arith.maximumf %[[MAX]], %[[B_RED]]
// CHECK: scf.yield %[[SUM_UPD]], %[[MAX_UPD]]
// CHECK: %[[MAX_RED:.*]] = "tt.reduce"(%[[RES]]#1) <{axis = 1 : i32}>
// CHECK: %[[MAX_OUT:.*]] = ttg.convert_layout %[[MAX_RED]]
// CHECK: %[[SUM_RED:.*]] = "tt.reduce"(%[[RES]]#0) <{axis = 1 : i32}>
// CHECK: %[[SUM_CVT:.*]] = ttg.convert_layout %[[SUM_RED]]
// CHECK: %[[SUM_OUT:.*]] = arith.addf %[[SUM_CVT]], %{{.*}}
// CHECK: tt.store %{{.*}}, %[[SUM_OUT]]
// CHECK: tt.store %{{.*}}, %[[MAX_OUT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @two_independent_accumulators(%a_ptr: tensor<32x4x!tt.ptr<f32>, #blocked>, %b_ptr: tensor<32x4x!tt.ptr<f32>, #blocked>, %sum_out: tensor<32x!tt.ptr<f32>, #slice>, %max_out: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %ninf = arith.constant dense<0xFF800000> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result:2 = scf.for %i = %start to %count step %step iter_args(%sum_acc = %zero, %max_acc = %ninf) -> (tensor<32xf32, #slice>, tensor<32xf32, #slice>) : i32 {
      %a = tt.load %a_ptr : tensor<32x4x!tt.ptr<f32>, #blocked>
      %a_red = "tt.reduce"(%a) <{axis = 1 : i32}> ({
      ^bb0(%l: f32, %r: f32):
        %s = arith.addf %l, %r : f32
        tt.reduce.return %s : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %sum_upd = arith.addf %sum_acc, %a_red : tensor<32xf32, #slice>
      %b = tt.load %b_ptr : tensor<32x4x!tt.ptr<f32>, #blocked>
      %b_red = "tt.reduce"(%b) <{axis = 1 : i32}> ({
      ^bb0(%l: f32, %r: f32):
        %m = arith.maximumf %l, %r : f32
        tt.reduce.return %m : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %max_upd = arith.maximumf %max_acc, %b_red : tensor<32xf32, #slice>
      scf.yield %sum_upd, %max_upd : tensor<32xf32, #slice>, tensor<32xf32, #slice>
    }
    tt.store %sum_out, %result#0 : tensor<32x!tt.ptr<f32>, #slice>
    tt.store %max_out, %result#1 : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// Two reductions in a loop, each accumulator's running value feeds the other reduce's load mask
// CHECK-LABEL: @mutually_dependent_accumulators
// CHECK: scf.for {{.*}} iter_args(%[[SUM:.*]] = %{{.*}}, %[[MAX:.*]] = %{{.*}})
// CHECK: tt.expand_dims %[[MAX]]
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: arith.addf %[[SUM]],
// CHECK: tt.expand_dims %[[SUM]]
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
// CHECK: arith.maximumf %[[MAX]],
#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @mutually_dependent_accumulators(%a_ptr: tensor<32x4x!tt.ptr<f32>, #blocked>, %b_ptr: tensor<32x4x!tt.ptr<f32>, #blocked>, %sum_out: tensor<32x!tt.ptr<f32>, #slice>, %max_out: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %ninf = arith.constant dense<0xFF800000> : tensor<32xf32, #slice>
    %thresh = arith.constant dense<0.000000e+00> : tensor<32x1xf32, #blocked>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result:2 = scf.for %i = %start to %count step %step iter_args(%sum_acc = %zero, %max_acc = %ninf) -> (tensor<32xf32, #slice>, tensor<32xf32, #slice>) : i32 {
      %max_col = tt.expand_dims %max_acc {axis = 1 : i32} : tensor<32xf32, #slice> -> tensor<32x1xf32, #blocked>
      %a_cmp = arith.cmpf ogt, %max_col, %thresh : tensor<32x1xf32, #blocked>
      %a_mask = tt.broadcast %a_cmp : tensor<32x1xi1, #blocked> -> tensor<32x4xi1, #blocked>
      %a = tt.load %a_ptr, %a_mask : tensor<32x4x!tt.ptr<f32>, #blocked>
      %a_red = "tt.reduce"(%a) <{axis = 1 : i32}> ({
      ^bb0(%l: f32, %r: f32):
        %s = arith.addf %l, %r : f32
        tt.reduce.return %s : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %sum_upd = arith.addf %sum_acc, %a_red : tensor<32xf32, #slice>
      %sum_col = tt.expand_dims %sum_acc {axis = 1 : i32} : tensor<32xf32, #slice> -> tensor<32x1xf32, #blocked>
      %b_cmp = arith.cmpf ogt, %sum_col, %thresh : tensor<32x1xf32, #blocked>
      %b_mask = tt.broadcast %b_cmp : tensor<32x1xi1, #blocked> -> tensor<32x4xi1, #blocked>
      %b = tt.load %b_ptr, %b_mask : tensor<32x4x!tt.ptr<f32>, #blocked>
      %b_red = "tt.reduce"(%b) <{axis = 1 : i32}> ({
      ^bb0(%l: f32, %r: f32):
        %m = arith.maximumf %l, %r : f32
        tt.reduce.return %m : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %max_upd = arith.maximumf %max_acc, %b_red : tensor<32xf32, #slice>
      scf.yield %sum_upd, %max_upd : tensor<32xf32, #slice>, tensor<32xf32, #slice>
    }
    tt.store %sum_out, %result#0 : tensor<32x!tt.ptr<f32>, #slice>
    tt.store %max_out, %result#1 : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The reduction axis is entirely in the registers of one thread
// CHECK-LABEL: @thread_local_axis
// CHECK-NOT: tt.reshape
// CHECK: "tt.reduce"({{.*}}) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @thread_local_axis(%values: tensor<32x32x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x32x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x32xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The reduction axis has one element per thread, but distributed across multiple threads in a warp
// CHECK-LABEL: @lane_spread_axis
// CHECK: %[[RES:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<32x32xf32
// CHECK: "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK: arith.addf %[[ACC]], %{{.*}}
// CHECK: scf.yield
// CHECK: "tt.reduce"(%[[RES]]) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @lane_spread_axis(%values: tensor<32x32x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x32x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x32xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The reduction axis has one element per thread, but distributed across multiple warps
// CHECK-LABEL: @warp_spread_axis
// CHECK: %[[RES:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<32x4xf32
// CHECK: "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK: arith.addf %[[ACC]], %{{.*}}
// CHECK: scf.yield
// CHECK: "tt.reduce"(%[[RES]]) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @warp_spread_axis(%values: tensor<32x4x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x4x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x4xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}

// -----

// The reduction axis has one element per thread, but distributed across multiple CTAs
// CHECK-LABEL: @cta_split_axis
// CHECK: %[[RES:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (tensor<32x2xf32
// CHECK: "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK: arith.addf %[[ACC]], %{{.*}}
// CHECK: scf.yield
// CHECK: "tt.reduce"(%[[RES]]) <{axis = 1 : i32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cta_split_axis(%values: tensor<32x64x!tt.ptr<f32>, #blocked>, %output: tensor<32x!tt.ptr<f32>, #slice>, %count: i32) {
    %zero = arith.constant dense<0.000000e+00> : tensor<32xf32, #slice>
    %start = arith.constant 0 : i32
    %step = arith.constant 1 : i32
    %result = scf.for %index = %start to %count step %step iter_args(%accumulator = %zero) -> (tensor<32xf32, #slice>) : i32 {
      %v = tt.load %values : tensor<32x64x!tt.ptr<f32>, #blocked>
      %reduced = "tt.reduce"(%v) <{axis = 1 : i32}> ({
      ^bb0(%a: f32, %b: f32):
        %sum = arith.addf %a, %b : f32
        tt.reduce.return %sum : f32
      }) : (tensor<32x64xf32, #blocked>) -> tensor<32xf32, #slice>
      %updated = arith.addf %accumulator, %reduced : tensor<32xf32, #slice>
      scf.yield %updated : tensor<32xf32, #slice>
    }
    tt.store %output, %result : tensor<32x!tt.ptr<f32>, #slice>
    tt.return
  }
}
