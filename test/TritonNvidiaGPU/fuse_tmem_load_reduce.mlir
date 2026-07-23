// RUN: triton-opt %s -split-input-file --triton-nvidia-tmem-load-reduce --allow-unregistered-dialect | FileCheck %s

// Fuse tmem_load + tt.reduce -> tmem_load, with redOp=max. The combiner
// for "arith.maxnumf" ignores NaNs, so the fused op does not set the NaN
// attribute.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_max(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {redOp = #ttng.redOp<max>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   tt.return %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_max(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Fuse (tmem_load, tt.reduce) -> tmem_load, with redOp=min
// and NaN=true, and arith.minimumf is the NaN-propagating variant.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_min_nan(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {NaN = true, redOp = #ttng.redOp<min>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   tt.return %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_min_nan(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.minimumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Look through "ttg.convert_layout" that sits between the "tmem_load" and the
// "tt.reduce" and check that the combine happens.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @tmem_load_reduce_fuse_through_cvt(
  // CHECK-SAME:    %[[ARG0:.+]]: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>)
  // CHECK-SAME:    -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT:   %{{.+}}, %[[RED:.+]] = ttng.tmem_load %[[ARG0]] {NaN = true, redOp = #ttng.redOp<max>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT:   %[[CVT:.+]] = ttg.convert_layout %[[RED]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT:   tt.return %[[CVT]] : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  // CHECK-NEXT: }
  tt.func public @tmem_load_reduce_fuse_through_cvt(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %cvt = ttg.convert_layout %0 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #linear>
    %1 = "tt.reduce"(%cvt) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maximumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
  }
}

// -----

// Negative test: check that that we do not fuse and generate "tcgen05.ld.red" if
// the target isn't sm103+.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_sm100
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_sm100(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: "arith.addf" is not a supported combiner.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_addf
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_addf(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.addf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test and current Ttriton limitation: make sure we don't
// fuse when the element is a i32 as the reduction on operation tmem_load
// is documented as f32-only.
// Even with a max/min-style integer combiner on sm103, the pattern must bail.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_i32
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_i32(%arg0: !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory> -> tensor<128x128xi32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: i32, %rhs: i32):
      %2 = arith.maxsi %lhs, %rhs : i32
      tt.reduce.return %2 : i32
    }) : (tensor<128x128xi32, #blocked>) -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: reduction along axis 0 (the M axis) must not fuse. The
// fused tcgen05.ld.red only reduces along the inner N axis.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_axis0
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_axis0(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
  }
}

// -----

// Negative test for the "already fused" bail-out.

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_already_fused
  // CHECK: %{{.+}}, %{{.+}} = ttng.tmem_load{{.+}}redOp = #ttng.redOp<max>
  // CHECK-NOT: ttng.tmem_load
  // CHECK: "tt.reduce"
  // CHECK-NOT: "tt.reduce"
  // CHECK: tt.return
  tt.func public @tmem_load_reduce_no_fuse_already_fused(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %a = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %a : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %b = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %b : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1, %2 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}

// -----

// Negative test: the layout below splits the N dimension across registers and warps,
// 2 warps along N, so N is not entirely in register.

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_load_reduce_no_fuse_n_sharded_across_warps
  // CHECK: ttng.tmem_load
  // CHECK-NOT: redOp
  // CHECK: "tt.reduce"
  tt.func public @tmem_load_reduce_no_fuse_n_sharded_across_warps(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> {
    %0 = ttng.tmem_load %arg0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory> -> tensor<128x128xf32, #blocked>
    %1 = "tt.reduce"(%0) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %2 = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %2 : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    tt.return %1 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  }
}
