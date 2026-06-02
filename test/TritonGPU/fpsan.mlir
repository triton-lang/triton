// RUN: split-file %s %t
// RUN: triton-opt %t/success.mlir -split-input-file -tritoninstrument-fp-sanitizer | FileCheck %t/success.mlir
// RUN: triton-opt %t/canonicalize.mlir -canonicalize | FileCheck %t/canonicalize.mlir
// RUN: not triton-opt %t/unsupported.mlir -tritoninstrument-fp-sanitizer 2>&1 | FileCheck %t/unsupported.mlir --check-prefix=FPSANERR

//--- success.mlir

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @dot_emulation
  tt.func public @dot_emulation() -> tensor<16x16xf32, #blocked> {
    // CHECK: scf.for
    // CHECK-NOT: tt.dot
    // CHECK-NOT: ttg.convert_layout
    %cst = arith.constant 1.000000e+00 : f16
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %a = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_operand_a>
    %b = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_operand_b>
    %out = tt.dot %a, %b, %zero : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #blocked>
    tt.return %out : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rank3_dot_emulation
  tt.func public @rank3_dot_emulation() -> tensor<2x16x16xf32, #blocked> {
    // CHECK: scf.for
    // CHECK: scf.for
    // CHECK-NOT: tt.dot
    // CHECK-NOT: ttg.convert_layout
    %one = arith.constant 1.000000e+00 : f16
    %zero = arith.constant dense<0.000000e+00> : tensor<2x16x16xf32, #blocked>
    %a = tt.splat %one : f16 -> tensor<2x16x16xf16, #dot_operand_a>
    %b = tt.splat %one : f16 -> tensor<2x16x16xf16, #dot_operand_b>
    %out = tt.dot %a, %b, %zero : tensor<2x16x16xf16, #dot_operand_a> * tensor<2x16x16xf16, #dot_operand_b> -> tensor<2x16x16xf32, #blocked>
    tt.return %out : tensor<2x16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [0, 1]}>
#dot_A = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_B = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @dot_scaled_emulation
  tt.func public @dot_scaled_emulation() -> tensor<16x16xf32, #blocked> {
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: scf.for
    // CHECK: ttg.barrier global_read|global_write
    // CHECK-NOT: ttg.dot_scaled
    // CHECK-NOT: ttg.convert_layout
     %cst = arith.constant 1.000000e+00 : f16
     %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
     %a = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_A>
     %b = tt.splat %cst : f16 -> tensor<16x16xf16, #dot_B>
     %out = tt.dot_scaled %a, %b, %zero lhs = fp16 rhs = fp16 {fastMath = false} : tensor<16x16xf16, #dot_A> * tensor<16x16xf16, #dot_B> -> tensor<16x16xf32, #blocked>
     tt.return %out : tensor<16x16xf32, #blocked>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @warp_group_dot_emulation
  tt.func public @warp_group_dot_emulation() -> tensor<64x32xf32, #mma> {
    // CHECK: ttg.local_load
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: tt.store
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: scf.for
    // CHECK: ttg.barrier global_read|global_write
    // CHECK: %[[RAW:.*]] = tt.load
    // CHECK: %[[OUT:.*]] = tti.experimental_fpsan_unembed %[[RAW]]
    // CHECK-NOT: ttng.warp_group_dot {{.*}} :
    // CHECK: ttng.warp_group_dot_wait %[[OUT]]
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x32xf32, #shared, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %c = arith.constant dense<0.000000e+00> : tensor<64x32xf32, #mma>
    %true = arith.constant true
    %d = ttng.warp_group_dot %a, %b, %c, %true {inputPrecision = 1 : i32, isAsync = true} : !ttg.memdesc<64x32xf32, #shared, #smem, mutable> * !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<64x32xf32, #mma>
    %wait = ttng.warp_group_dot_wait %d {pendings = 0 : i32} : tensor<64x32xf32, #mma>
    tt.return %wait : tensor<64x32xf32, #mma>
  }
}

// -----

#tmem_linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @tmem_scratch_payloads
  tt.func public @tmem_scratch_payloads(%arg0: tensor<128x64xf32, #tmem_linear>) -> tensor<128x64xf32, #tmem_linear> {
    // CHECK: %[[PAYLOAD:.*]] = tti.experimental_fpsan_embed %arg0
    // CHECK: tt.store {{.*}}, %[[PAYLOAD]]
    // CHECK: %[[RAW:.*]] = tt.load
    // CHECK: %[[OUT:.*]] = tti.experimental_fpsan_unembed %[[RAW]]
    // CHECK: tt.return %[[OUT]]
    // CHECK-NOT: ttng.tmem_store
    // CHECK-NOT: ttng.tmem_load
    %true = arith.constant true
    %tmem = ttng.tmem_alloc : () -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %arg0, %tmem, %true : tensor<128x64xf32, #tmem_linear> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %out = ttng.tmem_load %tmem : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #tmem_linear>
    tt.return %out : tensor<128x64xf32, #tmem_linear>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @binary_ops
  tt.func public @binary_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.addi
    // CHECK: arith.subi
    // CHECK: arith.muli
    // CHECK-NOT: arith.addf
    // CHECK-NOT: arith.subf
    // CHECK-NOT: arith.mulf
    %add = arith.addf %a, %b : tensor<4xf32>
    %sub = arith.subf %a, %b : tensor<4xf32>
    %mul = arith.mulf %a, %b : tensor<4xf32>
    %sum = arith.addf %add, %sub : tensor<4xf32>
    %out = arith.mulf %sum, %mul : tensor<4xf32>
    tt.return %out : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @neg_op
  tt.func public @neg_op(%a: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-DAG: %[[A:.*]] = tti.experimental_fpsan_embed %arg0 : (tensor<4xf32>) -> tensor<4xi32>
    // CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0> : tensor<4xi32>
    // CHECK: %[[NEG:.*]] = arith.subi %[[ZERO]], %[[A]] : tensor<4xi32>
    // CHECK: %[[OUT:.*]] = tti.experimental_fpsan_unembed %[[NEG]] : (tensor<4xi32>) -> tensor<4xf32>
    // CHECK-NOT: arith.negf
    %neg = arith.negf %a : tensor<4xf32>
    tt.return %neg : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @chained_ops
  tt.func public @chained_ops(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: %[[A:.*]] = tti.experimental_fpsan_embed %arg0 : (tensor<4xf32>) -> tensor<4xi32>
    // CHECK: %[[B:.*]] = tti.experimental_fpsan_embed %arg1 : (tensor<4xf32>) -> tensor<4xi32>
    // CHECK: %[[SUM0:.*]] = arith.addi %[[A]], %[[B]] : tensor<4xi32>
    // CHECK: %[[C:.*]] = tti.experimental_fpsan_embed %arg2 : (tensor<4xf32>) -> tensor<4xi32>
    // CHECK: %[[SUM1:.*]] = arith.addi %[[SUM0]], %[[C]] : tensor<4xi32>
    // CHECK: %[[OUT:.*]] = tti.experimental_fpsan_unembed %[[SUM1]] : (tensor<4xi32>) -> tensor<4xf32>
    // CHECK: tt.return %[[OUT]] : tensor<4xf32>
    %sum0 = arith.addf %a, %b : tensor<4xf32>
    %sum1 = arith.addf %sum0, %c : tensor<4xf32>
    tt.return %sum1 : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @div_rem_ops
  tt.func public @div_rem_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.muli
    // CHECK-NOT: arith.divf
    // CHECK-NOT: arith.remf
    %div = arith.divf %a, %b : tensor<4xf32>
    %rem = arith.remf %a, %b : tensor<4xf32>
    %out = arith.addf %div, %rem : tensor<4xf32>
    tt.return %out : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @fma_op
  tt.func public @fma_op(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK-NOT: math.fma
    %fma = math.fma %a, %b, %c : tensor<4xf32>
    tt.return %fma : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @unary_ops
  tt.func public @unary_ops(%a: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK-DAG: arith.constant dense<314159>
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.muli
    // CHECK: arith.xori
    // CHECK: arith.xori
    // CHECK-NOT: math.log
    // CHECK-NOT: math.sqrt
    %l = math.log %a : tensor<4xf32>
    %s = math.sqrt %l : tensor<4xf32>
    tt.return %s : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @exp_ops
  tt.func public @exp_ops(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
    // CHECK-DAG: arith.constant dense<594471359>
    // CHECK-DAG: arith.constant dense<1>
    // CHECK-DAG: arith.constant dense<0>
    // CHECK-DAG: arith.constant dense<-1555856531>
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.muli
    // CHECK: arith.andi
    // CHECK: arith.cmpi
    // CHECK: arith.select
    // CHECK-NOT: math.exp
    // CHECK-NOT: math.exp2
    %0 = math.exp %a : tensor<4xf32>
    %1 = math.exp2 %a : tensor<4xf32>
    tt.return %0, %1 : tensor<4xf32>, tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cast_extf
  tt.func public @cast_extf(%a: tensor<4xf16>) -> tensor<4xf32> {
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.extsi
    // CHECK-NOT: arith.extf
    %0 = arith.extf %a : tensor<4xf16> to tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cast_truncf
  tt.func public @cast_truncf(%a: tensor<4xf32>) -> tensor<4xf16> {
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.trunci
    // CHECK-NOT: arith.truncf
    %0 = arith.truncf %a : tensor<4xf32> to tensor<4xf16>
    tt.return %0 : tensor<4xf16>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cast_fp_to_fp
  tt.func public @cast_fp_to_fp(%a: tensor<4xf8E4M3FN>) -> tensor<4xf16> {
    // CHECK: tti.experimental_fpsan_embed
    // CHECK: arith.extsi
    // CHECK-NOT: tt.fp_to_fp
    %0 = tt.fp_to_fp %a : tensor<4xf8E4M3FN> -> tensor<4xf16>
    tt.return %0 : tensor<4xf16>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @cast_fp4_to_fp
  tt.func public @cast_fp4_to_fp(%a: tensor<16x8xi8, #blocked>) -> tensor<16x16xf16, #blocked> {
    // CHECK: arith.andi
    // CHECK: arith.shrui
    // CHECK: tt.join
    // CHECK: tt.reshape
    // CHECK-NOT: tt.trans
    // CHECK-NOT: ttg.fp4_to_fp
    %0 = ttg.fp4_to_fp %a {axis = 1 : i32} : tensor<16x8xi8, #blocked> -> tensor<16x16xf16, #blocked>
    tt.return %0 : tensor<16x16xf16, #blocked>
  }
}

// -----

// CHECK-LABEL: @extern_unary
tt.func public @extern_unary(%a: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tti.experimental_fpsan_embed
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a {libname = "", libpath = "", pure = true, symbol = "__nv_tanf"} : (tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_binary
tt.func public @extern_binary(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tti.experimental_fpsan_embed
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_ternary
tt.func public @extern_ternary(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tti.experimental_fpsan_embed
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b, %c {libname = "", libpath = "", pure = true, symbol = "__nv_fmaf"} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_mixed
tt.func public @extern_mixed(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: tti.experimental_fpsan_embed
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "__nv_ldexpf"} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

//--- canonicalize.mlir

module {
  // CHECK-LABEL: @fold_fpsan_embedding_roundtrips
  tt.func public @fold_fpsan_embedding_roundtrips(%arg0: tensor<4xi32>, %arg1: tensor<4xf32>) -> (tensor<4xi32>, tensor<4xf32>) {
    // CHECK-NOT: tti.experimental_fpsan
    // CHECK: tt.return %arg0, %arg1
    %0 = tti.experimental_fpsan_unembed %arg0 : (tensor<4xi32>) -> tensor<4xf32>
    %1 = tti.experimental_fpsan_embed %0 : (tensor<4xf32>) -> tensor<4xi32>
    %2 = tti.experimental_fpsan_embed %arg1 : (tensor<4xf32>) -> tensor<4xi32>
    %3 = tti.experimental_fpsan_unembed %2 : (tensor<4xi32>) -> tensor<4xf32>
    tt.return %1, %3 : tensor<4xi32>, tensor<4xf32>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @push_unembed_through_views
  tt.func public @push_unembed_through_views(
      %arg0: tensor<16x16xi32, #blocked>,
      %arg1: tensor<2x4xi32>,
      %arg2: tensor<8xi32>,
      %arg3: tensor<1x4xi32>,
      %arg4: tensor<4xi32>) ->
      (tensor<16x16xf32, #blocked1>, tensor<4x2xf32>, tensor<2x4xf32>,
       tensor<2x4xf32>, tensor<1x4xf32>) {
    // CHECK: %[[CVT:.*]] = ttg.convert_layout %arg0 : tensor<16x16xi32, #blocked> -> tensor<16x16xi32, #blocked1>
    // CHECK: %[[CVT_OUT:.*]] = tti.experimental_fpsan_unembed %[[CVT]] : (tensor<16x16xi32, #blocked1>) -> tensor<16x16xf32, #blocked1>
    // CHECK: %[[TRANS:.*]] = tt.trans %arg1 {order = array<i32: 1, 0>} : tensor<2x4xi32> -> tensor<4x2xi32>
    // CHECK: %[[TRANS_OUT:.*]] = tti.experimental_fpsan_unembed %[[TRANS]] : (tensor<4x2xi32>) -> tensor<4x2xf32>
    // CHECK: %[[RESHAPE:.*]] = tt.reshape %arg2 : tensor<8xi32> -> tensor<2x4xi32>
    // CHECK: %[[RESHAPE_OUT:.*]] = tti.experimental_fpsan_unembed %[[RESHAPE]] : (tensor<2x4xi32>) -> tensor<2x4xf32>
    // CHECK: %[[BCAST:.*]] = tt.broadcast %arg3 : tensor<1x4xi32> -> tensor<2x4xi32>
    // CHECK: %[[BCAST_OUT:.*]] = tti.experimental_fpsan_unembed %[[BCAST]] : (tensor<2x4xi32>) -> tensor<2x4xf32>
    // CHECK: %[[EXPAND:.*]] = tt.expand_dims %arg4 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
    // CHECK: %[[EXPAND_OUT:.*]] = tti.experimental_fpsan_unembed %[[EXPAND]] : (tensor<1x4xi32>) -> tensor<1x4xf32>
    %0 = tti.experimental_fpsan_unembed %arg0 : (tensor<16x16xi32, #blocked>) -> tensor<16x16xf32, #blocked>
    %1 = ttg.convert_layout %0 : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    %2 = tti.experimental_fpsan_unembed %arg1 : (tensor<2x4xi32>) -> tensor<2x4xf32>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<2x4xf32> -> tensor<4x2xf32>
    %4 = tti.experimental_fpsan_unembed %arg2 : (tensor<8xi32>) -> tensor<8xf32>
    %5 = tt.reshape %4 : tensor<8xf32> -> tensor<2x4xf32>
    %6 = tti.experimental_fpsan_unembed %arg3 : (tensor<1x4xi32>) -> tensor<1x4xf32>
    %7 = tt.broadcast %6 : tensor<1x4xf32> -> tensor<2x4xf32>
    %8 = tti.experimental_fpsan_unembed %arg4 : (tensor<4xi32>) -> tensor<4xf32>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<4xf32> -> tensor<1x4xf32>
    tt.return %1, %3, %5, %7, %9 : tensor<16x16xf32, #blocked1>, tensor<4x2xf32>,
                                          tensor<2x4xf32>, tensor<2x4xf32>, tensor<1x4xf32>
  }
}

//--- unsupported.mlir

module {
  tt.func public @dot_no_encoding(%a: tensor<16x16xf32>, %b: tensor<16x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    // FPSANERR: error: 'tt.dot' op unsupported by fpsan
    %out = tt.dot %a, %b, %c : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %out : tensor<16x16xf32>
  }
}
