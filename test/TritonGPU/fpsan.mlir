// RUN: split-file %s %t
// RUN: triton-opt %t/success.mlir -split-input-file -tritoninstrument-fp-sanitizer | FileCheck %t/success.mlir

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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @binary_ops
  tt.func public @binary_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: tt.bitcast
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
  // CHECK-LABEL: @div_rem_ops
  tt.func public @div_rem_ops(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: tt.bitcast
    // CHECK: arith.xori
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
    // CHECK: tt.bitcast
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
    // CHECK-DAG: arith.constant dense<1069066811>
    // CHECK-DAG: arith.constant dense<1>
    // CHECK-DAG: arith.constant dense<0>
    // CHECK-DAG: arith.constant dense<-1555856531>
    // CHECK: tt.bitcast
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
    // CHECK: tt.bitcast
    // CHECK: arith.extui
    // CHECK-NOT: arith.extf
    %0 = arith.extf %a : tensor<4xf16> to tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cast_truncf
  tt.func public @cast_truncf(%a: tensor<4xf32>) -> tensor<4xf16> {
    // CHECK: tt.bitcast
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
    // CHECK: tt.bitcast
    // CHECK: arith.extui
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
  // CHECK: tt.bitcast
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a {libname = "", libpath = "", pure = true, symbol = "__nv_tanf"} : (tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_binary
tt.func public @extern_binary(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "__nv_atan2f"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_ternary
tt.func public @extern_ternary(%a: tensor<4xf32>, %b: tensor<4xf32>, %c: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b, %c {libname = "", libpath = "", pure = true, symbol = "__nv_fmaf"} : (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: @extern_mixed
tt.func public @extern_mixed(%a: tensor<4xf32>, %b: tensor<4xi32>) -> tensor<4xf32> {
  // CHECK: tt.bitcast
  // CHECK: arith.addi
  // CHECK: arith.xori
  // CHECK-NOT: tt.extern_elementwise
  %0 = tt.extern_elementwise %a, %b {libname = "", libpath = "", pure = true, symbol = "__nv_ldexpf"} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
  tt.return %0 : tensor<4xf32>
}
