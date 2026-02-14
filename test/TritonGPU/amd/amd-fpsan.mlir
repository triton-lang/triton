// RUN: triton-opt %s -split-input-file -tritoninstrument-fp-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_operand_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_operand_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @unary_ops
  tt.func public @unary_ops(%a: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: tt.bitcast
    // CHECK: arith.xori
    // CHECK: arith.xori
    // CHECK-NOT: math.exp
    // CHECK-NOT: math.log
    // CHECK-NOT: math.sqrt
    %e = math.exp %a : tensor<4xf32>
    %l = math.log %e : tensor<4xf32>
    %s = math.sqrt %l : tensor<4xf32>
    tt.return %s : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @cast_extf
  tt.func public @cast_extf(%a: tensor<4xf16>) -> tensor<4xf32> {
    // CHECK: tt.bitcast
    // CHECK: arith.extui
    // CHECK: arith.shli
    // CHECK-NOT: arith.extf
    %0 = arith.extf %a : tensor<4xf16> to tensor<4xf32>
    tt.return %0 : tensor<4xf32>
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @cast_truncf
  tt.func public @cast_truncf(%a: tensor<4xf32>) -> tensor<4xf16> {
    // CHECK: tt.bitcast
    // CHECK: arith.shrui
    // CHECK: arith.trunci
    // CHECK-NOT: arith.truncf
    %0 = arith.truncf %a : tensor<4xf32> to tensor<4xf16>
    tt.return %0 : tensor<4xf16>
  }
}
