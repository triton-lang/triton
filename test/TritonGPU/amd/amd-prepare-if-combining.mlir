// RUN: triton-opt %s -split-input-file -tritonamdgpu-prepare-if-combining | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonamdgpu-prepare-if-combining -canonicalize | FileCheck %s --check-prefix=CANON


// CHECK-LABEL: op_between_ifs
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: op_between_ifs
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @op_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %0 = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %1 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %2 = tt.trans %0 {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %3 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %2, %2 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %2, %2 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %4 = ttg.convert_layout %3 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %5 = arith.addf %4, %1 : tensor<32x32xf32, #blocked>
    tt.return %5 : tensor<32x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: multiple_ops_between_ifs
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//  CHECK-NEXT: %[[CVT:.+]] = ttg.convert_layout %[[LOAD]]
//  CHECK-NEXT: tt.trans %[[CVT]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: multiple_ops_between_ifs
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @multiple_ops_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %0 = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %1 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %2 = ttg.convert_layout %0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %3 = tt.trans %2 {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %add = arith.addf %3, %3 : tensor<32x32xf32, #blocked>
      scf.yield %add : tensor<32x32xf32, #blocked>
    } else {
      %sub = arith.subf %3, %3 : tensor<32x32xf32, #blocked>
      scf.yield %sub : tensor<32x32xf32, #blocked>
    }
    %5 = arith.addf %4, %1 : tensor<32x32xf32, #blocked>
    tt.return %5 : tensor<32x32xf32, #blocked>
  }
}

// -----

// CHECK-LABEL: op_between_ifs_inside_for
//       CHECK: %[[LOAD:.+]] = ttg.local_load
//       CHECK: scf.for
//  CHECK-NEXT: tt.trans %[[LOAD]]
//  CHECK-NEXT: scf.if
// CANON-LABEL: op_between_ifs_inside_for
//       CANON: scf.for
//       CANON: scf.if
//   CANON-NOT: scf.if
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @op_between_ifs_inside_for(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %a) -> tensor<32x32xf32, #blocked> {
      %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
        %mul = arith.mulf %acc, %acc : tensor<32x32xf32, #blocked>
        scf.yield %mul : tensor<32x32xf32, #blocked>
      } else {
        %div = arith.divf %acc, %acc : tensor<32x32xf32, #blocked>
        scf.yield %div : tensor<32x32xf32, #blocked>
      }
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
      scf.yield %4 : tensor<32x32xf32, #blocked>
    }
    tt.return %result : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: ifs in different blocks
// CHECK-LABEL: ifs_in_different_blocks
//       CHECK: scf.if
//       CHECK: scf.for
//  CHECK-NEXT: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: ifs_in_different_blocks
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ifs_in_different_blocks(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %0) -> tensor<32x32xf32, #blocked> {
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      %4 = arith.addf %3, %acc : tensor<32x32xf32, #blocked>
      scf.yield %4 : tensor<32x32xf32, #blocked>
    }
    tt.return %result : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: nested ifs
// CHECK-LABEL: nested_ifs
//       CHECK: scf.if
//  CHECK-NEXT: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: nested_ifs
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @nested_ifs(%cond: i1, %cond2: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
      %2 = scf.if %cond2 -> tensor<32x32xf32, #blocked_transposed> {
        %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %add : tensor<32x32xf32, #blocked_transposed>
      } else {
        %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
        scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
      }
      %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
      scf.yield %3 : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    tt.return %0 : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: ifs with different conditions
// CHECK-LABEL: different_conditions
//       CHECK: scf.if
//  CHECK-NEXT: arith.mulf
//       CHECK: tt.trans
//  CHECK-NEXT: scf.if
// CANON-LABEL: different_conditions
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @different_conditions(%cond1: i1, %cond2: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond1 -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    %2 = scf.if %cond2 -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
    tt.return %4 : tensor<32x32xf32, #blocked>
  }
}

// -----

// Negative test: non-pure op (local_store) between ifs
// CHECK-LABEL: non_pure_op_between_ifs
//       CHECK: scf.if
//       CHECK: tt.trans
//  CHECK-NEXT: ttg.local_store
//  CHECK-NEXT: scf.if
// CANON-LABEL: non_pure_op_between_ifs
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_transposed = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared_transposed = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @non_pure_op_between_ifs(%cond: i1, %smem: !ttg.memdesc<32x32xf32, #shared, #smem>, %smem_trans: !ttg.memdesc<32x32xf32, #shared_transposed, #smem, mutable>, %a: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked> {
    %x = ttg.local_load %smem : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #blocked>
    %0 = scf.if %cond -> tensor<32x32xf32, #blocked> {
      %mul = arith.mulf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %mul : tensor<32x32xf32, #blocked>
    } else {
      %div = arith.divf %a, %a : tensor<32x32xf32, #blocked>
      scf.yield %div : tensor<32x32xf32, #blocked>
    }
    %1 = tt.trans %x {order = array<i32: 1, 0>} : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_transposed>
    ttg.local_store %1, %smem_trans : tensor<32x32xf32, #blocked_transposed> -> !ttg.memdesc<32x32xf32, #shared_transposed, #smem, mutable>
    %2 = scf.if %cond -> tensor<32x32xf32, #blocked_transposed> {
      %add = arith.addf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %add : tensor<32x32xf32, #blocked_transposed>
    } else {
      %sub = arith.subf %1, %1 : tensor<32x32xf32, #blocked_transposed>
      scf.yield %sub : tensor<32x32xf32, #blocked_transposed>
    }
    %3 = ttg.convert_layout %2 : tensor<32x32xf32, #blocked_transposed> -> tensor<32x32xf32, #blocked>
    %4 = arith.addf %3, %0 : tensor<32x32xf32, #blocked>
    tt.return %4 : tensor<32x32xf32, #blocked>
  }
}
