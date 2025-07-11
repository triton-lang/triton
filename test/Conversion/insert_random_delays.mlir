// RUN: triton-opt %s -split-input-file --insert-random-delays | FileCheck %s

// Insert random delay after.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {
  llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
    %0 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "barrier.sync 1 ;", ""  : () -> !llvm.void
    llvm.return
  }
}

// CHECK-LABEL:   llvm.func @murmurhash3_insert(
// CHECK-SAME:  %[[INITIAL_HASH:.*]]: i32, %[[PART:.*]]: i32) -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[HEX_CC9E2D51:.*]] = llvm.mlir.constant(-862048943 : i32) : i32
// CHECK:           %[[MUL_1:.*]] = llvm.mul %[[PART]], %[[HEX_CC9E2D51]] : i32
// CHECK:           %[[INT_15:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[SHL_1:.*]] = llvm.shl %[[MUL_1]], %[[INT_15]] : i32
// CHECK:           %[[INT_17:.*]] = llvm.mlir.constant(17 : i32) : i32
// CHECK:           %[[LSHR_1:.*]] = llvm.lshr %[[MUL_1]], %[[INT_17]] : i32
// CHECK:           %[[OR_1:.*]] = llvm.or %[[SHL_1]], %[[LSHR_1]] : i32
// CHECK:           %[[HEX_1B873593:.*]] = llvm.mlir.constant(461845907 : i32) : i32
// CHECK:           %[[MUL_2:.*]] = llvm.mul %[[OR_1]], %[[HEX_1B873593]] : i32
// CHECK:           %[[XOR_1:.*]] = llvm.xor %[[INITIAL_HASH]], %[[MUL_2]] : i32
// CHECK:           %[[INT_13:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[SHL_2:.*]] = llvm.shl %[[XOR_1]], %[[INT_13]] : i32
// CHECK:           %[[INT_19:.*]] = llvm.mlir.constant(19 : i32) : i32
// CHECK:           %[[LSHR_2:.*]] = llvm.lshr %[[XOR_1]], %[[INT_19]] : i32
// CHECK:           %[[OR_2:.*]] = llvm.or %[[SHL_2]], %[[LSHR_2]] : i32
// CHECK:           %[[INT_5:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[MUL_3:.*]] = llvm.mul %[[OR_2]], %[[INT_5]] : i32
// CHECK:           %[[HEX_E6546B64:.*]] = llvm.mlir.constant(-430675100 : i32) : i32
// CHECK:           %[[ADD_1:.*]] = llvm.add %[[MUL_3]], %[[HEX_E6546B64]] : i32
// CHECK:           llvm.return %[[ADD_1]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @murmurhash3_finish(
// CHECK-SAME:  %[[HASH:.*]]: i32) -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[INT_16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[LSHR_1:.*]] = llvm.lshr %[[HASH]], %[[INT_16]] : i32
// CHECK:           %[[XOR_1:.*]] = llvm.xor %[[HASH]], %[[LSHR_1]] : i32
// CHECK:           %[[HEX_85EBCA6B:.*]] = llvm.mlir.constant(-2048144789 : i32) : i32
// CHECK:           %[[MUL_1:.*]] = llvm.mul %[[XOR_1]], %[[HEX_85EBCA6B]] : i32
// CHECK:           %[[INT_13:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[LSHR_2:.*]] = llvm.lshr %[[MUL_1]], %[[INT_13]] : i32
// CHECK:           %[[XOR_2:.*]] = llvm.xor %[[MUL_1]], %[[LSHR_2]] : i32
// CHECK:           %[[HEX_C2B2AE35:.*]] = llvm.mlir.constant(-1028477387 : i32) : i32
// CHECK:           %[[MUL_2:.*]] = llvm.mul %[[XOR_2]], %[[HEX_C2B2AE35]] : i32
// CHECK:           %[[INT_16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[LSHR_3:.*]] = llvm.lshr %[[MUL_2]], %[[INT_16]] : i32
// CHECK:           %[[XOR_3:.*]] = llvm.xor %[[MUL_2]], %[[LSHR_3]] : i32
// CHECK:           llvm.return %[[XOR_3]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @state_hash() -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[INT_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[BLOCK_IDX_X:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:           %[[THREAD_IDX_X:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[INT_16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[SHL_1:.*]] = llvm.shl %[[BLOCK_IDX_X]], %[[INT_16]] : i32
// CHECK:           %[[OR_1:.*]] = llvm.or %[[SHL_1]], %[[THREAD_IDX_X]] : i32
// CHECK:           %[[INSERT_1:.*]] = llvm.call @murmurhash3_insert(%[[INT_0]], %[[OR_1]]) : (i32, i32) -> i32
// CHECK:           %[[BLOCK_IDX_Y:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK:           %[[THREAD_IDX_Y:.*]] = nvvm.read.ptx.sreg.tid.y : i32
// CHECK:           %[[INT_16_2:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[SHL_2:.*]] = llvm.shl %[[BLOCK_IDX_Y]], %[[INT_16_2]] : i32
// CHECK:           %[[OR_2:.*]] = llvm.or %[[SHL_2]], %[[THREAD_IDX_Y]] : i32
// CHECK:           %[[INSERT_2:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_1]], %[[OR_2]]) : (i32, i32) -> i32
// CHECK:           %[[BLOCK_IDX_Z:.*]] = nvvm.read.ptx.sreg.ctaid.z : i32
// CHECK:           %[[THREAD_IDX_Z:.*]] = nvvm.read.ptx.sreg.tid.z : i32
// CHECK:           %[[INT_16_3:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[SHL_3:.*]] = llvm.shl %[[BLOCK_IDX_Z]], %[[INT_16_3]] : i32
// CHECK:           %[[OR_3:.*]] = llvm.or %[[SHL_3]], %[[THREAD_IDX_Z]] : i32
// CHECK:           %[[INSERT_3:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_2]], %[[OR_3]]) : (i32, i32) -> i32
// CHECK:           %[[GLOBALTIMER_LO:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, %[[VAL_20:.*]];", "=r"  : () -> i32
// CHECK:           %[[INSERT_4:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_3]], %[[GLOBALTIMER_LO]]) : (i32, i32) -> i32
// CHECK:           %[[GLOBALTIMER_HI:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, %[[VAL_23:.*]];", "=r"  : () -> i32
// CHECK:           %[[INSERT_5:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_4]], %[[GLOBALTIMER_HI]]) : (i32, i32) -> i32
// CHECK:           %[[CLOCK_LO:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, %[[VAL_26:.*]];", "=r"  : () -> i32
// CHECK:           %[[INSERT_6:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_5]], %[[CLOCK_LO]]) : (i32, i32) -> i32
// CHECK:           %[[CLOCK_HI:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, %[[VAL_29:.*]];", "=r"  : () -> i32
// CHECK:           %[[INSERT_7:.*]] = llvm.call @murmurhash3_insert(%[[INSERT_6]], %[[CLOCK_HI]]) : (i32, i32) -> i32
// CHECK:           %[[FINISH:.*]] = llvm.call @murmurhash3_finish(%[[INSERT_7]]) : (i32) -> i32
// CHECK:           llvm.return %[[FINISH]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
// CHECK:           %[[VAL_0:.*]] = llvm.call @state_hash() : () -> i32
// CHECK:           %[[VAL_1:.*]] = llvm.zext %[[VAL_0]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(11 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_4:.*]] = llvm.trunc %[[VAL_3]] : i64 to i32
// CHECK:           llvm.inline_asm has_side_effects asm_dialect = att "nanosleep.u32 $0;", "r" %[[VAL_4]] : (i32) -> ()
// CHECK:           %[[VAL_5:.*]] = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "barrier.sync 1 ;", ""  : () -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }

// -----

// Insert random delay before.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {
  llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
    %0 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "barrier.arrive 1 ;", ""  : () -> !llvm.void
    llvm.return
  }
}

// CHECK-LABEL:   llvm.func @rewrite_barriers() attributes {allocation.offset = 32 : i32} {
// CHECK:           %[[VAL_0:.*]] = llvm.call @state_hash() : () -> i32
// CHECK:           %[[VAL_1:.*]] = llvm.zext %[[VAL_0]] : i32 to i64
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(11 : i64) : i64
// CHECK:           %[[VAL_3:.*]] = llvm.lshr %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_4:.*]] = llvm.trunc %[[VAL_3]] : i64 to i32
// CHECK:           llvm.inline_asm has_side_effects asm_dialect = att "nanosleep.u32 $0;", "r" %[[VAL_4]] : (i32) -> ()
// CHECK:           %[[VAL_5:.*]] = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "barrier.arrive 1 ;", ""  : () -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }
