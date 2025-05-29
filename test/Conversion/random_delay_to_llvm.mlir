// RUN: triton-opt %s --convert-triton-gpu-to-llvm | FileCheck %s

#blocked0 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @random_delay_test(%a_ptr_init : tensor<256x!tt.ptr<f32>, #blocked0>, %cst : tensor<256xi1, #blocked0>, %cst_0 : tensor<256xf32, #blocked0>) {
    ttng.random_delay
    tt.return
  }
}

// CHECK-LABEL:   llvm.func @murmurhash3_insert(
// CHECK-SAME:  %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_2:.*]] = llvm.mlir.constant(-862048943 : i32) : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mul %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(15 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.shl %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(17 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.lshr %[[VAL_3]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = llvm.or %[[VAL_5]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(461845907 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = llvm.xor %[[VAL_0]], %[[VAL_10]] : i32
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.shl %[[VAL_11]], %[[VAL_12]] : i32
// CHECK:           %[[VAL_14:.*]] = llvm.mlir.constant(19 : i32) : i32
// CHECK:           %[[VAL_15:.*]] = llvm.lshr %[[VAL_11]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_16:.*]] = llvm.or %[[VAL_13]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.mul %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(-430675100 : i32) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.add %[[VAL_18]], %[[VAL_19]] : i32
// CHECK:           llvm.return %[[VAL_20]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @murmurhash3_finish(
// CHECK-SAME:  %[[VAL_0:.*]]: i32) -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_1:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_2:.*]] = llvm.lshr %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = llvm.xor %[[VAL_0]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.constant(-2048144789 : i32) : i32
// CHECK:           %[[VAL_5:.*]] = llvm.mul %[[VAL_3]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.lshr %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = llvm.xor %[[VAL_5]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(-1028477387 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.mul %[[VAL_8]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.lshr %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           %[[VAL_13:.*]] = llvm.xor %[[VAL_10]], %[[VAL_12]] : i32
// CHECK:           llvm.return %[[VAL_13]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @state_hash() -> i32 attributes {sym_visibility = "private"} {
// CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_1:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:           %[[VAL_2:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_3:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_4:.*]] = llvm.shl %[[VAL_1]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_5:.*]] = llvm.or %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_6:.*]] = llvm.call @murmurhash3_insert(%[[VAL_0]], %[[VAL_5]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_7:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK:           %[[VAL_8:.*]] = nvvm.read.ptx.sreg.tid.y : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_10:.*]] = llvm.shl %[[VAL_7]], %[[VAL_9]] : i32
// CHECK:           %[[VAL_11:.*]] = llvm.or %[[VAL_10]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_12:.*]] = llvm.call @murmurhash3_insert(%[[VAL_6]], %[[VAL_11]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_13:.*]] = nvvm.read.ptx.sreg.ctaid.z : i32
// CHECK:           %[[VAL_14:.*]] = nvvm.read.ptx.sreg.tid.z : i32
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.shl %[[VAL_13]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_17:.*]] = llvm.or %[[VAL_16]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_18:.*]] = llvm.call @murmurhash3_insert(%[[VAL_12]], %[[VAL_17]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_19:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, globaltimer_lo;", "=r"  : () -> i32
// CHECK:           %[[VAL_20:.*]] = llvm.call @murmurhash3_insert(%[[VAL_18]], %[[VAL_19]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_21:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, globaltimer_hi;", "=r"  : () -> i32
// CHECK:           %[[VAL_22:.*]] = llvm.call @murmurhash3_insert(%[[VAL_20]], %[[VAL_21]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_23:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, clock;", "=r"  : () -> i32
// CHECK:           %[[VAL_24:.*]] = llvm.call @murmurhash3_insert(%[[VAL_22]], %[[VAL_23]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_25:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u32 $0, clock_hi;", "=r"  : () -> i32
// CHECK:           %[[VAL_26:.*]] = llvm.call @murmurhash3_insert(%[[VAL_24]], %[[VAL_25]]) : (i32, i32) -> i32
// CHECK:           %[[VAL_27:.*]] = llvm.call @murmurhash3_finish(%[[VAL_26]]) : (i32) -> i32
// CHECK:           llvm.return %[[VAL_27]] : i32
// CHECK:         }

// CHECK-LABEL:   llvm.func @random_delay_test(
// CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<(ptr<1>, ptr<1>)>, %[[VAL_1:.*]]: !llvm.struct<(i1, i1)>, %[[VAL_2:.*]]: !llvm.struct<(f32, f32)>, %[[VAL_3:.*]]: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
// CHECK:           %[[VAL_4:.*]] = llvm.call @state_hash() : () -> i32
// CHECK:           %[[VAL_5:.*]] = llvm.zext %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(11 : i64) : i64
// CHECK:           %[[VAL_7:.*]] = llvm.lshr %[[VAL_5]], %[[VAL_6]] : i64
// CHECK:           %[[VAL_8:.*]] = llvm.trunc %[[VAL_7]] : i64 to i32
// CHECK:           llvm.inline_asm has_side_effects asm_dialect = att "nanosleep.u32 $0;", "r" %[[VAL_8]] : (i32) -> ()
// CHECK:           llvm.return
// CHECK:         }
