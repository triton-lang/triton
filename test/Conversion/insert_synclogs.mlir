// RUN: triton-opt %s -split-input-file --insert-synclogs | FileCheck %s

// Insert synclog.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @init_barrier(%arg0: !llvm.struct<(ptr<3>, i32)>, %arg1: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
    %0 = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    %1 = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<3>, i32)>
    %2 = nvvm.read.ptx.sreg.tid.x : i32
    %3 = llvm.mlir.constant(0 : i32) : i32
    %4 = llvm.icmp "eq" %2, %3 : i32
    %5 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1; \09\0A", "b,r" %4, %0 : (i1, !llvm.ptr<3>) -> !llvm.void
    llvm.return
  }
}

// CHECK-LABEL:   llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// CHECK-LABEL:   llvm.func @init_barrier(
// CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<(ptr<3>, i32)>, %[[VAL_1:.*]]: !llvm.ptr<1>, %[[VAL_2:.*]]: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(ptr<3>, i32)>
// CHECK:           %[[VAL_4:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(ptr<3>, i32)>
// CHECK:           %[[VAL_5:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_6:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_7:.*]] = llvm.icmp "eq" %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           %[[VAL_8:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u64 $0, %[[VAL_9:.*]];", "=l"  : () -> i64
// CHECK:           %[[VAL_10:.*]] = llvm.trunc %[[VAL_8]] : i64 to i32
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(32 : i64) : i64
// CHECK:           %[[VAL_12:.*]] = llvm.lshr %[[VAL_8]], %[[VAL_11]] : i64
// CHECK:           %[[VAL_13:.*]] = llvm.trunc %[[VAL_12]] : i64 to i32
// CHECK:           %[[VAL_14:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:           %[[VAL_15:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_16:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK:           %[[VAL_17:.*]] = nvvm.read.ptx.sreg.tid.y : i32
// CHECK:           %[[VAL_18:.*]] = nvvm.read.ptx.sreg.ctaid.z : i32
// CHECK:           %[[VAL_19:.*]] = nvvm.read.ptx.sreg.tid.z : i32
// CHECK:           %[[VAL_20:.*]] = nvvm.read.ptx.sreg.cluster.ctarank : i32
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(32 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.urem %[[VAL_15]], %[[VAL_21]] : i32
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_24:.*]] = llvm.icmp "eq" %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_25:.*]] = llvm.icmp "eq" %[[VAL_17]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_26:.*]] = llvm.icmp "eq" %[[VAL_19]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_27:.*]] = llvm.icmp "eq" %[[VAL_14]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_28:.*]] = llvm.icmp "eq" %[[VAL_16]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_29:.*]] = llvm.icmp "eq" %[[VAL_18]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_30:.*]] = llvm.and %[[VAL_24]], %[[VAL_25]] : i1
// CHECK:           %[[VAL_31:.*]] = llvm.and %[[VAL_26]], %[[VAL_27]] : i1
// CHECK:           %[[VAL_32:.*]] = llvm.and %[[VAL_28]], %[[VAL_29]] : i1
// CHECK:           %[[VAL_33:.*]] = llvm.and %[[VAL_30]], %[[VAL_31]] : i1
// CHECK:           %[[VAL_34:.*]] = llvm.and %[[VAL_32]], %[[VAL_33]] : i1
// CHECK:           llvm.cond_br %[[VAL_34]], ^bb1, ^bb4
// CHECK:         ^bb1:
// CHECK:           %[[VAL_35:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_36:.*]] = llvm.atomicrmw add %[[VAL_1]], %[[VAL_35]] monotonic : !llvm.ptr<1>, i32
// CHECK:           %[[VAL_37:.*]] = llvm.add %[[VAL_36]], %[[VAL_35]] : i32
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(67108864 : i32) : i32
// CHECK:           %[[VAL_39:.*]] = llvm.icmp "ult" %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:           llvm.cond_br %[[VAL_39]], ^bb2, ^bb3
// CHECK:         ^bb2:
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_41:.*]] = llvm.add %[[VAL_40]], %[[VAL_36]] : i32
// CHECK:           %[[VAL_42:.*]] = llvm.getelementptr %[[VAL_1]]{{\[}}%[[VAL_41]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           %[[VAL_43:.*]] = llvm.mlir.constant(13 : i32) : i32
// CHECK:           %[[VAL_44:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_45:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_44]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_43]], %[[VAL_45]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_46:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_47:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_48:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_47]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_46]], %[[VAL_48]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_49:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_50:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_49]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_50]] : i1, !llvm.ptr<1>
// CHECK:           %[[VAL_51:.*]] = llvm.ptrtoint %[[VAL_3]] : !llvm.ptr<3> to i32
// CHECK:           %[[VAL_52:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_53:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_52]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_51]], %[[VAL_53]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_54:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_55:.*]] = llvm.getelementptr %[[VAL_42]]{{\[}}%[[VAL_54]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           %[[VAL_56:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_57:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_56]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_57]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_58:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_59:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_58]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_13]], %[[VAL_59]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_60:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_61:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_60]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_15]], %[[VAL_61]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_62:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_63:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_62]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_17]], %[[VAL_63]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_64:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_65:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_64]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_19]], %[[VAL_65]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_66:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_67:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_66]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_14]], %[[VAL_67]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_68:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_69:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_68]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_16]], %[[VAL_69]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_70:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_71:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_70]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_18]], %[[VAL_71]] : i32, !llvm.ptr<1>
// CHECK:           %[[VAL_72:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_73:.*]] = llvm.getelementptr %[[VAL_55]]{{\[}}%[[VAL_72]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
// CHECK:           llvm.store %[[VAL_20]], %[[VAL_73]] : i32, !llvm.ptr<1>
// CHECK:           llvm.br ^bb4
// CHECK:         ^bb3:
// CHECK:           %[[VAL_74:.*]] = llvm.atomicrmw sub %[[VAL_1]], %[[VAL_35]] monotonic : !llvm.ptr<1>, i32
// CHECK:           llvm.br ^bb4
// CHECK:         ^bb4:
// CHECK:           %[[VAL_75:.*]] = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1; \09\0A", "b,r" %[[VAL_7]], %[[VAL_3]] : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }
