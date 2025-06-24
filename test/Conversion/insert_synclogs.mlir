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

// CHECK-LABEL:   llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.mlir.global internal constant @printfFormat_0("@$0 mbarrier.init.shared::cta.b64 [$1], 1;  time=%[[VAL_0:.*]] thread=%[[VAL_1:.*]],%[[VAL_1]],%[[VAL_1]] block=%[[VAL_1]],%[[VAL_1]],%[[VAL_1]] cta_rank=%[[VAL_1]] %[[VAL_2:.*]] %[[VAL_3:.*]] \00") {addr_space = 0 : i32}
// CHECK:         llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL:   llvm.func @init_barrier(
// CHECK-SAME:  %[[VAL_0:.*]]: !llvm.struct<(ptr<3>, i32)>, %[[VAL_1:.*]]: !llvm.ptr<1>) attributes {nvvm.kernel = 1 : ui1, nvvm.reqntid = array<i32: 128>} {
// CHECK:           %[[VAL_2:.*]] = llvm.extractvalue %[[VAL_0]][0] : !llvm.struct<(ptr<3>, i32)>
// CHECK:           %[[VAL_3:.*]] = llvm.extractvalue %[[VAL_0]][1] : !llvm.struct<(ptr<3>, i32)>
// CHECK:           %[[VAL_4:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_5:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_6:.*]] = llvm.icmp "eq" %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_7:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u64 $0, %[[VAL_8:.*]];", "=l"  : () -> i64
// CHECK:           %[[VAL_9:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:           %[[VAL_10:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_11:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK:           %[[VAL_12:.*]] = nvvm.read.ptx.sreg.tid.y : i32
// CHECK:           %[[VAL_13:.*]] = nvvm.read.ptx.sreg.ctaid.z : i32
// CHECK:           %[[VAL_14:.*]] = nvvm.read.ptx.sreg.tid.z : i32
// CHECK:           %[[VAL_15:.*]] = nvvm.read.ptx.sreg.cluster.ctarank : i32
// CHECK:           %[[VAL_16:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_17]]{{\[}}%[[VAL_16]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_22:.*]] = llvm.sext %[[VAL_6]] : i1 to i32
// CHECK:           %[[VAL_23:.*]] = llvm.alloca %[[VAL_19]] x !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_24:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_25:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_25]] : i64, !llvm.ptr
// CHECK:           %[[VAL_26:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_27:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_10]], %[[VAL_27]] : i32, !llvm.ptr
// CHECK:           %[[VAL_28:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_29:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_12]], %[[VAL_29]] : i32, !llvm.ptr
// CHECK:           %[[VAL_30:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_31:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 3] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_14]], %[[VAL_31]] : i32, !llvm.ptr
// CHECK:           %[[VAL_32:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_33:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 4] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_9]], %[[VAL_33]] : i32, !llvm.ptr
// CHECK:           %[[VAL_34:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_35:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 5] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_11]], %[[VAL_35]] : i32, !llvm.ptr
// CHECK:           %[[VAL_36:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_37:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 6] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_13]], %[[VAL_37]] : i32, !llvm.ptr
// CHECK:           %[[VAL_38:.*]] = llvm.mlir.constant(7 : i32) : i32
// CHECK:           %[[VAL_39:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 7] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_15]], %[[VAL_39]] : i32, !llvm.ptr
// CHECK:           %[[VAL_40:.*]] = llvm.mlir.constant(8 : i32) : i32
// CHECK:           %[[VAL_41:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 8] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_22]], %[[VAL_41]] : i32, !llvm.ptr
// CHECK:           %[[VAL_42:.*]] = llvm.mlir.constant(9 : i32) : i32
// CHECK:           %[[VAL_43:.*]] = llvm.getelementptr %[[VAL_23]]{{\[}}%[[VAL_20]], 9] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32, i32, i32, ptr<3>)>
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_43]] : !llvm.ptr<3>, !llvm.ptr
// CHECK:           %[[VAL_44:.*]] = llvm.bitcast %[[VAL_23]] : !llvm.ptr to !llvm.ptr
// CHECK:           %[[VAL_45:.*]] = llvm.call @vprintf(%[[VAL_18]], %[[VAL_44]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           %[[VAL_46:.*]] = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$0 mbarrier.init.shared::cta.b64 [$1], 1; \09\0A", "b,r" %[[VAL_6]], %[[VAL_2]] : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK:           llvm.return
// CHECK:         }
