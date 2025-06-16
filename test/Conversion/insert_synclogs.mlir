// RUN: triton-opt %s -split-input-file --insert-synclogs | FileCheck %s

// Insert random delay after.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 11 : i32} {
  llvm.func @insert_synclog() attributes {allocation.offset = 32 : i32} {
    nvvm.barrier0
    llvm.return
  }
}

// CHECK-LABEL:   llvm.func @vprintf(!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.mlir.global internal constant @printfFormat_0("nvvm.barrier0 time=%[[VAL_0:.*]] thread=%[[VAL_1:.*]],%[[VAL_1]],%[[VAL_1]] block=%[[VAL_1]],%[[VAL_1]],%[[VAL_1]] \0A\00") {addr_space = 0 : i32}

// CHECK-LABEL:   llvm.func @insert_synclog() attributes {allocation.offset = 32 : i32} {
// CHECK:           %[[VAL_0:.*]] = llvm.inline_asm has_side_effects asm_dialect = att "mov.u64 $0, %[[VAL_1:.*]];", "=l"  : () -> i64
// CHECK:           %[[VAL_2:.*]] = nvvm.read.ptx.sreg.ctaid.x : i32
// CHECK:           %[[VAL_3:.*]] = nvvm.read.ptx.sreg.tid.x : i32
// CHECK:           %[[VAL_4:.*]] = nvvm.read.ptx.sreg.ctaid.y : i32
// CHECK:           %[[VAL_5:.*]] = nvvm.read.ptx.sreg.tid.y : i32
// CHECK:           %[[VAL_6:.*]] = nvvm.read.ptx.sreg.ctaid.z : i32
// CHECK:           %[[VAL_7:.*]] = nvvm.read.ptx.sreg.tid.z : i32
// CHECK:           %[[VAL_8:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.addressof @printfFormat_0 : !llvm.ptr
// CHECK:           %[[VAL_10:.*]] = llvm.getelementptr %[[VAL_9]]{{\[}}%[[VAL_8]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK:           %[[VAL_11:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_12:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_13:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           %[[VAL_14:.*]] = llvm.alloca %[[VAL_11]] x !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)> : (i32) -> !llvm.ptr
// CHECK:           %[[VAL_15:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:           %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_0]], %[[VAL_16]] : i64, !llvm.ptr
// CHECK:           %[[VAL_17:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:           %[[VAL_18:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_2]], %[[VAL_18]] : i32, !llvm.ptr
// CHECK:           %[[VAL_19:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:           %[[VAL_20:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 2] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_3]], %[[VAL_20]] : i32, !llvm.ptr
// CHECK:           %[[VAL_21:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:           %[[VAL_22:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 3] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_4]], %[[VAL_22]] : i32, !llvm.ptr
// CHECK:           %[[VAL_23:.*]] = llvm.mlir.constant(4 : i32) : i32
// CHECK:           %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 4] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_5]], %[[VAL_24]] : i32, !llvm.ptr
// CHECK:           %[[VAL_25:.*]] = llvm.mlir.constant(5 : i32) : i32
// CHECK:           %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 5] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_6]], %[[VAL_26]] : i32, !llvm.ptr
// CHECK:           %[[VAL_27:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK:           %[[VAL_28:.*]] = llvm.getelementptr %[[VAL_14]]{{\[}}%[[VAL_12]], 6] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i64, i32, i32, i32, i32, i32, i32)>
// CHECK:           llvm.store %[[VAL_7]], %[[VAL_28]] : i32, !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.bitcast %[[VAL_14]] : !llvm.ptr to !llvm.ptr
// CHECK:           %[[VAL_30:.*]] = llvm.call @vprintf(%[[VAL_10]], %[[VAL_29]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:           nvvm.barrier0
// CHECK:           llvm.return
// CHECK:         }
