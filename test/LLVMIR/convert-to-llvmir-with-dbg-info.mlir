// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | \
// RUN: mlir-translate --mlir-to-llvmir | FileCheck %s

// NOTE: that we have to enable both --enable-line-info --extract-variable-info
// to get DILocation and DILocalVariable when converting LLVMIR otherwise they
// will be dropped


module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @add_kernel(%arg0: !llvm.ptr<1> loc(#loc10), %arg1: !llvm.ptr<1> loc(#loc11), %arg2: !llvm.ptr<1> loc(#loc12), %arg3: i32 loc(#loc13), %arg4: !llvm.ptr<1>) {
    // CHECK-DAG: distinct !DISubprogram({{.*}}, retainedNodes:
    // CHECK-DAG: !DISubroutineType(cc: DW_CC_normal, types:
    // CHECK-DAG: !DIDerivedType(tag: DW_TAG_pointer_type, name: "pointer",
    // CHECK-DAG: !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

    // CHECK: !DILocalVariable(name: "x_ptr", arg: 1, scope:
    // CHECK: !DILocalVariable(name: "y_ptr", arg: 2, scope:
    // CHECK: !DILocalVariable(name: "out_ptr", arg: 3, scope:
    // CHECK: !DILocalVariable(name: "n_elements", arg: 4, scope:

    %constant_i32 = llvm.mlir.constant(9 : i32) : i32
    %constant_i16 = llvm.mlir.constant(0 : i16) : i16
    %constant_i64 = llvm.mlir.constant(9 : i64) : i64

    // CHECK: !DILocalVariable(name: "pid", scope:
    %pid = rocdl.workgroup.id.x : i32 loc(#loc14)

    // CHECK: !DILocalVariable(name: "block_start", scope:
    %block_start = llvm.mul %pid, %constant_i32 : i32 loc(#loc15)

    // CHECK: !DILocalVariable(name: "offsets", scope:
    %offsets = llvm.add %block_start, %constant_i32 : i32 loc(#loc16)

    // CHECK: !DILocalVariable(name: "mask", scope:
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32 loc(#loc17)
    %mask_i1 = llvm.select %mask, %constant_i32, %constant_i32 : i1, i32 loc(#loc18)

    // CHECK: !DILocalVariable(name: "x", scope:
    %x_ptr = llvm.getelementptr %arg0[%block_start] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %x_buffer_ptr = rocdl.make.buffer.rsrc %x_ptr, %constant_i16, %constant_i64, %constant_i32 : <1> to <8> loc(#loc18)
    %x_val = rocdl.raw.ptr.buffer.load %x_buffer_ptr, %mask_i1, %constant_i32, %constant_i32 : vector<4xf32> loc(#loc18)
    %x_scalar = llvm.extractelement %x_val[%constant_i32 : i32] : vector<4xf32> loc(#loc18)

    // CHECK: !DILocalVariable(name: "y", scope:
    %y_ptr = llvm.getelementptr %arg1[%block_start] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %y_buffer_ptr = rocdl.make.buffer.rsrc %y_ptr, %constant_i16, %constant_i64, %constant_i32 : <1> to <8> loc(#loc19)
    %y_val = rocdl.raw.ptr.buffer.load %y_buffer_ptr, %mask_i1, %constant_i32, %constant_i32 : vector<4xf32> loc(#loc19)
    %y_scalar = llvm.extractelement %y_val[%constant_i32 : i32] : vector<4xf32> loc(#loc19)

    // CHECK: !DILocalVariable(name: "output", scope:
    %output = llvm.fadd %x_scalar, %y_scalar : f32 loc(#loc20)

    llvm.return
  }
}
#loc = loc("01-vector-add.py":30:0)
#loc2 = loc("01-vector-add.py":39:10)
#loc3 = loc("01-vector-add.py":44:18)
#loc5 = loc("01-vector-add.py":45:14)
#loc6 = loc("01-vector-add.py":47:11)
#loc7 = loc("01-vector-add.py":50:8)
#loc8 = loc("01-vector-add.py":51:8)
#loc9 = loc("01-vector-add.py":52:13)
#loc10 = loc("x_ptr"(#loc))
#loc11 = loc("y_ptr"(#loc))
#loc12 = loc("out_ptr"(#loc))
#loc13 = loc("n_elements"(#loc))
#loc14 = loc("pid"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("offsets"(#loc5))
#loc17 = loc("mask"(#loc6))
#loc18 = loc("x"(#loc7))
#loc19 = loc("y"(#loc8))
#loc20 = loc("output"(#loc9))
