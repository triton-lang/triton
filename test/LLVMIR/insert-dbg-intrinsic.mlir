// RUN: triton-opt %s -split-input-file -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | FileCheck %s

#loc = loc("01-vector-add.py":30:0)
#loc7 = loc("x_ptr"(#loc))
#loc8 = loc("y_ptr"(#loc))
#loc9 = loc("out_ptr"(#loc))
#loc10 = loc("n_elements"(#loc))
// CHECK: #llvm.di_local_variable<{{.*}}, name = "x_ptr", {{.*}}>
// CHECK: #llvm.di_local_variable<{{.*}}, name = "y_ptr", {{.*}}>
// CHECK: #llvm.di_local_variable<{{.*}}, name = "out_ptr", {{.*}}>
// CHECK: #llvm.di_local_variable<{{.*}}, name = "n_elements", {{.*}}>
// CHECK: #llvm.di_subprogram<{{.*}} retainedNodes = {{.*}}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32 } {
  llvm.func @add_kernel(%arg0: !llvm.ptr<1> {tt.pointee_type = f32} loc(#loc7),
                        %arg1: !llvm.ptr<1> {tt.pointee_type = f32} loc(#loc8),
                        %arg2: !llvm.ptr<1> {tt.pointee_type = f32} loc(#loc9),
                        %arg3: i32 loc(#loc10), %arg4: !llvm.ptr<1>) {
    // CHECK: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %x_ptr :
    // CHECK: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %y_ptr :
    // CHECK: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %out_ptr :
    // CHECK: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %n_elements :
    %constant_i32 = llvm.mlir.constant(3 : index) : i32

    // CHECK: %pid = rocdl.workgroup.id.x
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %pid :
    %pid = rocdl.workgroup.id.x : i32 loc(#loc14)

    // CHECK: %block_start = llvm.mul %pid
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %block_start :
    %block_start = llvm.mul %pid, %constant_i32 : i32 loc(#loc15)

    // CHECK: %offsets = llvm.add %block_start
    // CHECK-NEXT: llvm.intr.dbg.value #di_local_variable{{([0-9]*)?}} = %offsets :
    %offsets = llvm.add %block_start, %constant_i32 : i32 loc(#loc16)
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32 loc(#loc17)

    llvm.return
  }
}
#loc2 = loc("01-vector-add.py":39:10)
#loc3 = loc("01-vector-add.py":44:18)
#loc5 = loc("01-vector-add.py":45:14)
#loc6 = loc("01-vector-add.py":47:11)
#loc14 = loc("pid"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("offsets"(#loc5))
#loc17 = loc("mask"(#loc6))


// -----

// COM: Check llvm struct, llvm array can be successfully converted to DIType
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // CHECK: #llvm.di_basic_type<tag = DW_TAG_base_type, name = "int"
  // CHECK: #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "struct"
  // CHECK: #llvm.di_composite_type<tag = DW_TAG_array_type, name = "array"
  // CHECK: #llvm.di_derived_type<tag = DW_TAG_pointer_type, name = "pointer"
  llvm.func @multi_arg_type_kernel(%arg0: !llvm.struct<(i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32)>,
                                %arg1: !llvm.array<4 x i8>,
                                %arg2: !llvm.ptr<1> {tt.pointee_type = i16},
                                %arg3: i32) attributes {noinline = false} {
    %constant_i32 = llvm.mlir.constant(3 : index) : i32
    %pid = rocdl.workgroup.id.x : i32
    %block_start = llvm.mul %pid, %constant_i32 : i32
    %offsets = llvm.add %block_start, %constant_i32 : i32
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32
    llvm.return
  }
}
