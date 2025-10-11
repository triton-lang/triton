// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix --enable-line-info --extract-variable-info | \
// RUN: mlir-translate --mlir-to-llvmir | FileCheck %s

// NOTE: that we have to enable both --enable-line-info --extract-variable-info
// to get DILocation and DILocalVariable when converting LLVMIR otherwise they
// will be dropped

#di_file = #llvm.di_file<"01-vector-add.py" in "">
#di_null_type = #llvm.di_null_type
#di_subroutine_type = #llvm.di_subroutine_type<callingConvention = DW_CC_normal>

#di_compile_unit = #llvm.di_compile_unit<id = distinct[0]<>, sourceLanguage = DW_LANG_C, file = #di_file,
                                         producer = "triton", isOptimized = true, emissionKind = LineTablesOnly>
#di_subprogram = #llvm.di_subprogram<id = distinct[0]<>, compileUnit = #di_compile_unit,
                                     scope = #di_file, name = "add_kernel", linkageName = "add_kernel",
                                     file = #di_file, line = 30, scopeLine = 30,
                                     subprogramFlags = "Definition|Optimized", type = #di_subroutine_type>

#di_local_variable0 = #llvm.di_local_variable<scope = #di_subprogram, name = "pid", file = #di_file, type = #di_null_type>
#di_local_variable1 = #llvm.di_local_variable<scope = #di_subprogram, name = "block_start", file = #di_file, type = #di_null_type>
#di_local_variable2 = #llvm.di_local_variable<scope = #di_subprogram, name = "offsets", file = #di_file, type = #di_null_type>
#di_local_variable3 = #llvm.di_local_variable<scope = #di_subprogram, name = "mask", file = #di_file, type = #di_null_type>
#di_local_variable4 = #llvm.di_local_variable<scope = #di_subprogram, name = "x", file = #di_file, type = #di_null_type>
#di_local_variable5 = #llvm.di_local_variable<scope = #di_subprogram, name = "y", file = #di_file, type = #di_null_type>
#di_local_variable6 = #llvm.di_local_variable<scope = #di_subprogram, name = "output", file = #di_file, type = #di_null_type>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @add_kernel(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>,
                        %arg2: !llvm.ptr<1>, %arg3: i32, %arg4: !llvm.ptr<1>) {
    %constant_i32 = llvm.mlir.constant(9 : i32) : i32
    %constant_i16 = llvm.mlir.constant(0 : i16) : i16
    %constant_i64 = llvm.mlir.constant(9 : i64) : i64

    // CHECK: !DILocalVariable(name: "pid", scope:
    %pid = rocdl.workgroup.id.x : i32 loc(#loc14)
    llvm.intr.dbg.value #di_local_variable0 = %pid : i32 loc(#loc2)

    // CHECK: !DILocalVariable(name: "block_start", scope:
    %block_start = llvm.mul %pid, %constant_i32 : i32 loc(#loc15)
    llvm.intr.dbg.value #di_local_variable1 = %block_start : i32 loc(#loc3)

    // CHECK: !DILocalVariable(name: "offsets", scope:
    %offsets = llvm.add %block_start, %constant_i32 : i32 loc(#loc16)
    llvm.intr.dbg.value #di_local_variable2 = %offsets : i32 loc(#loc5)

    // CHECK: !DILocalVariable(name: "mask", scope:
    %mask = llvm.icmp "slt" %offsets, %arg3 : i32 loc(#loc17)
    %mask_i1 = llvm.select %mask, %constant_i32, %constant_i32 : i1, i32 loc(#loc18)
    llvm.intr.dbg.value #di_local_variable3 = %mask : i1 loc(#loc6)

    // CHECK: !DILocalVariable(name: "x", scope:
    %x_ptr = llvm.getelementptr %arg0[%block_start] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %x_buffer_ptr = rocdl.make.buffer.rsrc %x_ptr, %constant_i16, %constant_i64, %constant_i32 : <1> to <8> loc(#loc18)
    llvm.intr.dbg.value #di_local_variable4 = %x_buffer_ptr : !llvm.ptr<8> loc(#loc8)
    %x_val = rocdl.raw.ptr.buffer.load %x_buffer_ptr, %mask_i1, %constant_i32, %constant_i32 : vector<4xf32> loc(#loc18)
    %x_scalar = llvm.extractelement %x_val[%constant_i32 : i32] : vector<4xf32> loc(#loc18)

    // CHECK: !DILocalVariable(name: "y", scope:
    %y_ptr = llvm.getelementptr %arg1[%block_start] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, f32
    %y_buffer_ptr = rocdl.make.buffer.rsrc %y_ptr, %constant_i16, %constant_i64, %constant_i32 : <1> to <8> loc(#loc19)
    llvm.intr.dbg.value #di_local_variable5 = %y_buffer_ptr : !llvm.ptr<8> loc(#loc10)
    %y_val = rocdl.raw.ptr.buffer.load %y_buffer_ptr, %mask_i1, %constant_i32, %constant_i32 : vector<4xf32> loc(#loc19)
    %y_scalar = llvm.extractelement %y_val[%constant_i32 : i32] : vector<4xf32> loc(#loc19)

    // CHECK: !DILocalVariable(name: "output", scope:
    %output = llvm.fadd %x_scalar, %y_scalar : f32 loc(#loc20)
    llvm.intr.dbg.value #di_local_variable6 = %output : f32 loc(#loc11)

    llvm.return
  } loc(#loc21)
}
#loc = loc("01-vector-add.py":30:0)
#loc2 = loc("01-vector-add.py":39:10)
#loc3 = loc("01-vector-add.py":44:18)
#loc5 = loc("01-vector-add.py":45:14)
#loc6 = loc("01-vector-add.py":47:11)
#loc8 = loc("01-vector-add.py":50:8)
#loc10 = loc("01-vector-add.py":51:8)
#loc11 = loc("01-vector-add.py":52:13)
#loc14 = loc("pid"(#loc2))
#loc15 = loc("block_start"(#loc3))
#loc16 = loc("offsets"(#loc5))
#loc17 = loc("mask"(#loc6))
#loc18 = loc("x"(#loc8))
#loc19 = loc("y"(#loc10))
#loc20 = loc("output"(#loc11))
#loc21 = loc(fused<#di_subprogram>[#loc])
