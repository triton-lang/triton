// RUN: triton-opt %s -split-input-file --convert-proton-nvidia-gpu-to-llvm -cse | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: no_conversion
  tt.func @no_conversion() {
    // CHECK: ttg.local_alloc
    // CHECK: nvvm.barrier0
    // CHECK: tt.return
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    gpu.barrier
    tt.return
  }
}


// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_read_counter
  tt.func @convert_read_counter() {
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock;", "=r"  : () -> i32
    %1 = proton_gpu.read_counter : i32
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_init
  tt.func @convert_init() {
    // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[PTR:.*]] = llvm.alloca %[[SIZE]] x i32 : (i32) -> !llvm.ptr<5>
    // CHECK: %[[VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.store %[[VAL]], %[[PTR]] : i32, !llvm.ptr<5>
    // CHECK: tt.return
    %0 = proton_gpu.init_buffer_index : <i32, 5>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_segment_setup
  tt.func @convert_segment_setup() -> i1 {
    // CHECK: %1 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: %2 = llvm.sext %1 : i32 to i64
    // CHECK: %3 = llvm.trunc %2 : i64 to i32
    // CHECK: %4 = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %5 = llvm.udiv %3, %4 : i32
    // CHECK: %6 = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %7 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: %8 = llvm.icmp "eq" %5, %7 : i32
    // CHECK: %9 = llvm.select %8, %7, %6 : i1, i32
    // CHECK: %10 = llvm.mlir.constant(128 : i32) : i32
    // CHECK: %11 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %12 = llvm.icmp "eq" %5, %11 : i32
    // CHECK: %13 = llvm.select %12, %10, %9 : i1, i32
    // CHECK: %14 = llvm.mlir.constant(256 : i32) : i32
    // CHECK: %15 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %16 = llvm.icmp "eq" %5, %15 : i32
    // CHECK: %17 = llvm.select %16, %14, %13 : i1, i32
    // CHECK: %18 = llvm.icmp "eq" %17, %6 : i32
    // CHECK: %19 = llvm.urem %3, %4 : i32
    // CHECK: %20 = llvm.icmp "eq" %19, %7 : i32
    // CHECK: %21 = llvm.and %18, %20 : i1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<96xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_base %0, {granularity = 1 : i32, warpIds = array<i32: 0, 1, 2>} : !ttg.memdesc<96xi32, #shared, #smem, mutable> -> i32
    %4 = proton_gpu.check_segment_writer %3 : i1
    tt.return %4 : i1
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_circular_store
  tt.func @convert_circular_store() {
    // CHECK: %7 = builtin.unrealized_conversion_cast %6 : !ttg.memdesc<512xi32, #shared, #smem, mutable> to !llvm.struct<(ptr<3>, i32)>

    // setup the index

    // CHECK: %8 = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %9 = llvm.alloca %8 x i32 : (i32) -> !llvm.ptr<5>
    // CHECK: %10 = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.store %10, %9 : i32, !llvm.ptr<5>

    // compute the warp id

    // CHECK: %11 = nvvm.read.ptx.sreg.tid.x : i32
    // CHECK: %12 = llvm.sext %11 : i32 to i64
    // CHECK: %13 = llvm.trunc %12 : i64 to i32
    // CHECK: %14 = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %15 = llvm.udiv %13, %14 : i32

    // compute the segment base

    // CHECK: %16 = llvm.mlir.constant(-1 : i32) : i32
    // CHECK: %17 = llvm.icmp "eq" %15, %10 : i32
    // CHECK: %18 = llvm.select %17, %10, %16 : i1, i32
    // CHECK: %19 = llvm.mlir.constant(512 : i32) : i32
    // CHECK: %20 = llvm.icmp "eq" %15, %8 : i32
    // CHECK: %21 = llvm.select %20, %19, %18 : i1, i32
    // CHECK: %22 = llvm.mlir.constant(1024 : i32) : i32
    // CHECK: %23 = llvm.mlir.constant(2 : i32) : i32
    // CHECK: %24 = llvm.icmp "eq" %15, %23 : i32
    // CHECK: %25 = llvm.select %24, %22, %21 : i1, i32
    // CHECK: %26 = llvm.mlir.constant(1536 : i32) : i32
    // CHECK: %27 = llvm.mlir.constant(3 : i32) : i32
    // CHECK: %28 = llvm.icmp "eq" %15, %27 : i32
    // CHECK: %29 = llvm.select %28, %26, %25 : i1, i32

    // Check if the thread is writer

    // CHECK: %30 = llvm.icmp "eq" %29, %16 : i32
    // CHECK: %31 = llvm.urem %13, %14 : i32
    // CHECK: %32 = llvm.icmp "eq" %31, %10 : i32
    // CHECK: %33 = llvm.and %30, %32 : i1

    // for simplicity, we use the scf dialect.
    // In practice, it would have been lowered to cf dialect

    // CHECK: scf.for %arg0 = %5 to %1 step %3 {
    // CHECK:  scf.for %arg1 = %5 to %1 step %3 {
    // CHECK:    %34 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock;", "=r"  : () -> i32
    // CHECK:    %35 = llvm.extractvalue %7[0] : !llvm.struct<(ptr<3>, i32)>
    // CHECK:    %36 = llvm.load %9 : !llvm.ptr<5> -> i32
    // CHECK:    %37 = llvm.add %36, %23 : i32
    // CHECK:    llvm.store %37, %9 : i32, !llvm.ptr<5>
    // CHECK:    %38 = llvm.mlir.constant(128 : i32) : i32
    // CHECK:    %39 = llvm.urem %36, %38 : i32
    // CHECK:    %40 = llvm.add %29, %39 : i32
    // CHECK:    %41 = llvm.getelementptr %35[%40] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK:    %42 = llvm.mlir.undef : vector<2xi32>
    // CHECK:    %43 = llvm.insertelement %8, %42[%10 : i32] : vector<2xi32>
    // CHECK:    %44 = llvm.insertelement %34, %43[%8 : i32] : vector<2xi32>
    // CHECK:    %45 = llvm.extractelement %44[%10 : i32] : vector<2xi32>
    // CHECK:    %46 = llvm.extractelement %44[%8 : i32] : vector<2xi32>
    // CHECK:    %47 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$3 st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b" %41, %45, %46, %33 : (!llvm.ptr<3>, i32, i32, i1) -> !llvm.void
    // CHECK:    %48 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock;", "=r"  : () -> i32
    // CHECK:    %49 = llvm.load %9 : !llvm.ptr<5> -> i32
    // CHECK:    %50 = llvm.add %49, %23 : i32
    // CHECK:    llvm.store %50, %9 : i32, !llvm.ptr<5>
    // CHECK:    %51 = llvm.urem %49, %38 : i32
    // CHECK:    %52 = llvm.add %29, %51 : i32
    // CHECK:    %53 = llvm.getelementptr %35[%52] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK:    %54 = llvm.mlir.constant(-2147483647 : i32) : i32
    // CHECK:    %55 = llvm.insertelement %54, %42[%10 : i32] : vector<2xi32>
    // CHECK:    %56 = llvm.insertelement %34, %55[%8 : i32] : vector<2xi32>
    // CHECK:    %57 = llvm.extractelement %56[%10 : i32] : vector<2xi32>
    // CHECK:    %58 = llvm.extractelement %56[%8 : i32] : vector<2xi32>
    // CHECK:    %59 = llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$3 st.shared.v2.b32 [ $0 + 0 ], { $1, $2 };", "r,r,r,b" %53, %57, %58, %33 : (!llvm.ptr<3>, i32, i32, i1) -> !llvm.void
    // CHECK:  }
    // CHECK:}
    // CHECK:tt.return
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %2 = proton_gpu.init_buffer_index : <i32, 5>
    %3 = proton_gpu.segment_base %0, {granularity = 1 : i32, warpIds = array<i32: 0, 1, 2, 3>} : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> i32
    %4 = proton_gpu.check_segment_writer %3 : i1
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %8 = proton_gpu.read_counter : i32
        proton_gpu.circular_store start %0, %2, %8, %3, %4 {scopeId = 1 : i32} : !ttg.memdesc<512xi32, #shared, #smem, mutable>, <i32, 5>, i32, i32, i1
        %9 = proton_gpu.read_counter : i32
        proton_gpu.circular_store end %0, %2, %8, %3, %4 {scopeId = 1 : i32} : !ttg.memdesc<512xi32, #shared, #smem, mutable>, <i32, 5>, i32, i32, i1
      }
    }
    tt.return
  }
}
