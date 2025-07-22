// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="arch=gfx942" --verify-diagnostics | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt %s -split-input-file -convert-proton-amd-gpu-to-llvm="arch=gfx942" --convert-builtin-func-to-llvm --verify-diagnostics | FileCheck -allow-unused-prefixes --check-prefix=CONVERT-BUILTIN %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: no_conversion
  llvm.func @no_conversion() {
    //CHECK: rocdl.barrier
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    gpu.barrier
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_read_counter
  llvm.func @convert_read_counter() -> i32 {
    //CHECK: llvm.call_intrinsic "llvm.amdgcn.s.memtime"() : () -> i64
    //CHECK: llvm.trunc %{{.*}} : i64 to i32
    %1 = proton_gpu.read_counter : i32
    llvm.return %1 : i32
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_smem_segment_setup
   tt.func @convert_smem_segment_setup() -> !proton_gpu.segment<384, #smem, warp, [0, 1, 2]> {
    // CHECK-DAG: rocdl.workitem.id.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
    // CHECK-DAG: %[[P3:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR3:.*]] = llvm.select %[[P3]], %{{.*}}, %[[ADDR2]]
    %0 = ttg.local_alloc : () -> !ttg.memdesc<96xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_alloc %0 : !ttg.memdesc<96xi32, #shared, #smem, mutable> -> !proton_gpu.segment<384, #smem, warp, [0, 1, 2]>
    tt.return %3 : !proton_gpu.segment<384, #smem, warp, [0, 1, 2]>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_circular_store_smem
  llvm.func @convert_circular_store_smem() {
    // CHECK-DAG: rocdl.workitem.id.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
  	// CHECK-DAG: %[[CYCLE1:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memtime"()
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp, [0, 1]>
    %8 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %3, %8 {scopeId = 1 : i32} : !proton_gpu.segment<2048, #smem, warp, [0, 1]>, i32
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_global_scratch_alloc
  llvm.func @convert_global_scratch_alloc(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    // CHECK-DAG: rocdl.workgroup.id.x
    // CHECK-DAG: rocdl.workgroup.id.y
    // CHECK-DAG: rocdl.workgroup.id.z
    // CHECK-DAG: rocdl.grid.dim.x
    // CHECK-DAG: rocdl.grid.dim.y
    // CHECK-DAG: %[[PID:.*]] = llvm.trunc %15 : i64 to i32
    // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(384 : i32)
    // CHECK-DAG: %{{.*}} = llvm.mul %[[PID]], %[[SIZE]] : i32
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_initialize
  // CHECK: llvm.cond_br %{{.*}}, ^bb1, ^bb2
  // CHECK: ^bb1:

  // CHECK-DAG: %[[PREAMBLE:.*]] = llvm.mlir.constant(-559038737 : i32)
  // CHECK-DAG: %[[PREAMBLE_OFFSET:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[PREAMBLE_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[PREAMBLE_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
  // CHECK-DAG: llvm.store %[[PREAMBLE]], %{{.*}} : i32, !llvm.ptr<1>

  // CHECK-DAG: %[[PID:.*]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK-DAG: %[[PID_OFFSET:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[PID_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[PID_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %[[PID]], %[[PID_PTR]] : i32, !llvm.ptr<1>

  // CHECK-DAG: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID, 0, 3)", "=s"  : () -> i32
  // CHECK-DAG: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)", "=s"  : () -> i32
  // CHECK-DAG: llvm.inline_asm asm_dialect = att operand_attrs = [] "s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 13, 3)", "=s"  : () -> i32
  // CHECK-DAG: %[[SMID_OFFSET:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[SMID_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMID_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %{{.*}}, %[[SMID_PTR]] : i32, !llvm.ptr<1>

  // CHECK-DAG: %[[INIT_TIME_RAW:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
  // CHECK-DAG: %[[TEN:.*]] = llvm.mlir.constant(10 : i64) : i64
  // CHECK-DAG: %[[INIT_TIME:.*]] = llvm.mul %[[INIT_TIME_RAW]], %[[TEN]] : i64
  // CHECK-DAG: %[[INIT_TIME_OFFSET:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[INIT_TIME_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[INIT_TIME_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %[[INIT_TIME]], %[[INIT_TIME_PTR]] : i64, !llvm.ptr<1>

  // CHECK: ^bb2:
  // CHECK: llvm.return
  llvm.func @convert_smem_initialize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %0 : !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_finalize
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb1, ^bb9
  // CONVERT-BUILTIN: ^bb1:  // pred: ^bb0
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.br ^bb2(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb2(%{{.*}}: i32):  // 2 preds: ^bb1, ^bb8
  // CONVERT-BUILTIN: llvm.cond_br %2, ^bb3, ^bb4
  // CONVERT-BUILTIN: bb3:  // pred: ^bb2
  // CONVERT-BUILTIN: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CONVERT-BUILTIN: llvm.br ^bb5(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb4:  // pred: ^bb2
  // CONVERT-BUILTIN: llvm.br ^bb5(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb5(%{{.*}}: i32):  // 2 preds: ^bb3, ^bb4
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb6, ^bb7
  // CONVERT-BUILTIN: ^bb6:  // pred: ^bb5
  // CONVERT-BUILTIN: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CONVERT-BUILTIN: llvm.br ^bb8(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb7:  // pred: ^bb5
  // CONVERT-BUILTIN: llvm.br ^bb8(%{{.*}} : i32)
  // CONVERT-BUILTIN: ^bb8(%{{.*}}: i32):  // 2 preds: ^bb6, ^bb7
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb2(%{{.*}} : i32), ^bb9
  // CONVERT-BUILTIN: ^bb9:  // 2 preds: ^bb0, ^bb8
  // CONVERT-BUILTIN: llvm.cond_br %{{.*}}, ^bb10, ^bb11
  // CONVERT-BUILTIN: ^bb10:  // pred: ^bb9
  // CONVERT-BUILTIN: llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
  // CONVERT-BUILTIN: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
  // CONVERT-BUILTIN: llvm.br ^bb11
  // CONVERT-BUILTIN: ^bb11:  // 2 preds: ^bb9, ^bb10
  // CHECK: llvm.return
  llvm.func @convert_smem_finalize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    %2 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp>
    proton_gpu.finalize %2, %1 : !proton_gpu.segment<2048, #smem, warp>, !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: use_clock64
  llvm.func @use_clock64() {
    // CHECK-DAG: %[[CYCLE:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memtime"()
    // CHECK-DAG: %[[CYCLE64:.*]] = llvm.bitcast %[[CYCLE]] : i64 to vector<2xi32>
    // CHECK-DAG: llvm.extractelement %[[CYCLE64]]
    // CHECK-DAG: llvm.extractelement %[[CYCLE64]]
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp, [0, 1]>
    %8 = proton_gpu.read_counter : i64
    proton_gpu.circular_store start %3, %8 {scopeId = 1 : i32} : !proton_gpu.segment<2048, #smem, warp, [0, 1]>, i64
    llvm.return
  }
}
