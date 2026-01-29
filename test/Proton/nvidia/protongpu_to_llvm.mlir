// RUN: triton-opt %s -split-input-file -convert-proton-nvidia-gpu-to-llvm -cse --verify-diagnostics | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: no_conversion
  llvm.func @no_conversion() {
    // CHECK: ttg.barrier local|global_read|global_write
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    ttg.barrier local|global_read|global_write
    llvm.return
  }
}


// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_read_counter
  llvm.func @convert_read_counter() {
    // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock;", "=r"  : () -> i32
    %1 = proton_gpu.read_counter : i32
    llvm.return
  }
}


// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_smem_segment_setup
   tt.func @convert_smem_segment_setup() -> !proton_gpu.segment<384, #smem, warp, [0, 1, 2]> {
    // CHECK-DAG: nvvm.read.ptx.sreg.tid.x
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
  // CHECK-LABEL: convert_circular_smem_store_nested
  llvm.func @convert_circular_smem_store_nested() {
    // CHECK-DAG: nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
    // CHECK-DAG: scf.for
    // CHECK-DAG: scf.for
    // CHECK-DAG: %[[CYCLE1:.*]] = llvm.inline_asm has_side_effects{{.*}}%clock
    // CHECK-DAG: %[[INDEX:.*]] = llvm.urem
    // CHECK-DAG: %[[SMEM_OFFSET:.*]] = llvm.add {{.*}}, %[[INDEX]]
    // CHECK-DAG: %[[SMEM_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMEM_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK-DAG: llvm.inline_asm has_side_effects{{.*}}st.shared.v2.b32{{.*}}%[[SMEM_PTR]], %{{.*}}, %{{.*}}, %{{.*}}
    // CHECK-DAG: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<3>, i32)>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp, [0, 1]>
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %8 = proton_gpu.read_counter : i32
        proton_gpu.circular_store start %3, %8 {scopeId = 1 : i32} : !proton_gpu.segment<2048, #smem, warp, [0, 1]>, i32
      }
    }
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_circular_smem_store_flat
  llvm.func @convert_circular_smem_store_flat() {
    // CHECK-DAG: nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
    // CHECK-DAG: %[[CYCLE1:.*]] = llvm.inline_asm has_side_effects{{.*}}%clock
    // CHECK-DAG: %[[INDEX:.*]] = llvm.urem
    // CHECK-DAG: %[[SMEM_OFFSET:.*]] = llvm.add %{{.*}} %[[INDEX]]
    // CHECK-DAG: %[[SMEM_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMEM_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK-DAG: llvm.inline_asm has_side_effects{{.*}}st.shared.v2.b32{{.*}}%[[SMEM_PTR]], %{{.*}}, %{{.*}}, %{{.*}}
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
    // CHECK-DAG: nvvm.read.ptx.sreg.ctaid.x
    // CHECK-DAG: nvvm.read.ptx.sreg.ctaid.y
    // CHECK-DAG: nvvm.read.ptx.sreg.ctaid.z
    // CHECK-DAG: nvvm.read.ptx.sreg.nctaid.x
    // CHECK-DAG: nvvm.read.ptx.sreg.nctaid.y
    // CHECK-DAG: %[[PID:.*]] = llvm.trunc %15 : i64 to i32
    // CHECK-DAG: %[[SIZE:.*]] = llvm.mlir.constant(384 : i32)
    // CHECK-DAG: %{{.*}} = llvm.mul %[[PID]], %[[SIZE]] : i32
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, backend = "proton", nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_initialize
  // CHECK-DAG: llvm.cond_br %{{.*}}, ^bb1, ^bb2
  // CHECK-DAG: ^bb1:

  // CHECK-DAG: %[[PREAMBLE:.*]] = llvm.mlir.constant(-559038737 : i32)
  // CHECK-DAG: %[[PREAMBLE_OFFSET:.*]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK-DAG: %[[PREAMBLE_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[PREAMBLE_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
  // CHECK-DAG: llvm.store %[[PREAMBLE]], %[[PREAMBLE_PTR]] : i32, !llvm.ptr<1>

  // CHECK-DAG: %[[PID:.*]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK-DAG: %[[PID_OFFSET:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-DAG: %[[PID_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[PID_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %[[PID]], %[[PID_PTR]] : i32, !llvm.ptr<1>

  // CHECK-DAG: %[[SMID:.*]] = nvvm.read.ptx.sreg.smid
  // CHECK-DAG: %[[SMID_OFFSET:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: %[[SMID_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMID_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %[[SMID]], %[[SMID_PTR]] : i32, !llvm.ptr<1>

  // CHECK-DAG: %[[INIT_TIME:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-DAG: %[[INIT_TIME_OFFSET:.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK-DAG: %[[INIT_TIME_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[INIT_TIME_OFFSET]]] : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>
  // CHECK-DAG: llvm.store %[[INIT_TIME]], %[[INIT_TIME_PTR]] : i64, !llvm.ptr<1>

  // CHECK: ^bb2:
  // CHECK: llvm.return
  llvm.func @convert_smem_initialize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.global_scratch_alloc {alignment = 128 : i32, backend = "proton", nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %0 : !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_finalize
  // CHECK: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<3>, i32)>
  // CHECK: llvm.store
  // CHECK: llvm.cond_br %{{.*}}, ^bb1, ^bb2
  // CHECK: ^bb1: // pred: ^bb0
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CHECK: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
  // CHECK: llvm.br ^bb2
  // CHECK: ^bb2: // 2 preds: ^bb0, ^bb1
  // CHECK: llvm.cond_br %{{.*}}, ^bb3, ^bb4
  // CHECK: ^bb3: // pred: ^bb2
  // CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr<1>
  // CHECK: llvm.br ^bb4
  // CHECK: ^bb4: // 2 preds: ^bb2, ^bb3
  // CHECK: llvm.cond_br %{{.*}}, ^[[LOOP_HEAD:bb[0-9]+]](%{{.*}} : i32), ^[[EXIT:bb[0-9]+]]
  // CHECK: ^[[LOOP_HEAD]](%{{.*}}: i32):
  // CHECK: llvm.cond_br %{{.*}}, ^[[LOOP_BODY:bb[0-9]+]](%{{.*}} : i32), ^[[EXIT]]
  // CHECK: ^[[LOOP_BODY]](%{{.*}}: i32):
  // CHECK: llvm.getelementptr
  // CHECK: llvm.store
  // CHECK: llvm.store
  // CHECK: ^[[EXIT]]:
  // CHECK: llvm.cond_br %{{.*}}, ^[[POST:bb[0-9]+]], ^[[RET:bb[0-9]+]]
  // CHECK: ^[[POST]]:
  // CHECK: %{{.*}} = llvm.mlir.constant(8 : i32) : i32
  // CHECK: %[[POST_FINAL_TIME_PTR:.*]] = llvm.getelementptr %{{.*}}{{\[}}%{{.*}}{{\]}} : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i32
  // CHECK: %[[POST_FINAL_TIME:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK: llvm.store %[[POST_FINAL_TIME]], %[[POST_FINAL_TIME_PTR]] : i64, !llvm.ptr<1>
  // CHECK: llvm.br ^[[RET]]
  // CHECK: ^[[RET]]:
  // CHECK: llvm.return
  llvm.func @convert_smem_finalize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, backend = "proton", nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
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
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock;", "=r"  : () -> i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "mov.u32 $0, %clock_hi;", "=r"  : () -> i32
  // CHECK: llvm.inline_asm has_side_effects asm_dialect = att operand_attrs = [] "@$3 st.shared.v2.b32{{.*}}(!llvm.ptr<3>, i32, i32, i1)
  llvm.func @use_clock64() {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_alloc %0 : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.segment<2048, #smem, warp, [0, 1]>
    %8 = proton_gpu.read_counter : i64
    proton_gpu.circular_store start %3, %8 {scopeId = 1 : i32} : !proton_gpu.segment<2048, #smem, warp, [0, 1]>, i64
    llvm.return
  }
}
