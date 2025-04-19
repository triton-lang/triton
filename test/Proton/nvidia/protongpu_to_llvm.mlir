// RUN: triton-opt %s -split-input-file -convert-proton-nvidia-gpu-to-llvm -cse --verify-diagnostics | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: no_conversion
  llvm.func @no_conversion() {
    // CHECK: nvvm.barrier0
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    gpu.barrier
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
  // CHECK-LABEL: convert_init
  llvm.func @convert_init() {
    // CHECK: %[[SIZE:.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK: %[[PTR:.*]] = llvm.alloca %[[SIZE]] x i32 : (i32) -> !llvm.ptr<5>
    // CHECK: %[[VAL:.*]] = llvm.mlir.constant(0 : i32) : i32
    // CHECK: llvm.store %[[VAL]], %[[PTR]] : i32, !llvm.ptr<5>
    %0 = proton_gpu.init_buffer_index : <i32, 5>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_smem_segment_setup
   tt.func @convert_smem_segment_setup() -> !proton_gpu.seg {
    // CHECK-DAG: nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
    // CHECK-DAG: %[[P3:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR3:.*]] = llvm.select %[[P3]], %{{.*}}, %[[ADDR2]]
    // CHECK-DAG: builtin.unrealized_conversion_cast %[[ADDR3]]
    %0 = ttg.local_alloc : () -> !ttg.memdesc<96xi32, #shared, #smem, mutable>
    %3 = proton_gpu.segment_base %0, {selectIds = array<i32: 0, 1, 2>} : !ttg.memdesc<96xi32, #shared, #smem, mutable> -> !proton_gpu.seg
    tt.return %3 : !proton_gpu.seg
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
 module attributes {"ttg.num-warps" = 8 : i32} {
   // CHECK-LABEL: convert_stack_segment_setup
   tt.func @convert_stack_segment_setup() -> !proton_gpu.seg {
     // CHECK-DAG: llvm.mlir.undef : !llvm.struct<(ptr, i32)>
     %0 = proton_gpu.stack_alloc : !ttg.memdesc<96xi32, #shared, #proton_gpu.stack_memory, mutable>
     %3 = proton_gpu.segment_base %0, {selectIds = array<i32: 0, 1, 2>} : !ttg.memdesc<96xi32, #shared, #proton_gpu.stack_memory, mutable> -> !proton_gpu.seg
     tt.return %3 : !proton_gpu.seg
   }
 }

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_circular_smem_store
  llvm.func @convert_circular_smem_store() {
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
    // CHECK-DAG: %[[SMEM_OFFSET:.*]] = llvm.add %[[ADDR2]], %[[INDEX]]
    // CHECK-DAG: %[[SMEM_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMEM_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK-DAG: %[[SMEM_P:.*]] = llvm.and
    // CHECK-DAG: llvm.inline_asm has_side_effects{{.*}}st.shared.v2.b32{{.*}}%[[SMEM_PTR]], %{{.*}}, %{{.*}}, %[[SMEM_P]]
    // CHECK-DAG: llvm.extractvalue {{.*}}[0] : !llvm.struct<(ptr<3>, i32)>
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %2 = proton_gpu.init_buffer_index : <i32, 5>
    %3 = proton_gpu.segment_base %0, {granularity = 1 : i32, selectIds = array<i32: 0, 1>} : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.seg
    scf.for %arg0 = %c0 to %c4 step %c1 {
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %8 = proton_gpu.read_counter : i32
        proton_gpu.circular_store start %0, %2, %8, %3 {scopeId = 1 : i32} : !ttg.memdesc<512xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
      }
    }
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: convert_circular_store_smem
  llvm.func @convert_circular_store_smem() {
    // CHECK-DAG: nvvm.read.ptx.sreg.tid.x
    // CHECK-DAG: %[[WARPID:.*]] = llvm.udiv
    // CHECK-DAG: %[[P1:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR1:.*]] = llvm.select %[[P1]]
    // CHECK-DAG: %[[P2:.*]] = llvm.icmp "eq" %[[WARPID]], %{{.*}}
    // CHECK-DAG: %[[ADDR2:.*]] = llvm.select %[[P2]], %{{.*}}, %[[ADDR1]]
    // CHECK-DAG: %[[CYCLE1:.*]] = llvm.inline_asm has_side_effects{{.*}}%clock
    // CHECK-DAG: %[[INDEX:.*]] = llvm.urem
    // CHECK-DAG: %[[SMEM_OFFSET:.*]] = llvm.add %[[ADDR2]], %[[INDEX]]
    // CHECK-DAG: %[[SMEM_PTR:.*]] = llvm.getelementptr %{{.*}}[%[[SMEM_OFFSET]]] : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i32
    // CHECK-DAG: %[[SMEM_P:.*]] = llvm.and
    // CHECK-DAG: llvm.inline_asm has_side_effects{{.*}}st.shared.v2.b32{{.*}}%[[SMEM_PTR]], %{{.*}}, %{{.*}}, %[[SMEM_P]]
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %2 = proton_gpu.init_buffer_index : <i32, 5>
    %3 = proton_gpu.segment_base %0, {granularity = 1 : i32, selectIds = array<i32: 0, 1>} : !ttg.memdesc<512xi32, #shared, #smem, mutable> -> !proton_gpu.seg
    %8 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %0, %2, %8, %3 {scopeId = 1 : i32} : !ttg.memdesc<512xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
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
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: convert_smem_finalize
  // CHECK-DAG: llvm.nvvm.read.ptx.sreg.smid
  // CHECK-DAG: llvm.extractvalue %{{.*}}[0] : !llvm.struct<(ptr<3>, i32)>
  // CHECK-DAG: llvm.store
  // CHECK-DAG: llvm.cond_br %{{.*}}, ^bb1, ^bb3
  // CHECK-DAG: ^bb1:
  // CHECK-DAG: %[[PREAMBLE:.*]] = llvm.mlir.constant(-559038737 : i32)
  // CHECK-DAG: llvm.store %[[PREAMBLE]], %{{.*}} : i32, !llvm.ptr<1>
  // CHECK-DAG: %[[PID:.*]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK-DAG: llvm.store
  // CHECK-DAG: %[[STEP:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-DAG: llvm.store
  // CHECK-DAG: llvm.br ^bb2
  // CHECK-DAG: ^bb2(%[[I:.*]]: i32):
  // CHECK-DAG: llvm.store
  // CHECK-DAG: llvm.store
  // CHECK-DAG: %[[UPPER:.*]] = llvm.mlir.constant(510 : i32) : i32
  // CHECK-DAG: %[[P2:.*]] = llvm.icmp "slt" %[[I]], %[[UPPER]] : i32
  // CHECK-DAG: %[[I_NEW:.*]] = llvm.add %[[I]], %[[STEP]] : i32
  // CHECK-DAG: llvm.cond_br %[[P2]], ^bb2(%[[I_NEW]] : i32), ^bb3
  // CHECK-DAG: ^bb3:
  // CHECK-DAG: llvm.return
  llvm.func @convert_smem_finalize(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<512xi32, #shared, #smem, mutable>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    %2 = proton_gpu.init_buffer_index : <i32, 5>
    proton_gpu.finalize %0, %2, %1 : !ttg.memdesc<512xi32, #shared, #smem, mutable>, <i32, 5>, <i32>
    llvm.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32} {
   tt.func @convert_stack_alloc_invalid(){
     // expected-error @+1 {{'proton_gpu.stack_alloc' op proton stack size must be positive and non-zero}}
     %1 = proton_gpu.stack_alloc : !ttg.memdesc<0xi32, #shared, #proton_gpu.stack_memory, mutable>
     tt.return
   }
 }

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 8 : i32} {
   tt.func @convert_stack_alloc_invalid2(){
     // expected-error @+1 {{'proton_gpu.stack_alloc' op proton stack buffer element type must be int 32}}
     %1 = proton_gpu.stack_alloc : !ttg.memdesc<96xi8, #shared, #proton_gpu.stack_memory, mutable>
     tt.return
   }
 }
