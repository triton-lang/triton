// RUN: triton-opt %s -split-input-file --convert-proton-nvidia-gpu-to-llvm -cse | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: no_conversion
  tt.func @no_conversion() {
    // CHECK: ttg.local_alloc
    // CHECK: gpu.barrier
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
