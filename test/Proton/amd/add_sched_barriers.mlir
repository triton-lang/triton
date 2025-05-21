// RUN: triton-opt %s -split-input-file  -add-sched-barriers --verify-diagnostics | FileCheck %s --check-prefix=CHECK
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: read_counter
  llvm.func @read_counter() {
    // CHECK: rocdl.sched.barrier 0
    %1 = proton_gpu.read_counter : i32
    llvm.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 384 : i32} {
  // CHECK-LABEL: nested_record
  llvm.func @nested_record(%arg: !llvm.ptr<1>) attributes {noinline = false, nvvm.kernel = 1 : ui1} {
    //CHECK: rocdl.sched.barrier 0
    //CHECK: proton_gpu.read_counter
    //CHECK: proton_gpu.circular_store
    //CHECK: rocdl.sched.barrier 0
    //CHECK: scf.for
    //CHECK:   rocdl.sched.barrier 0
    //CHECK:   proton_gpu.read_counter
    //CHECK:   proton_gpu.circular_store
    //CHECK:   rocdl.sched.barrier 0
    //CHECK:   scf.for
    //CHECK:     rocdl.sched.barrier 0
    //CHECK:     proton_gpu.read_counter
    //CHECK:     proton_gpu.circular_store
    //CHECK:     rocdl.sched.barrier 0
    //CHECK:   }
    //CHECK:   rocdl.sched.barrier 0
    //CHECK:   proton_gpu.read_counter
    //CHECK:   proton_gpu.circular_store
    //CHECK:   rocdl.sched.barrier 0
    //CHECK: }
    //CHECK: rocdl.sched.barrier 0
    //CHECK: proton_gpu.read_counter
    //CHECK: proton_gpu.circular_store
    //CHECK: rocdl.sched.barrier 0
    //CHECK: rocdl.sched.barrier 0
    //CHECK: proton_gpu.read_counter
    //CHECK: proton_gpu.circular_store
    //CHECK: rocdl.sched.barrier 0
    //CHECK: gpu.barrier
    //CHECK: proton_gpu.finalize
    //CHECK: llvm.return
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i32>
    %2 = proton_gpu.init_buffer_index : <i32, 5>
    %3 = proton_gpu.segment_base %0, {selectIds = array<i32>} : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> !proton_gpu.seg
    %4 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %0, %2, %4, %3 {scopeId = 0 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
    scf.for %arg0 = %c0 to %c4 step %c1 {
      %7 = proton_gpu.read_counter : i32
      proton_gpu.circular_store start %0, %2, %7, %3 {scopeId = 1 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
      scf.for %arg1 = %c0 to %c4 step %c1 {
        %9 = proton_gpu.read_counter : i32
        proton_gpu.circular_store end %0, %2, %9, %3 {scopeId = 1 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
      }
      %8 = proton_gpu.read_counter : i32
      proton_gpu.circular_store end %0, %2, %8, %3 {scopeId = 0 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
    }
    %5 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %0, %2, %5, %3 {scopeId = 2 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
    %6 = proton_gpu.read_counter : i32
    proton_gpu.circular_store end %0, %2, %6, %3 {scopeId = 2 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
    gpu.barrier
    proton_gpu.finalize %0, %2, %1 : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, <i32>
    llvm.return
  }
}
