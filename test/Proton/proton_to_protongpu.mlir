// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem=1024" -canonicalize -cse %s | FileCheck %s

module {
  // CHECK-LABEL: no_record
  tt.func @no_record() {
    // CHECK: tt.return
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK: %[[INDEX:.*]] = proton_gpu.init_buffer_index : <i32, 5>
  // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_base %[[BUF]]
  // CHECK: %[[START:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START]], %[[SEGMENT]] {scopeId = 0 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
  // CHECK: %[[END:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END]], %[[SEGMENT]] {scopeId = 0 : i32} : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, i32, !proton_gpu.seg
  // CHECK: gpu.barrier
  // CHECK: proton_gpu.finalize %[[BUF]], %[[INDEX]], %[[SCRATCH]] : !ttg.memdesc<256xi32, #shared, #smem, mutable>, <i32, 5>, <i32>
  // CHECK: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: scf_record
  tt.func @scf_record() {
    %i = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc
    // CHECK: %[[INDEX:.*]] = proton_gpu.init_buffer_index
    // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_base %[[BUF]]
    // CHECK: %[[START0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START0]], %[[SEGMENT]] {scopeId = 0 : i32}
    // CHECK: scf.for
    // CHECK: %[[START1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START1]], %[[SEGMENT]] {scopeId = 1 : i32}
    // CHECK: %[[END1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END1]], %[[SEGMENT]] {scopeId = 1 : i32}
    // CHECK: }
    // CHECK: %[[END0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END0]], %[[SEGMENT]] {scopeId = 0 : i32}
    // CHECK: gpu.barrier
    // CHECK: proton_gpu.finalize %[[BUF]], %[[INDEX]], %[[SCRATCH]]
    proton.record start "name1"
    scf.for %arg0 = %i to %c4 step %c1 {
      proton.record start "name0"
      proton.record end "name0"
    }
    proton.record end "name1"
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: nested_record
  tt.func @nested_record() {
    %i = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc
    // CHECK: %[[INDEX:.*]] = proton_gpu.init_buffer_index
    // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_base %[[BUF]]
    // CHECK: %[[START0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START0]], %[[SEGMENT]] {scopeId = 0 : i32}
    // CHECK: scf.for
    // CHECK: %[[START1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START1]], %[[SEGMENT]] {scopeId = 1 : i32}
    // CHECK: scf.for
    // CHECK: %[[END1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END1]], %[[SEGMENT]] {scopeId = 1 : i32}
    // CHECK: }
    // CHECK: %[[END0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END0]], %[[SEGMENT]] {scopeId = 0 : i32}
    // CHECK: }
    // CHECK: %[[START2:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[BUF]], %[[INDEX]], %[[START2]], %[[SEGMENT]] {scopeId = 2 : i32}
    // CHECK: %[[END2:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[BUF]], %[[INDEX]], %[[END2]], %[[SEGMENT]] {scopeId = 2 : i32}
    // CHECK: gpu.barrier
    // CHECK: proton_gpu.finalize %[[BUF]], %[[INDEX]], %[[SCRATCH]]
    proton.record start "name1"
    scf.for %arg0 = %i to %c4 step %c1 {
      proton.record start "name0"
      scf.for %arg1 = %i to %c4 step %c1 {
        proton.record end "name0"
      }
      proton.record end "name1"
    }
    proton.record start "name2"
    proton.record end "name2"
    tt.return
  }
}
