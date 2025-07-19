// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem-size=32768" -canonicalize -cse %s | FileCheck %s

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
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK: %[[START:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK: %[[END:.*]] = proton_gpu.read_counter : i32
  // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK: gpu.barrier
  // CHECK: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
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
    // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
    // CHECK: %[[START0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START0]] {scopeId = 0 : i32}
    // CHECK: scf.for
    // CHECK: %[[START1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START1]] {scopeId = 1 : i32}
    // CHECK: %[[END1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END1]] {scopeId = 1 : i32}
    // CHECK: }
    // CHECK: %[[END0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END0]] {scopeId = 0 : i32}
    // CHECK: gpu.barrier
    // CHECK: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]]
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
    // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
    // CHECK: %[[START0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START0]] {scopeId = 0 : i32}
    // CHECK: scf.for
    // CHECK: %[[START1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START1]] {scopeId = 1 : i32}
    // CHECK: scf.for
    // CHECK: %[[END1:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END1]] {scopeId = 1 : i32}
    // CHECK: }
    // CHECK: %[[END0:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END0]] {scopeId = 0 : i32}
    // CHECK: }
    // CHECK: %[[START2:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store start %[[SEGMENT]], %[[START2]] {scopeId = 2 : i32}
    // CHECK: %[[END2:.*]] = proton_gpu.read_counter : i32
    // CHECK: proton_gpu.circular_store end %[[SEGMENT]], %[[END2]] {scopeId = 2 : i32}
    // CHECK: gpu.barrier
    // CHECK: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]]
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

// -----

// CHECK: #shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
// CHECK: #smem = #ttg.shared_memory
// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {
// CHECK:   tt.func @convert_warp_specialize() {
// CHECK:     %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
// CHECK:     proton_gpu.init_ctx %[[SCRATCH]] : !tt.ptr<i32>
// CHECK:     %[[MEMDESC:.*]] = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
// CHECK:     %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[MEMDESC]] : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>
// CHECK:     %[[COUNTER1:.*]] = proton_gpu.read_counter : i32
// CHECK:     proton_gpu.circular_store start %[[SEGMENT]], %[[COUNTER1]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:     ttg.warp_specialize(%[[MEMDESC]], %[[SCRATCH]])
// CHECK:     default {
// CHECK:       %[[COUNTER2:.*]] = proton_gpu.read_counter : i32
// CHECK:       proton_gpu.circular_store start %[[SEGMENT]], %[[COUNTER2]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:       %[[COUNTER3:.*]] = proton_gpu.read_counter : i32
// CHECK:       proton_gpu.circular_store end %[[SEGMENT]], %[[COUNTER3]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:       ttg.warp_yield
// CHECK:     }
// CHECK:     partition0(%[[ARG0:.*]]: !ttg.memdesc<256xi32, #shared, #smem, mutable>, %[[ARG1:.*]]: !tt.ptr<i32>) num_warps(1) {
// CHECK:       %[[SEGMENT2:.*]] = proton_gpu.segment_alloc %[[ARG0]] : !ttg.memdesc<256xi32, #shared, #smem, mutable> -> <1024, #smem, warp>
// CHECK:       proton_gpu.restore_ctx %[[SEGMENT2]], %[[ARG1]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
// CHECK:       %[[COUNTER4:.*]] = proton_gpu.read_counter : i32
// CHECK:       proton_gpu.circular_store start %[[SEGMENT2]], %[[COUNTER4]] {scopeId = 2 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:       %[[COUNTER5:.*]] = proton_gpu.read_counter : i32
// CHECK:       proton_gpu.circular_store end %[[SEGMENT2]], %[[COUNTER5]] {scopeId = 2 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:       proton_gpu.save_ctx %[[SEGMENT2]], %[[ARG1]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
// CHECK:       ttg.warp_return
// CHECK:     } : (!ttg.memdesc<256xi32, #shared, #smem, mutable>, !tt.ptr<i32>) -> ()
// CHECK:     %[[COUNTER6:.*]] = proton_gpu.read_counter : i32
// CHECK:     proton_gpu.circular_store end %[[SEGMENT]], %[[COUNTER6]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
// CHECK:     gpu.barrier
// CHECK:     proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
// CHECK:     tt.return
// CHECK:   }
// CHECK: }
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32} {
  tt.func @convert_warp_specialize() {
    proton.record start "kernel"
    ttg.warp_specialize()
    default {
      proton.record start "default"
      proton.record end "default"
      ttg.warp_yield
    }
    partition0() num_warps(1) {
      proton.record start "partition0"
      proton.record end "partition0"
      ttg.warp_return
    } : () -> ()
    proton.record end "kernel"
    tt.return
  }
}
