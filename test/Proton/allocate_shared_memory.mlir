// RUN: triton-opt --split-input-file -allocate-shared-memory -convert-proton-to-protongpu="max-shared-mem-size=4096" -allocate-proton-shared-memory %s | FileCheck %s

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: ttg.shared = 1664 : i32
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: allocate_aligned
  tt.func @allocate_aligned(%A : !tt.ptr<f16>) {
  %cst0 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record start "name0"
  %cst1 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  %cst2 = ttg.local_alloc : () -> !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst0 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst1 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  proton.record end "name0"
  ttg.local_dealloc %cst2 : !ttg.memdesc<16x16xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc  {allocation.offset = 1536 : i32}
  tt.return
  }
}

// -----

#A_SHARED = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: ttg.shared = 64 : i32
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: no_proton
  tt.func @no_proton(%A : !tt.ptr<f16>) {
  %cst0 = ttg.local_alloc : () -> !ttg.memdesc<1x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  ttg.local_dealloc %cst0 : !ttg.memdesc<1x32xf16, #A_SHARED, #ttg.shared_memory, mutable>
  // CHECK: ttg.local_alloc
  // CHECK-NOT: ttg.local_alloc
  tt.return
  }
}
