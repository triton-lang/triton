// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //CHECK: tt.func @aref_get_put
  // CHECK-NEXT:   [[ZERO:%.*]] = arith.constant 0 : i32
  // CHECK-NEXT:   [[ONE:%.*]] = arith.constant 1 : i32
  // CHECK-NEXT:   [[EMPTY:%.*]] = ttg.local_alloc {aref_empty_mbarriers}
  // CHECK-NEXT:   [[FULL:%.*]] = ttg.local_alloc {aref_full_mbarriers}
  // CHECK-NEXT:   scf.for
  // CHECK-NEXT:     [[EMPTYSLICE:%.*]] = ttg.memdesc_subview [[EMPTY]]
  // CHECK-NEXT:     ttng.init_barrier [[EMPTYSLICE]], 0
  // CHECK-NEXT:     [[FULLSLICE:%.*]] = ttg.memdesc_subview [[FULL]]
  // CHECK-NEXT:     ttng.init_barrier [[FULLSLICE]], 1
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[EMPTYSLICE2:%.*]] = ttg.memdesc_subview [[EMPTY]]
  // CHECK-NEXT:   ttng.wait_barrier [[EMPTYSLICE2]], [[ONE]]
  // CHECK-NEXT:   [[A:%.*]] = ttg.memdesc_subview %arg0
  // CHECK-NEXT:   [[B:%.*]] = ttg.memdesc_subview %arg1
  // CHECK-NEXT:   "foo"([[A]], [[B]])
  // CHECK-NEXT:   [[FULLSLICE2:%.*]] = ttg.memdesc_subview [[FULL]]
  // CHECK-NEXT:   ttng.arrive_barrier [[FULLSLICE2]], 1
  // CHECK-NEXT:   [[FULLSLICE3:%.*]] = ttg.memdesc_subview [[FULL]]
  // CHECK-NEXT:   ttng.wait_barrier [[FULLSLICE3]], [[ZERO]]
  // CHECK-NEXT:   [[AA:%.*]] = ttg.memdesc_subview %arg0
  // CHECK-NEXT:   [[BB:%.*]] = ttg.memdesc_subview %arg1
  // CHECK-NEXT:   "bar"([[AA]], [[BB]])
  // CHECK-NEXT:   [[EMPTYSLICE3:%.*]] = ttg.memdesc_subview [[EMPTY]]
  // CHECK-NEXT:   ttng.arrive_barrier [[EMPTYSLICE3]],
  // CHECK-NEXT:   tt.return
  // CHECK-NEXT: }
  tt.func @aref_get_put(%d : !ttg.memdesc<1x64x16xf16, #shared0, #tmem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #tmem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1:2 = nvws.aref.put.enter %0[%c0_i32, %c1_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #tmem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
    "foo"(%1#0, %1#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
    nvws.aref.put.exit %0[%c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #tmem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %2:2 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #tmem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>
    "bar"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #tmem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
    nvws.aref.get.exit %0[%c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #tmem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    tt.return
  }
}
