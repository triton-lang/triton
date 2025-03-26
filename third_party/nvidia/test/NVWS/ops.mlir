// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_single
  // CHECK: nvws.aref.create
  tt.func @aref_create_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get_single
  // CHECK: nvws.aref.get
  tt.func @aref_get_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1 = nvws.aref.get %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>) -> !ttg.memdesc<1x64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_single
  // CHECK: nvws.aref.put
  tt.func @aref_put_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    nvws.aref.put %0 as (%b0 : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>) -> ()
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_batch
  // CHECK: nvws.aref.create
  tt.func @aref_create_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get_batch
  // CHECK: nvws.aref.get
  tt.func @aref_get_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    %1 = nvws.aref.get %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      nvws.aref.return %b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> !ttg.memdesc<64x16xf16, #shared0, #smem>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_batch
  // CHECK: nvws.aref.put
  tt.func @aref_put_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : !ttg.memdesc<64x16xf16, #shared0, #smem>, %b1 : !ttg.memdesc<16x32xf16, #shared0, #smem>) {
      nvws.aref.return
    } : (!nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>], 1>, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put_tensor
  // CHECK: nvws.aref.put
  tt.func @aref_put_tensor(%d : tensor<1x64x16xf16>, %e : tensor<1x16x32xf16>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>
    nvws.aref.put %0[%c0_i32] as (%b0 : tensor<64x16xf16>, %b1 : tensor<16x32xf16>) {
      %1 = math.exp %b0 : tensor<64x16xf16>
      %2 = math.cos %b1 : tensor<16x32xf16>
      nvws.aref.return %1, %2 : tensor<64x16xf16>, tensor<16x32xf16>
    } : (!nvws.aref<[tensor<1x64x16xf16>, tensor<1x16x32xf16>], 1>, i32) -> ()
    tt.return
  }
}
