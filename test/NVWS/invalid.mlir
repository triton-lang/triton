// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_get_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<2x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @below {{Leading dims of sliced aref inputs don't match}}
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<2x16x32xf16, #shared0, #smem>]>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_get_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<2x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @below {{Aref buffer is used elsewhere, Aref cannot guarantee async safety}}
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<2x16x32xf16, #shared0, #smem>]>
    %1 = ttng.tmem_alloc %d : (!ttg.memdesc<1x64x16xf16, #shared0, #smem>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_put_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %c0_i32 = arith.constant 0 : i32
    // expected-error @below {{Aref has different number of arguments than enter}}
    %1, %token = nvws.aref.put.enter %0[%c0_i32, %c0_i32] :
      !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
      -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @aref_put_batch(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    // expected-error @below {{Dimensions don't match}}
    %1:3 = nvws.aref.put.enter %0[%c0_i32, %c0_i32] :
      !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
      -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<32x32xf16, #shared0, #smem>, !ttg.async.token
    tt.return
  }
}
