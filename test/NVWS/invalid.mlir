// RUN: triton-opt --split-input-file %s --nvws-insert-tmem-aref --verify-diagnostics

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

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.target" = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tmem_subview() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %true = arith.constant true
    %zero = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
    %parent = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %view = ttng.tmem_subslice %parent {offset = 128 : i32, dim = 0 : i32} : !ttg.memdesc<256x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    // expected-error @+1 {{TMEM subviews NYI in the pipeliner}}
    ttng.tmem_store %zero, %view, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 256x128>
    scf.for %i = %c0 to %c1 step %c1 {
      scf.yield
    } {tt.warp_specialize}
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
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
