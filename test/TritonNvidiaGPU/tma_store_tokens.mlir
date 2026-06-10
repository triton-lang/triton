// RUN: triton-opt %s | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32}>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#offsets = #ttg.slice<{dim = 0, parent = #blocked}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_store_tokens
  // CHECK: %{{.*}} = ttng.async_tma_copy_local_to_global
  // CHECK: %{{.*}} = ttng.async_tma_reduce
  // CHECK: %{{.*}} = ttng.async_tma_scatter
  // CHECK: ttng.async_tma_store_wait %{{.*}} {read_only}
  tt.func @tma_store_tokens(%desc: !tt.tensordesc<128x128xf32, #shared>,
                            %scatter_desc: !tt.tensordesc<1x128xf32, #shared>,
                            %alloc: !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory>,
                            %x: i32,
                            %offsets: tensor<128xi32, #offsets>) {
    %tok0 = ttng.async_tma_copy_local_to_global %desc[%x, %x] %alloc : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory>
    %tok1 = ttng.async_tma_reduce add, %desc[%x, %x] %alloc : !tt.tensordesc<128x128xf32, #shared>, !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory>
    %tok2 = ttng.async_tma_scatter %scatter_desc[%offsets, %x] %alloc : !tt.tensordesc<1x128xf32, #shared>, tensor<128xi32, #offsets>, i32, !ttg.memdesc<128x128xf32, #shared, #ttg.shared_memory>
    ttng.async_tma_store_wait %tok0 {read_only}
    ttng.async_tma_store_wait %tok1
    ttng.async_tma_store_wait %tok2
    tt.return
  }
}
