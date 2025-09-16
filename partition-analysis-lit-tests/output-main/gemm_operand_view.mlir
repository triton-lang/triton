#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @gemm_operand_view(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
    %true = arith.constant {ttg.partitions = [1 : i32]} true
    %false = arith.constant {ttg.partitions = [1 : i32]} false
    %c0_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 0 : i32
    %c64_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 64 : i32
    %result, %token = ttng.tmem_alloc {ttg.partitions = [0 : i32, 1 : i32]} : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32  : i32 {
      %0 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partitions = [0 : i32]} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked>
      %1 = ttg.local_alloc %0 {ttg.partitions = [0 : i32]} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %2 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>, ttg.partitions = [1 : i32]} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %3 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>, ttg.partitions = [0 : i32]} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %4 = ttg.memdesc_subslice %2[0, 0] {ttg.partitions = [1 : i32]} : !ttg.memdesc<64x64xf16, #shared1, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%token], %false, %true {ttg.partitions = [1 : i32]} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %6 = ttg.local_load %3 {ttg.partitions = [0 : i32]} : !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<64x64xf16, #blocked>
      %result_0, %token_1 = ttng.tmem_load %result[%5] {ttg.partitions = [0 : i32]} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked1>
      "use"(%6, %result_0) {ttg.partitions = [0 : i32]} : (tensor<64x64xf16, #blocked>, tensor<256x64xf32, #blocked1>) -> ()
    } {tt.warp_specialize, ttg.partitions = [0 : i32, 1 : i32]}
    tt.return
  }
}

