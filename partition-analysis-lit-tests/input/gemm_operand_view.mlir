#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @gemm_operand_view(%arg0: !ttg.memdesc<256x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, %arg1: !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>>, %arg3: f32, %arg4: i32) {
    %true = arith.constant true
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>, #ttng.tensor_memory, mutable>, !ttg.async.token)
    scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32  : i32 {
      %0 = tt.descriptor_load %arg1[%arg5, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>>> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %1 = ttg.local_alloc %0 : (tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>) -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>
      %2 = ttg.memdesc_trans %1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
      %3 = ttg.memdesc_subslice %2[0, 0] : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>
      %4 = ttng.tc_gen5_mma %arg0, %3, %result[%token], %false, %true : !ttg.memdesc<256x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory>, !ttg.memdesc<256x64xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>, #ttng.tensor_memory, mutable>
      %5 = ttg.local_load %2 : !ttg.memdesc<64x64xf16, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>, #ttg.shared_memory> -> tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>
      %result_0, %token_1 = ttng.tmem_load %result[%4] : !ttg.memdesc<256x64xf32, #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>
      "use"(%5, %result_0) : (tensor<64x64xf16, #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>>, tensor<256x64xf32, #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>>) -> ()
    } {tt.warp_specialize}
    tt.return
  }
}
