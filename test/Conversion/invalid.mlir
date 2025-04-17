// RUN: not triton-opt %s --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @padded_fp4_tma_misaligned_load(%ptr : !tt.ptr<i8>, %bar : !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable>) {
  %dst  = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1x128xi8, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true}>, #ttg.shared_memory, mutable>
  %c0   = arith.constant 0   : i32
  %c64  = arith.constant 64  : i32
  %pred = arith.constant true
  // CHECK: Illegal fp4 padded tensor coordinate; contiguous coordinate (64) is not a multiple of 128
  ttng.async_tma_copy_global_to_local %ptr[%c0, %c64] %dst, %bar, %pred : !tt.ptr<i8>, !ttg.memdesc<1xi64, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>, #ttg.shared_memory, mutable> -> !ttg.memdesc<1x128xi8, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true}>, #ttg.shared_memory, mutable>
  tt.return
}

}
