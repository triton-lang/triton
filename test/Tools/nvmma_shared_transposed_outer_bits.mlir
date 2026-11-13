// RUN: triton-tensor-layout -l "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>" -t "tensor<512x128xf16>" -use-hw-view | FileCheck %s --check-prefix=PLAIN
// RUN: triton-tensor-layout -l "#ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>" -t "tensor<128x512xf16>" -use-hw-view | FileCheck %s --check-prefix=TRANSPOSED

// ttg.memdesc_trans only reinterprets the memory, so the transposed layout has
// to be the exact transpose of the plain one, including the offset bits above
// the 256x64 tile.

// PLAIN:      Offset: 16384 -> (256,  0)
// PLAIN:      Offset: 32768 -> (  0, 64)

// TRANSPOSED: Offset: 16384 -> (  0,256)
// TRANSPOSED: Offset: 32768 -> ( 64,  0)
