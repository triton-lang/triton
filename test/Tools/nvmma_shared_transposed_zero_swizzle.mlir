// RUN: triton-tensor-layout -l "#ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 8}>" -t "tensor<16x128xi8>" -use-hw-view | FileCheck %s

// CHECK:      Offset: 0 -> ( 0,  0)
// CHECK-NEXT: Offset: 1 -> ( 1,  0)
// CHECK:      Offset: 16 -> ( 0,  1)
