// RUN: triton-opt %s --tritonamdgpu-optimize-fma-dot=ksplit=8 -split-input-file | FileCheck --check-prefixes=K8,CHECK %s
// RUN: triton-opt %s --tritonamdgpu-optimize-fma-dot=ksplit=64 -split-input-file | FileCheck --check-prefixes=K64,CHECK %s

// CHECK-LABEL: dot_m1_n32
// K8:       %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<1x512xi8, #{{.*}}> -> tensor<1x8x64xi8, #{{.*}}>
// K64:      %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<1x512xi8, #{{.*}}> -> tensor<1x64x8xi8, #{{.*}}>
// CHECK:    %[[DOT_A_TMP2:.*]] = tt.trans %[[DOT_A_TMP1:.*]]
// CHECK:    %[[DOT_A:.*]] = triton_gpu.convert_layout %[[DOT_A_TMP2:.*]]
// CHECK:    %[[DOT_B_TMP1:.*]] = tt.load {{.*}} : tensor<512x32x!tt.ptr<i8>
// K8:       %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x32xi8, #{{.*}}> -> tensor<8x64x32xi8, #{{.*}}>
// K64:      %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x32xi8, #{{.*}}> -> tensor<64x8x32xi8, #{{.*}}>
// CHECK:    %[[DOT_B:.*]] = triton_gpu.convert_layout %[[DOT_B_TMP2]]
// K8:       %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<8x1x32xi32, #{{.*}}>
// K64:      %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<64x1x8xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<64x8x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<64x1x32xi32, #{{.*}}>
// CHECK:    %[[RED:.*]] = "tt.reduce"(%[[DOT_D]])
// CHECK:    %[[STORE_VAL:.*]] = triton_gpu.convert_layout %[[RED]]
// CHECK:    tt.store {{.*}}, %[[STORE_VAL]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @dot_m1_n32(%a: tensor<1x512xi8, #blocked>, %bPtr: tensor<512x32x!tt.ptr<i8>, #blocked>, %outPtr: tensor<1x32x!tt.ptr<i32>, #blocked>) {
    %aOp = triton_gpu.convert_layout %a : tensor<1x512xi8, #blocked> -> tensor<1x512xi8, #dot_operand_a>
    %b = tt.load %bPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x32x!tt.ptr<i8>, #blocked>
    %bOp = triton_gpu.convert_layout %b : tensor<512x32xi8, #blocked> -> tensor<512x32xi8, #dot_operand_b>
    %c = arith.constant dense<0> : tensor<1x32xi32, #blocked>
    %0 = tt.dot %aOp, %bOp, %c, inputPrecision = tf32 : tensor<1x512xi8, #dot_operand_a> * tensor<512x32xi8, #dot_operand_b> -> tensor<1x32xi32, #blocked>
    tt.store %outPtr, %0 : tensor<1x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dot_m1_n256
// K8:       %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<1x512xi8, #{{.*}}> -> tensor<1x8x64xi8, #{{.*}}>
// K64:      %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<1x512xi8, #{{.*}}> -> tensor<1x64x8xi8, #{{.*}}>
// CHECK:    %[[DOT_A_TMP2:.*]] = tt.trans %[[DOT_A_TMP1:.*]]
// CHECK:    %[[DOT_A:.*]] = triton_gpu.convert_layout %[[DOT_A_TMP2:.*]]
// CHECK:    %[[DOT_B_TMP1:.*]] = tt.load {{.*}} : tensor<512x256x!tt.ptr<i8>
// K8:       %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x256xi8, #{{.*}}> -> tensor<8x64x256xi8, #{{.*}}>
// K64:      %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x256xi8, #{{.*}}> -> tensor<64x8x256xi8, #{{.*}}>
// CHECK:    %[[DOT_B:.*]] = triton_gpu.convert_layout %[[DOT_B_TMP2]]
// K8:       %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<8x64x256xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<8x1x256xi32, #{{.*}}>
// K64:      %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<64x1x8xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<64x8x256xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<64x1x256xi32, #{{.*}}>
// CHECK:    %[[RED:.*]] = "tt.reduce"(%[[DOT_D]])
// CHECK:    %[[STORE_VAL:.*]] = triton_gpu.convert_layout %[[RED]]
// CHECK:    tt.store {{.*}}, %[[STORE_VAL]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @dot_m1_n256(%a: tensor<1x512xi8, #blocked>, %bPtr: tensor<512x256x!tt.ptr<i8>, #blocked>, %outPtr: tensor<1x256x!tt.ptr<i32>, #blocked>) {
    %aOp = triton_gpu.convert_layout %a : tensor<1x512xi8, #blocked> -> tensor<1x512xi8, #dot_operand_a>
    %b = tt.load %bPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x256x!tt.ptr<i8>, #blocked>
    %bOp = triton_gpu.convert_layout %b : tensor<512x256xi8, #blocked> -> tensor<512x256xi8, #dot_operand_b>
    %c = arith.constant dense<0> : tensor<1x256xi32, #blocked>
    %0 = tt.dot %aOp, %bOp, %c, inputPrecision = tf32 : tensor<1x512xi8, #dot_operand_a> * tensor<512x256xi8, #dot_operand_b> -> tensor<1x256xi32, #blocked>
    tt.store %outPtr, %0 : tensor<1x256x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: dot_m2_n32
// K8:       %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<2x512xi8, #{{.*}}> -> tensor<2x8x64xi8, #{{.*}}>
// K64:      %[[DOT_A_TMP1:.*]] = tt.reshape {{.*}} {allow_reorder = true} : tensor<2x512xi8, #{{.*}}> -> tensor<2x64x8xi8, #{{.*}}>
// CHECK:    %[[DOT_A_TMP2:.*]] = tt.trans %[[DOT_A_TMP1:.*]]
// CHECK:    %[[DOT_A:.*]] = triton_gpu.convert_layout %[[DOT_A_TMP2:.*]]
// CHECK:    %[[DOT_B_TMP1:.*]] = tt.load {{.*}} : tensor<512x32x!tt.ptr<i8>
// K8:       %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x32xi8, #{{.*}}> -> tensor<8x64x32xi8, #{{.*}}>
// K64:      %[[DOT_B_TMP2:.*]] = tt.reshape %[[DOT_B_TMP1]] {allow_reorder = true} : tensor<512x32xi8, #{{.*}}> -> tensor<64x8x32xi8, #{{.*}}>
// CHECK:    %[[DOT_B:.*]] = triton_gpu.convert_layout %[[DOT_B_TMP2]]
// K8:       %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<8x2x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<8x2x32xi32, #{{.*}}>
// K64:      %[[DOT_D:.*]] = tt.dot %[[DOT_A]], %[[DOT_B]], %{{.*}} : tensor<64x2x8xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #{{.*}}}>> * tensor<64x8x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #{{.*}}}>> -> tensor<64x2x32xi32, #{{.*}}>
// CHECK:    %[[RED:.*]] = "tt.reduce"(%[[DOT_D]])
// CHECK:    %[[STORE_VAL:.*]] = triton_gpu.convert_layout %[[RED]]
// CHECK:    tt.store {{.*}}, %[[STORE_VAL]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#blocked}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#blocked}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func @dot_m2_n32(%a: tensor<2x512xi8, #blocked>, %bPtr: tensor<512x32x!tt.ptr<i8>, #blocked>, %outPtr: tensor<2x32x!tt.ptr<i32>, #blocked>) {
    %aOp = triton_gpu.convert_layout %a : tensor<2x512xi8, #blocked> -> tensor<2x512xi8, #dot_operand_a>
    %b = tt.load %bPtr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<512x32x!tt.ptr<i8>, #blocked>
    %bOp = triton_gpu.convert_layout %b : tensor<512x32xi8, #blocked> -> tensor<512x32xi8, #dot_operand_b>
    %c = arith.constant dense<0> : tensor<2x32xi32, #blocked>
    %0 = tt.dot %aOp, %bOp, %c, inputPrecision = tf32 : tensor<2x512xi8, #dot_operand_a> * tensor<512x32xi8, #dot_operand_b> -> tensor<2x32xi32, #blocked>
    tt.store %outPtr, %0 : tensor<2x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
