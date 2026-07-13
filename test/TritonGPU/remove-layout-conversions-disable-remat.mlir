// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=DEFAULT
// RUN: triton-opt %s -tritongpu-remove-layout-conversions='disable-remat=true' | FileCheck %s --check-prefix=DISABLED

// DEFAULT-LABEL: @nested_convert
// DEFAULT: %[[RESHAPE:.+]] = tt.reshape
// DEFAULT-NEXT: %[[CONVERT:.+]] = ttg.convert_layout %[[RESHAPE]]
// DEFAULT-NEXT: %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[CONVERT]]
// DEFAULT-NEXT: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}} %[[LHS]], %[[RHS]]
// DEFAULT-NEXT: tt.return %[[ASM]]

// DISABLED-LABEL: @nested_convert
// DISABLED: %[[RESHAPE:.+]] = tt.reshape
// DISABLED-NEXT: %[[INPUT_CONVERT:.+]] = ttg.convert_layout %[[RESHAPE]]
// DISABLED-NEXT: %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[INPUT_CONVERT]]
// DISABLED-NEXT: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}} %[[LHS]], %[[RHS]]
// DISABLED-NEXT: %[[OUTPUT_CONVERT:.+]] = ttg.convert_layout %[[ASM]]
// DISABLED-NEXT: tt.return %[[OUTPUT_CONVERT]]

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[0, 1]]}>
#src = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [0, 128]], block = [[0, 256]]}>
#packed = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], warp = [[0, 64, 0], [1, 0, 0], [2, 0, 0]], block = [[0, 128, 0]]}>
#target = #ttg.linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [[0, 64], [0, 0], [0, 0]], block = [[0, 128]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @nested_convert(%arg0: tensor<128x512xf32, #src>) -> tensor<128x256xi64, #target> {
    %0 = ttg.convert_layout %arg0 : tensor<128x512xf32, #src> -> tensor<128x512xf32, #blocked>
    %1 = tt.reshape %0 : tensor<128x512xf32, #blocked> -> tensor<128x256x2xf32, #packed>
    %outLHS, %outRHS = tt.split %1 : tensor<128x256x2xf32, #packed> -> tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>
    %2 = tt.elementwise_inline_asm "mov.b64 $0, { $1, $2 };" {constraints = "=l,r,r", packed_element = 1 : i32, pure = true} %outLHS, %outRHS : tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>, tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>>
    %3 = ttg.convert_layout %2 : tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #target>
    tt.return %3 : tensor<128x256xi64, #target>
  }
}
