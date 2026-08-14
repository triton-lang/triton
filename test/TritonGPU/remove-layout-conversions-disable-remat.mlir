// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=DEFAULT
// RUN: triton-opt %s -tritongpu-remove-layout-conversions='disable-remat-splitting=true' | FileCheck %s --check-prefix=NO-SPLIT

// DEFAULT-LABEL: @nested_convert_single_use
// DEFAULT: %[[RESHAPE:.+]] = tt.reshape
// DEFAULT-NEXT: %[[CONVERT:.+]] = ttg.convert_layout %[[RESHAPE]]
// DEFAULT-NEXT: %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[CONVERT]]
// DEFAULT-NEXT: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}} %[[LHS]], %[[RHS]]
// DEFAULT-NEXT: tt.return %[[ASM]]

// NO-SPLIT-LABEL: @nested_convert_single_use
// NO-SPLIT: %[[RESHAPE:.+]] = tt.reshape
// NO-SPLIT-NEXT: %[[INPUT_CONVERT:.+]] = ttg.convert_layout %[[RESHAPE]]
// NO-SPLIT-NEXT: %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[INPUT_CONVERT]]
// NO-SPLIT-NEXT: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}} %[[LHS]], %[[RHS]]
// NO-SPLIT-NEXT: tt.return %[[ASM]]

// DEFAULT-LABEL: @nested_convert_multi_use
// DEFAULT: %[[RESHAPE:.+]] = tt.reshape
// DEFAULT-NEXT: %[[TARGET_INPUT:.+]] = ttg.convert_layout %[[RESHAPE]]
// DEFAULT-NEXT: %[[ORIGINAL_INPUT:.+]] = ttg.convert_layout %[[RESHAPE]]
// DEFAULT: %[[TARGET:.+]] = tt.elementwise_inline_asm
// DEFAULT-NEXT: %[[ORIGINAL:.+]] = tt.elementwise_inline_asm
// DEFAULT-NEXT: tt.return %[[TARGET]], %[[TARGET]], %[[ORIGINAL]]

// NO-SPLIT-LABEL: @nested_convert_multi_use
// NO-SPLIT: %[[RESHAPE:.+]] = tt.reshape
// NO-SPLIT-NEXT: %[[INPUT_CONVERT:.+]] = ttg.convert_layout %[[RESHAPE]]
// NO-SPLIT-NEXT: %[[LHS:.+]], %[[RHS:.+]] = tt.split %[[INPUT_CONVERT]]
// NO-SPLIT-NEXT: %[[ASM:.+]] = tt.elementwise_inline_asm {{.*}} %[[LHS]], %[[RHS]]
// NO-SPLIT-NEXT: %[[OUTPUT_CONVERT:.+]] = ttg.convert_layout %[[ASM]]
// NO-SPLIT-NEXT: tt.return %[[OUTPUT_CONVERT]], %[[OUTPUT_CONVERT]], %[[ASM]]

// NO-SPLIT-LABEL: @hoist_broadcast
// NO-SPLIT: %[[CONVERT:.+]] = ttg.convert_layout %arg0
// NO-SPLIT-NEXT: %[[BROADCAST:.+]] = tt.broadcast %[[CONVERT]]
// NO-SPLIT-NEXT: tt.return %[[BROADCAST]]

// NO-SPLIT-LABEL: @hoist_ext
// NO-SPLIT: %[[CONVERT:.+]] = ttg.convert_layout %arg0
// NO-SPLIT-NEXT: %[[EXT:.+]] = arith.extui %[[CONVERT]]
// NO-SPLIT-NEXT: tt.return %[[EXT]]

// NO-SPLIT-LABEL: @reject_hoist_broadcast_multi_use
// NO-SPLIT: %[[BROADCAST:.+]] = tt.broadcast %arg0
// NO-SPLIT-NEXT: %[[CONVERT:.+]] = ttg.convert_layout %[[BROADCAST]]
// NO-SPLIT-NEXT: tt.return %[[CONVERT]], %[[BROADCAST]]

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0], CGALayout = [[0, 1]]}>
#src = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64], [0, 128]], block = [[0, 256]]}>
#packed = #ttg.linear<{register = [[0, 0, 1], [0, 1, 0], [4, 0, 0], [8, 0, 0], [16, 0, 0], [32, 0, 0], [64, 0, 0]], lane = [[0, 2, 0], [0, 4, 0], [0, 8, 0], [0, 16, 0], [0, 32, 0]], warp = [[0, 64, 0], [1, 0, 0], [2, 0, 0]], block = [[0, 128, 0]]}>
#target = #ttg.linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [[0, 64], [0, 0], [0, 0]], block = [[0, 128]]}>
#hoist_src = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1], CGALayout = [[0, 1]]}>
#hoist_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [8, 1], order = [0, 1], CGALayout = [[0, 1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @nested_convert_single_use(%arg0: tensor<128x512xf32, #src>) -> tensor<128x256xi64, #target> {
    %0 = ttg.convert_layout %arg0 : tensor<128x512xf32, #src> -> tensor<128x512xf32, #blocked>
    %1 = tt.reshape %0 : tensor<128x512xf32, #blocked> -> tensor<128x256x2xf32, #packed>
    %outLHS, %outRHS = tt.split %1 : tensor<128x256x2xf32, #packed> -> tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>
    %2 = tt.elementwise_inline_asm "mov.b64 $0, { $1, $2 };" {constraints = "=l,r,r", packed_element = 1 : i32, pure = true} %outLHS, %outRHS : tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>, tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>>
    %3 = ttg.convert_layout %2 : tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #target>
    tt.return %3 : tensor<128x256xi64, #target>
  }

  tt.func @nested_convert_multi_use(%arg0: tensor<128x512xf32, #src>) -> (tensor<128x256xi64, #target>, tensor<128x256xi64, #target>, tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>>) {
    %0 = ttg.convert_layout %arg0 : tensor<128x512xf32, #src> -> tensor<128x512xf32, #blocked>
    %1 = tt.reshape %0 : tensor<128x512xf32, #blocked> -> tensor<128x256x2xf32, #packed>
    %outLHS, %outRHS = tt.split %1 : tensor<128x256x2xf32, #packed> -> tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>
    %2 = tt.elementwise_inline_asm "mov.b64 $0, { $1, $2 };" {constraints = "=l,r,r", packed_element = 1 : i32, pure = true} %outLHS, %outRHS : tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>>, tensor<128x256xf32, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>>
    %3 = ttg.convert_layout %2 : tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #target>
    %4 = ttg.convert_layout %2 : tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>> -> tensor<128x256xi64, #target>
    tt.return %3, %4, %2 : tensor<128x256xi64, #target>, tensor<128x256xi64, #target>, tensor<128x256xi64, #ttg.slice<{dim = 2, parent = #packed}>>
  }

  tt.func @hoist_broadcast(%arg0: tensor<64x1xi32, #hoist_src>) -> tensor<64x2xi32, #hoist_dst> {
    %0 = tt.broadcast %arg0 : tensor<64x1xi32, #hoist_src> -> tensor<64x2xi32, #hoist_src>
    %1 = ttg.convert_layout %0 : tensor<64x2xi32, #hoist_src> -> tensor<64x2xi32, #hoist_dst>
    tt.return %1 : tensor<64x2xi32, #hoist_dst>
  }

  tt.func @hoist_ext(%arg0: tensor<64x2xi1, #hoist_src>) -> tensor<64x2xi32, #hoist_dst> {
    %0 = arith.extui %arg0 : tensor<64x2xi1, #hoist_src> to tensor<64x2xi32, #hoist_src>
    %1 = ttg.convert_layout %0 : tensor<64x2xi32, #hoist_src> -> tensor<64x2xi32, #hoist_dst>
    tt.return %1 : tensor<64x2xi32, #hoist_dst>
  }

  tt.func @reject_hoist_broadcast_multi_use(%arg0: tensor<64x1xi32, #hoist_src>) -> (tensor<64x2xi32, #hoist_dst>, tensor<64x2xi32, #hoist_src>) {
    %0 = tt.broadcast %arg0 : tensor<64x1xi32, #hoist_src> -> tensor<64x2xi32, #hoist_src>
    %1 = ttg.convert_layout %0 : tensor<64x2xi32, #hoist_src> -> tensor<64x2xi32, #hoist_dst>
    tt.return %1, %0 : tensor<64x2xi32, #hoist_dst>, tensor<64x2xi32, #hoist_src>
  }
}
