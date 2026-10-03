// RUN: triton-opt %s -triton-narrow-redundant-loads | FileCheck %s

// A group-quantized GEMM scale: the scale index divides the K coordinate by the
// group size, so a K tile inside one group reads the same scale for all of its K
// positions. Dim 1 still picks a different scale per lane.
// CHECK-LABEL: @narrow_group_quant_scale
tt.func @narrow_group_quant_scale(%base: !tt.ptr<f16>, %kBlock: i32) -> tensor<32x128xf16> {
  %kPerBlock = arith.constant 32 : i32
  %groupSize = arith.constant dense<128> : tensor<32x1xi32>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %kBase = arith.muli %kBlock, %kPerBlock : i32
  %kBaseT = tt.splat %kBase : i32 -> tensor<32x1xi32>
  %k = arith.addi %kBaseT, %kCol : tensor<32x1xi32>
  %group = arith.divui %k, %groupSize : tensor<32x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %groupB = tt.broadcast %group : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %off = arith.addi %nB, %groupB : tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// The same kernel with a K tile four times the group size. The tile spans four
// groups, so the scale is not constant along the whole dimension.
// CHECK-LABEL: @no_narrow_k_tile_straddles_groups
tt.func @no_narrow_k_tile_straddles_groups(%base: !tt.ptr<f16>, %kBlock: i32) -> tensor<512x128xf16> {
  %kPerBlock = arith.constant 512 : i32
  %groupSize = arith.constant dense<128> : tensor<512x1xi32>
  %kRange = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
  %kBase = arith.muli %kBlock, %kPerBlock : i32
  %kBaseT = tt.splat %kBase : i32 -> tensor<512x1xi32>
  %k = arith.addi %kBaseT, %kCol : tensor<512x1xi32>
  %group = arith.divui %k, %groupSize : tensor<512x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %groupB = tt.broadcast %group : tensor<512x1xi32> -> tensor<512x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<512x128xi32>
  %off = arith.addi %nB, %groupB : tensor<512x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<512x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<512x128x!tt.ptr<f16>>, tensor<512x128xi32>
  // CHECK: tt.load %{{.*}} : tensor<512x128x!tt.ptr<f16>>
  %val = tt.load %addr : tensor<512x128x!tt.ptr<f16>>
  tt.return %val : tensor<512x128xf16>
}

// Which dimension is redundant follows the address, not its position.
// CHECK-LABEL: @narrow_uniform_along_dim_one
tt.func @narrow_uniform_along_dim_one(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %off = tt.broadcast %mCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<32x1x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<32x1xf16> -> tensor<32x128xf16>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// A dequantizing GEMM reads both a scale and a zero point per group.
// CHECK-LABEL: @narrow_two_loads
tt.func @narrow_two_loads(%scaleBase: !tt.ptr<f16>, %zeroBase: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %off = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %scalePtrs = tt.splat %scaleBase : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %scaleAddr = tt.addptr %scalePtrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[SCALE:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NEXT: %[[SCALE_FULL:.*]] = tt.broadcast %[[SCALE]] : tensor<1x128xf16> -> tensor<32x128xf16>
  %scale = tt.load %scaleAddr : tensor<32x128x!tt.ptr<f16>>
  %zeroPtrs = tt.splat %zeroBase : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %zeroAddr = tt.addptr %zeroPtrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[ZERO:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NEXT: %[[ZERO_FULL:.*]] = tt.broadcast %[[ZERO]] : tensor<1x128xf16> -> tensor<32x128xf16>
  %zero = tt.load %zeroAddr : tensor<32x128x!tt.ptr<f16>>
  // CHECK: arith.subf %[[SCALE_FULL]], %[[ZERO_FULL]]
  %res = arith.subf %scale, %zero : tensor<32x128xf16>
  tt.return %res : tensor<32x128xf16>
}

// Only the address is narrowed; uses of the loaded values stay on the full tile.
// CHECK-LABEL: @narrow_integer_load
tt.func @narrow_integer_load(%base: !tt.ptr<i8>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %off = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<i8> -> tensor<32x128x!tt.ptr<i8>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<i8>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<i8>>
  // CHECK-NEXT: %[[FULL:.*]] = tt.broadcast %[[VAL]] : tensor<1x128xi8> -> tensor<32x128xi8>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<i8>>
  // CHECK: arith.sitofp %[[FULL]]
  %res = arith.sitofp %val : tensor<32x128xi8> to tensor<32x128xf16>
  tt.return %res : tensor<32x128xf16>
}

// Elementwise arithmetic in the address is re-materialized by cloning it over
// narrowed operands.
// CHECK-LABEL: @narrow_address_through_arithmetic
tt.func @narrow_address_through_arithmetic(%base: !tt.ptr<f16>, %packed: i32) -> tensor<32x128xf16> {
  %maskBits = arith.constant dense<255> : tensor<1x128xi32>
  %shift = arith.constant dense<8> : tensor<1x128xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %packedT = tt.splat %packed : i32 -> tensor<1x128xi32>
  %biased = arith.addi %packedT, %nRow : tensor<1x128xi32>
  %shifted = arith.shrui %biased, %shift : tensor<1x128xi32>
  %group = arith.andi %shifted, %maskBits : tensor<1x128xi32>
  %off = tt.broadcast %group : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// Narrowing a volatile load would change how many times it is issued.
// CHECK-LABEL: @no_narrow_volatile_load
tt.func @no_narrow_volatile_load(%base: !tt.ptr<f16>) -> tensor<16x64xf16> {
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  // CHECK: tt.load %{{.*}} {isVolatile = true} : tensor<16x64x!tt.ptr<f16>>
  %val = tt.load %ptrs {isVolatile = true} : tensor<16x64x!tt.ptr<f16>>
  tt.return %val : tensor<16x64xf16>
}

// A load off a bare splat reads one address for the whole tile.
// CHECK-LABEL: @narrow_fully_uniform
tt.func @narrow_fully_uniform(%base: !tt.ptr<f16>) -> tensor<16x64xf16> {
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<16x64x!tt.ptr<f16>>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x1x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<1x1xf16> -> tensor<16x64xf16>
  %val = tt.load %ptrs : tensor<16x64x!tt.ptr<f16>>
  tt.return %val : tensor<16x64xf16>
}

// Narrowing both dimensions at once makes tt.expand_dims drop its axis while
// the dimension it expands is itself collapsing.
// CHECK-LABEL: @narrow_uniform_along_both_dims
tt.func @narrow_uniform_along_both_dims(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %groupSize = arith.constant dense<32> : tensor<32x1xi32>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %group = arith.divui %kCol, %groupSize : tensor<32x1xi32>
  %off = tt.broadcast %group : tensor<32x1xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}} : tensor<1x1x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<1x1xf16> -> tensor<32x128xf16>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// An ordinary tiled load reads a distinct element per lane along both dims.
// CHECK-LABEL: @no_narrow_contiguous_tile
tt.func @no_narrow_contiguous_tile(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %stride = arith.constant dense<128> : tensor<32x1xi32>
  %mRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %mCol = tt.expand_dims %mRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %row = arith.muli %mCol, %stride : tensor<32x1xi32>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %rowB = tt.broadcast %row : tensor<32x1xi32> -> tensor<32x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %off = arith.addi %rowB, %nB : tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: tt.load %{{.*}} : tensor<32x128x!tt.ptr<f16>>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// A mask uniform along the redundant dim narrows with the address, and its zero
// fill narrows as a splat.
// CHECK-LABEL: @narrow_masked_uniform_along_dim
tt.func @narrow_masked_uniform_along_dim(%base: !tt.ptr<f16>, %n: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %limit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMask = arith.cmpi slt, %nRow, %limit : tensor<1x128xi32>
  %mask = tt.broadcast %nMask : tensor<1x128xi1> -> tensor<32x128xi1>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: %[[VAL:.*]] = tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<1x128x!tt.ptr<f16>>
  // CHECK-NEXT: tt.broadcast %[[VAL]] : tensor<1x128xf16> -> tensor<32x128xf16>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// The address is uniform along dim 0 but the mask is not, so which lane a value
// comes from decides whether it is the loaded value or the fill.
// CHECK-LABEL: @no_narrow_mask_varies_along_dim
tt.func @no_narrow_mask_varies_along_dim(%base: !tt.ptr<f16>, %k: i32) -> tensor<32x128xf16> {
  %zero = arith.constant dense<0.0> : tensor<32x128xf16>
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %limit = tt.splat %k : i32 -> tensor<32x1xi32>
  %kMask = arith.cmpi slt, %kCol, %limit : tensor<32x1xi32>
  %mask = tt.broadcast %kMask : tensor<32x1xi1> -> tensor<32x128xi1>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f16>>
  %val = tt.load %addr, %mask, %zero : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// The analysis sees through a transpose, but the rewrite has no rule for one.
// Nothing of the half-built narrow address is left behind.
// CHECK-LABEL: @no_narrow_address_not_rematerializable
tt.func @no_narrow_address_not_rematerializable(%base: !tt.ptr<f16>) -> tensor<32x128xf16> {
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nCol = tt.expand_dims %nRange {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
  %wide = tt.broadcast %nCol : tensor<128x1xi32> -> tensor<128x32xi32>
  %off = tt.trans %wide {order = array<i32: 1, 0>} : tensor<128x32xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %off : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // The CHECK-NOT covers the rolled-back address, which would otherwise be
  // emitted ahead of the load and left dead.
  // CHECK-NOT: tensor<1x128x!tt.ptr<f16>>
  // CHECK: tt.load %{{.*}} : tensor<32x128x!tt.ptr<f16>>
  %val = tt.load %addr : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}

// Address and mask are uniform along dim 0 but the fill is not, and the analysis
// does not account for it.
// CHECK-LABEL: @no_narrow_fill_varies_along_dim
tt.func @no_narrow_fill_varies_along_dim(%base: !tt.ptr<f16>, %n: i32) -> tensor<32x128xf16> {
  %kRange = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
  %kCol = tt.expand_dims %kRange {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
  %kB = tt.broadcast %kCol : tensor<32x1xi32> -> tensor<32x128xi32>
  %fill = arith.sitofp %kB : tensor<32x128xi32> to tensor<32x128xf16>
  %nRange = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %nRow = tt.expand_dims %nRange {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  %limit = tt.splat %n : i32 -> tensor<1x128xi32>
  %nMask = arith.cmpi slt, %nRow, %limit : tensor<1x128xi32>
  %mask = tt.broadcast %nMask : tensor<1x128xi1> -> tensor<32x128xi1>
  %nB = tt.broadcast %nRow : tensor<1x128xi32> -> tensor<32x128xi32>
  %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
  %addr = tt.addptr %ptrs, %nB : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
  // CHECK: tt.load %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x128x!tt.ptr<f16>>
  %val = tt.load %addr, %mask, %fill : tensor<32x128x!tt.ptr<f16>>
  tt.return %val : tensor<32x128xf16>
}
