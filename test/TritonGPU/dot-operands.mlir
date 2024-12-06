// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -canonicalize | FileCheck %s

#Cv2 = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#Av2k1 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=1}>
#Bv2k1 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=1}>
#Av2k2 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=2}>
#Bv2k2 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=2}>
#Av2k4 = #ttg.dot_op<{opIdx = 0, parent = #Cv2, kWidth=4}>
#Bv2k4 = #ttg.dot_op<{opIdx = 1, parent = #Cv2, kWidth=4}>
#ALR = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALC = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#BLR = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLC = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: tt.func @push_elementwise
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]] {{.*}} #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
// CHECK: %[[BCVT:.*]] = ttg.convert_layout %{{.*}} : {{.*}} tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
// CHECK: %[[C:.*]] = tt.dot %[[AF16]], %[[BCVT]]
// CHECK-SAME: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<16x16xf32, #mma>
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = ttg.convert_layout %a : tensor<16x16xf16, #ALR> -> tensor<16x16xf16, #Av2k4>
  %dotb = ttg.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// CHECK: tt.func @succeeds_if_arg_is_not_convert_layout
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @succeeds_if_arg_is_not_convert_layout(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %dotai8 = ttg.convert_layout %ai8 : tensor<16x16xi8, #ALR> -> tensor<16x16xi8, #Av2k4>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #BLC>
  %dotaf8 = tt.bitcast %dotai8 : tensor<16x16xi8, #Av2k4> -> tensor<16x16xf8E5M2, #Av2k4>
  %dota = tt.fp_to_fp %dotaf8 : tensor<16x16xf8E5M2, #Av2k4> -> tensor<16x16xf16, #Av2k4>
  %dotb = ttg.convert_layout %b : tensor<16x16xf16, #BLC> -> tensor<16x16xf16, #Bv2k4>
  %newc = tt.dot %dota, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}

// CHECK: tt.func @push_inline_asm_op
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.elementwise_inline_asm {{.*}} %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_inline_asm_op(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %dotb: tensor<16x16xf16, #Bv2k4>,
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa : tensor<16x16x!tt.ptr<i8>, #ALR>
  %dotaf8 = tt.bitcast %ai8 : tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %dota = tt.elementwise_inline_asm "{ cvt.rn.satfinite.e4m3x2.f16x2 $0, $1; }" {constraints = "=r,r", packed_element = 2 : i32, pure = true} %dotaf8 : tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota_cvt = ttg.convert_layout %dota : tensor<16x16xf16, #ALR> -> tensor<16x16xf16, #Av2k4>
  %newc = tt.dot %dota_cvt, %dotb, %c : tensor<16x16xf16, #Av2k4> * tensor<16x16xf16, #Bv2k4> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}

}

// -----

#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: #[[BA:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @push_convert_both_operands
// CHECK: %[[ALOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BA]]>
// CHECK: %[[BLOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BB]]>
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BCVT:.*]] = ttg.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: tt.dot %[[AEXT]], %[[BEXT]], %{{.*}} : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @push_convert_both_operands(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blockedA>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %al = ttg.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = ttg.convert_layout %be : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}

}

// -----

#blockedA = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:80"} {

// CHECK: #[[BA:.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #[[BB:.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #[[MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = []}>

// CHECK: tt.func @update_kwidth_slice
// CHECK: %[[CST:.+]] = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[ALOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BA]]>
// CHECK: %[[BLOAD:.*]] = tt.load %{{.*}} : tensor<16x16x!tt.ptr<f16>, #[[BB]]>
// CHECK: %[[ACVT:.*]] = ttg.convert_layout %[[ALOAD]] : tensor<16x16xf16, #[[BA]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[AEXT:.*]] = arith.extf %[[ACVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BCVT:.*]] = ttg.convert_layout %[[BLOAD]] : tensor<16x16xf16, #[[BB]]> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[BEXT:.*]] = arith.extf %[[BCVT]] : tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> to tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: %[[ADD:.+]] = arith.addf %[[BEXT]], %[[CST]] : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>>
// CHECK: tt.dot %[[AEXT]], %[[ADD]], %{{.*}} : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #[[MMA]], kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #[[MMA]], kWidth = 2}>> -> tensor<16x16xf32, #mma>
tt.func @update_kwidth_slice(
                   %pa: tensor<16x16x!tt.ptr<f16>, #blockedA> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #blockedB> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #mma>) -> tensor<16x16xf32, #mma>{
  %cst = arith.constant dense<1.000000e+00> : tensor<16x16xf32, #blockedB>
  %a = tt.load %pa : tensor<16x16x!tt.ptr<f16>, #blockedA>
  %b = tt.load %pb : tensor<16x16x!tt.ptr<f16>, #blockedB>
  %ae = arith.extf %a : tensor<16x16xf16, #blockedA> to tensor<16x16xf32, #blockedA>
  %be = arith.extf %b : tensor<16x16xf16, #blockedB> to tensor<16x16xf32, #blockedB>
  %add = arith.addf %be, %cst : tensor<16x16xf32, #blockedB>
  %al = ttg.convert_layout %ae : tensor<16x16xf32, #blockedA> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
  %bl = ttg.convert_layout %add : tensor<16x16xf32, #blockedB> -> tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
  %r = tt.dot %al, %bl, %c : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
  tt.return %r : tensor<16x16xf32, #mma>
}

}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A
//    CHECK: %[[A:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf16, #mma> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: ttng.warp_group_dot %[[A]], {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A(%arg0: tensor<128x64xf16, #mma>, %arg1: !ttg.memdesc<64x64xf16, #shared, #smem>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = ttg.local_alloc %arg0 : (tensor<128x64xf16, #mma>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
  %r = ttng.warp_group_dot %A, %arg1, %arg2 : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_operand_A_fp8
//    CHECK: %[[A:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
//    CHECK: ttng.warp_group_dot %[[A]], {{.*}} : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * !ttg.memdesc<64x64xf8E5M2, #shared, #smem> -> tensor<128x64xf32, #mma>
tt.func @mma_v3_reg_operand_A_fp8(%arg0: tensor<128x64xf8E5M2, #mma>, %arg1: !ttg.memdesc<64x64xf8E5M2, #shared, #smem>, %arg2: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
  %A = ttg.local_alloc %arg0 : (tensor<128x64xf8E5M2, #mma>) -> !ttg.memdesc<128x64xf8E5M2, #shared1, #smem>
  %r = ttng.warp_group_dot %A, %arg1, %arg2 : !ttg.memdesc<128x64xf8E5M2, #shared1, #smem> * !ttg.memdesc<64x64xf8E5M2, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.return %r : tensor<128x64xf32, #mma>
}
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @a_impl
// CHECK-NOT: %[[SELECT:.*]] = arith.select {{.*}} : tensor<128x128xi1, #ttg.dot_op<{{.*}}>, tensor<128x128xf16, #ttg.dot_op<{{.*}}>
  tt.func @a_impl(%pa: tensor<128x128x!tt.ptr<f16>, #blocked>) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_3 = arith.constant dense<5> : tensor<128x1xi32, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked>
    %tl = tt.load %pa : tensor<128x128x!tt.ptr<f16>, #blocked>
    %tr = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %te = tt.expand_dims %tr {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %tc = arith.cmpi slt, %te, %cst_3 : tensor<128x1xi32, #blocked>
    %tb = tt.broadcast %tc : tensor<128x1xi1, #blocked> -> tensor<128x128xi1, #blocked>
    %ts = arith.select %tb, %tl, %cst_4 : tensor<128x128xi1, #blocked>, tensor<128x128xf16, #blocked>
    %conv = ttg.convert_layout %ts : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %td = tt.dot %cst_0, %conv, %cst : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    tt.return %td : tensor<128x128xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<bf16>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xbf16, #blocked> -> tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = tt.fp_to_fp %[[A_DOTOP]] : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_CASTED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise(%pa: tensor<128x64x!tt.ptr<bf16>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a_bf16 = tt.load %pa : tensor<128x64x!tt.ptr<bf16>, #blocked>
    %a = tt.fp_to_fp %a_bf16 : tensor<128x64xbf16, #blocked> -> tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK: tt.func @mma_v3_reg_push_elementwise_chained
//    CHECK: %[[CST_DOTOP:.*]] = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_BLOCK:.*]] = tt.load %{{.*}} : tensor<128x64x!tt.ptr<i8>, #blocked>
//    CHECK: %[[A_DOTOP:.*]] = ttg.convert_layout %[[A_BLOCK]] : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_CASTED:.*]] = arith.sitofp %[[A_DOTOP]] : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> to tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_SCALED:.*]] = arith.mulf %[[A_CASTED]], %[[CST_DOTOP]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[A_NEGATED:.*]] = arith.negf %[[A_SCALED]] : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
//    CHECK: %[[R:.*]] = ttng.warp_group_dot %[[A_NEGATED]], %{{.*}}, %{{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
  tt.func @mma_v3_reg_push_elementwise_chained(%pa: tensor<128x64x!tt.ptr<i8>, #blocked>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %a_i8 = tt.load %pa : tensor<128x64x!tt.ptr<i8>, #blocked>
    %a_f16 = arith.sitofp %a_i8 : tensor<128x64xi8, #blocked> to tensor<128x64xf16, #blocked>
    %a_scaled = arith.mulf %a_f16, %cst : tensor<128x64xf16, #blocked>
    %a_negated = arith.negf %a_scaled : tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a_negated: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mma_reorder_transpose
// CHECK: ttg.local_alloc
// CHECK: ttg.memdesc_trans
// CHECK: ttng.warp_group_dot
  tt.func @mma_reorder_transpose(%t: tensor<64x128xf16, #blocked1>, %dotb: !ttg.memdesc<64x64xf16, #shared, #smem>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<64x128xf16, #blocked1> -> tensor<128x64xf16, #blocked>
    %dota = ttg.local_alloc %a: (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared1, #smem>
    %r = ttng.warp_group_dot %dota, %dotb, %dotc : !ttg.memdesc<128x64xf16, #shared1, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: mmav2_reorder_transpose
// CHECK: ttg.local_alloc
// CHECK: ttg.memdesc_trans
// CHECK: ttg.local_load
// CHECK: tt.dot
  tt.func @mmav2_reorder_transpose(%t: tensor<32x128xf16, #blocked1>, %dotb: tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %dotc: tensor<128x64xf32, #mma>) -> tensor<128x64xf32, #mma>{
    %a = tt.trans %t {order = array<i32: 1, 0>} : tensor<32x128xf16, #blocked1> -> tensor<128x32xf16, #blocked>
    %cv = ttg.convert_layout %a : tensor<128x32xf16, #blocked> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %r = tt.dot %cv, %dotb, %dotc, inputPrecision = tf32 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x64xf32, #mma>
    tt.return %r : tensor<128x64xf32, #mma>
  }
}
