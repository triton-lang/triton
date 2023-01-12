// RUN: triton-opt %s -split-input-file -tritongpu-combine -tritongpu-update-mma-for-volta 2>&1 | FileCheck %s

// -----

// check the UpdateMMAVersionMinorForVolta pattern
#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 4], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8 ,order = [1, 0]}>
#mma0 = #triton_gpu.mma<{versionMajor=1, versionMinor=0, warpsPerCTA=[4,4]}>
// Here, the isMMAv1Row of a and b's dot_operands mismatch #mma0's versionMinor,
// and the pattern should update the versionMinor.
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, isMMAv1Row=true}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, isMMAv1Row=false}>
// It creates a new MMA layout to fit with $a and $b's dot_operand, and get the right warpsPerCTA
// The ID of this MMA instance should be 0.
// CHECK: [[new_mma:#mma.*]] = #triton_gpu.mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [4, 4]}>
module attributes {"triton_gpu.num-warps" = 16 : i32} {
  // CHECK-LABEL: dot_mmav1
  func @dot_mmav1(%A: tensor<64x64xf16, #blocked0>, %B: tensor<64x64xf16, #blocked0>) -> tensor<64x64xf32, #blocked0> {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked0>
    %AA = triton_gpu.convert_layout %A : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_a>
    %BB = triton_gpu.convert_layout %B : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_b>
    %CC = triton_gpu.convert_layout %C : (tensor<64x64xf32, #blocked0>) -> tensor<64x64xf32, #mma0>

    // CHECK: {{.*}} = tt.dot {{.*}}, {{.*}}, %cst {allowTF32 = true} : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[new_mma]], isMMAv1Row = true}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[new_mma]], isMMAv1Row = true}>> -> tensor<64x64xf32, [[new_mma]]>
    %D = tt.dot %AA, %BB, %CC {allowTF32 = true} : tensor<64x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<64x64xf32, #mma0>
    %res = triton_gpu.convert_layout %D : (tensor<64x64xf32, #mma0>) -> tensor<64x64xf32, #blocked0>

    return %res : tensor<64x64xf32, #blocked0>
  }
}


// -----
// Check id in multiple MMA layout instances

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 4], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8 ,order = [1, 0]}>
#mma0 = #triton_gpu.mma<{versionMajor=1, versionMinor=0, warpsPerCTA=[4,4]}>
// mma id=1, with all other boolean flags be false, should get a versionMinor of 16(= 1 * 1<<4)
#mma1 = #triton_gpu.mma<{versionMajor=1, versionMinor=16, warpsPerCTA=[4,4]}>

// Will still get two MMA layouts
// CHECK: [[new_mma:#mma.*]] = #triton_gpu.mma<{versionMajor = 1, versionMinor = 3, warpsPerCTA = [4, 4]}>
// CHECK: [[new_mma1:#mma.*]] = #triton_gpu.mma<{versionMajor = 1, versionMinor = 19, warpsPerCTA = [4, 4]}>

#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, isMMAv1Row=true}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, isMMAv1Row=false}>
#dot_operand_a1 = #triton_gpu.dot_op<{opIdx=0, parent=#mma1, isMMAv1Row=true}>
#dot_operand_b1 = #triton_gpu.dot_op<{opIdx=1, parent=#mma1, isMMAv1Row=false}>

module attributes {"triton_gpu.num-warps" = 16 : i32} {
  // CHECK-LABEL: dot_mmav1
  func @dot_mmav1(%A: tensor<64x64xf16, #blocked0>, %B: tensor<64x64xf16, #blocked0>) -> tensor<64x64xf32, #blocked0> {
    %C = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked0>
    %AA = triton_gpu.convert_layout %A : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_a>
    %BB = triton_gpu.convert_layout %B : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_b>
    %CC = triton_gpu.convert_layout %C : (tensor<64x64xf32, #blocked0>) -> tensor<64x64xf32, #mma0>

    %AA1 = triton_gpu.convert_layout %A : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_a1>
    %BB1 = triton_gpu.convert_layout %B : (tensor<64x64xf16, #blocked0>) -> tensor<64x64xf16, #dot_operand_b1>
    %CC1 = triton_gpu.convert_layout %C : (tensor<64x64xf32, #blocked0>) -> tensor<64x64xf32, #mma1>

    // CHECK: {{.*}} = tt.dot {{.*}}, {{.*}}, {{.*}} {allowTF32 = true} : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[new_mma]], isMMAv1Row = true}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[new_mma]], isMMAv1Row = true}>> -> tensor<64x64xf32, [[new_mma]]>
    // CHECK: {{.*}} = tt.dot {{.*}}, {{.*}}, {{.*}} {allowTF32 = true} : tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = [[new_mma1]], isMMAv1Row = true}>> * tensor<64x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = [[new_mma1]], isMMAv1Row = true}>> -> tensor<64x64xf32, [[new_mma1]]>
    %D = tt.dot %AA, %BB, %CC {allowTF32 = true} : tensor<64x64xf16, #dot_operand_a> * tensor<64x64xf16, #dot_operand_b> -> tensor<64x64xf32, #mma0>
    %D1 = tt.dot %AA1, %BB1, %CC1 {allowTF32 = true} : tensor<64x64xf16, #dot_operand_a1> * tensor<64x64xf16, #dot_operand_b1> -> tensor<64x64xf32, #mma1>
    %res = triton_gpu.convert_layout %D : (tensor<64x64xf32, #mma0>) -> tensor<64x64xf32, #blocked0>
    %res1 = triton_gpu.convert_layout %D1 : (tensor<64x64xf32, #mma1>) -> tensor<64x64xf32, #blocked0>
    %sum = arith.addf %res, %res1 : tensor<64x64xf32, #blocked0>

    return %sum : tensor<64x64xf32, #blocked0>
  }
}
