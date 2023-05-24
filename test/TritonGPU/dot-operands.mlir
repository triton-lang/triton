// RUN: triton-opt %s -split-input-file -tritongpu-optimize-dot-operands -tritongpu-remove-layout-conversions -canonicalize | FileCheck %s

#Cv2 = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#Av2 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv2, kWidth=2}>
#Bv2 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv2, kWidth=2}>
#Cv1 = #triton_gpu.mma<{versionMajor = 1, warpsPerCTA = [4, 1]}>
#Av1 = #triton_gpu.dot_op<{opIdx = 0, parent = #Cv1}>
#Bv1 = #triton_gpu.dot_op<{opIdx = 1, parent = #Cv1}>
#ALR = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALC = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#BLR = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#BLC = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [0, 1]}>

// CHECK: tt.func @push_elementwise1
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[ALOAD]]
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ACVT]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[C:.*]] = tt.dot %[[AF16]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise1(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALR>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = triton_gpu.convert_layout %a : (tensor<16x16xf16, #ALR>) -> tensor<16x16xf16, #Av2>
  %dotb = triton_gpu.convert_layout %b : (tensor<16x16xf16, #BLC>) -> tensor<16x16xf16, #Bv2>
  %newc = tt.dot %dota, %dotb, %c {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #Av2> * tensor<16x16xf16, #Bv2> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// Not modified for row-row
// CHECK: tt.func @push_elementwise2
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ALOAD]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[AF16]]
// CHECK: %[[C:.*]] = tt.dot %[[ACVT]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise2(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALR>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLR>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = triton_gpu.convert_layout %a : (tensor<16x16xf16, #ALR>) -> tensor<16x16xf16, #Av2>
  %dotb = triton_gpu.convert_layout %b : (tensor<16x16xf16, #BLR>) -> tensor<16x16xf16, #Bv2>
  %newc = tt.dot %dota, %dotb, %c {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #Av2> * tensor<16x16xf16, #Bv2> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// Not modified for col-row
// CHECK: tt.func @push_elementwise3
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ALOAD]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[AF16]]
// CHECK: %[[C:.*]] = tt.dot %[[ACVT]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise3(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALC>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLR>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALC> -> tensor<16x16xf8E5M2, #ALC>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALC> -> tensor<16x16xf16, #ALC>
  %dota = triton_gpu.convert_layout %a : (tensor<16x16xf16, #ALC>) -> tensor<16x16xf16, #Av2>
  %dotb = triton_gpu.convert_layout %b : (tensor<16x16xf16, #BLR>) -> tensor<16x16xf16, #Bv2>
  %newc = tt.dot %dota, %dotb, %c {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #Av2> * tensor<16x16xf16, #Bv2> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}

// Not modified for col-col
// CHECK: tt.func @push_elementwise4
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ALOAD]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[AF16]]
// CHECK: %[[C:.*]] = tt.dot %[[ACVT]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma>
tt.func @push_elementwise4(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv2>) -> tensor<16x16xf32, #Cv2>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALC>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALC> -> tensor<16x16xf8E5M2, #ALC>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALC> -> tensor<16x16xf16, #ALC>
  %dota = triton_gpu.convert_layout %a : (tensor<16x16xf16, #ALC>) -> tensor<16x16xf16, #Av2>
  %dotb = triton_gpu.convert_layout %b : (tensor<16x16xf16, #BLC>) -> tensor<16x16xf16, #Bv2>
  %newc = tt.dot %dota, %dotb, %c {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #Av2> * tensor<16x16xf16, #Bv2> -> tensor<16x16xf32, #Cv2>
  tt.return %newc : tensor<16x16xf32, #Cv2>
}


// Not modified for Volta
// CHECK: tt.func @push_elementwise5
// CHECK: %[[ALOAD:.*]] = tt.load %arg0
// CHECK: %[[AF8E5:.*]] = tt.bitcast %[[ALOAD]]
// CHECK: %[[AF16:.*]] = tt.fp_to_fp %[[AF8E5]]
// CHECK: %[[ACVT:.*]] = triton_gpu.convert_layout %[[AF16]]
// CHECK: %[[C:.*]] = tt.dot %[[ACVT]]
// CHECK: tt.return %[[C]] : tensor<16x16xf32, #mma1>
tt.func @push_elementwise5(
                   %pa: tensor<16x16x!tt.ptr<i8>, #ALR> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %pb: tensor<16x16x!tt.ptr<f16>, #BLC> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %c: tensor<16x16xf32, #Cv1>) -> tensor<16x16xf32, #Cv1>{
  %ai8 = tt.load %pa {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xi8, #ALR>
  %b = tt.load %pb {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BLC>
  %af8 = tt.bitcast %ai8: tensor<16x16xi8, #ALR> -> tensor<16x16xf8E5M2, #ALR>
  %a = tt.fp_to_fp %af8: tensor<16x16xf8E5M2, #ALR> -> tensor<16x16xf16, #ALR>
  %dota = triton_gpu.convert_layout %a : (tensor<16x16xf16, #ALR>) -> tensor<16x16xf16, #Av1>
  %dotb = triton_gpu.convert_layout %b : (tensor<16x16xf16, #BLC>) -> tensor<16x16xf16, #Bv1>
  %newc = tt.dot %dota, %dotb, %c {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #Av1> * tensor<16x16xf16, #Bv1> -> tensor<16x16xf32, #Cv1>
  tt.return %newc : tensor<16x16xf32, #Cv1>
}
