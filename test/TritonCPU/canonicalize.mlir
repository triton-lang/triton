// RUN: triton-opt %s -split-input-file -triton-cpu-canonicalize | FileCheck %s

// Fold transfer read and shape cast.

// CHECK-LABEL: @fold_transfer_read_shape_cast
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK:       vector.transfer_write %[[VAL]]

module {
  tt.func public @fold_transfer_read_shape_cast(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %in_p = tt.make_tensor_ptr %arg0, [%c2_i64, %c2_i64, %c16_i64, %c16_i64], [%c512_i64, %c256_i64, %c16_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x16x16xbf16>>
    %out_p = tt.make_tensor_ptr %arg1, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xbf16>>
    %memref1 = triton_cpu.extract_memref %in_p : <tensor<1x1x16x16xbf16>> -> memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>
    %indices1:4 = triton_cpu.extract_indices %in_p : <tensor<1x1x16x16xbf16>> -> index, index, index, index
    %val1 = vector.transfer_read %memref1[%indices1#0, %indices1#1, %indices1#2, %indices1#3], %cst {in_bounds = [true, true, true, true]} : memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>, vector<1x1x16x16xbf16>
    %val2 = vector.shape_cast %val1 : vector<1x1x16x16xbf16> to vector<16x16xbf16>
    %memref2 = triton_cpu.extract_memref %out_p : <tensor<16x16xbf16>> -> memref<16x16xbf16, strided<[16, 1]>>
    %indices2:2 = triton_cpu.extract_indices %out_p : <tensor<16x16xbf16>> -> index, index
    vector.transfer_write %val2, %memref2[%indices2#0, %indices2#1] {in_bounds = [true, true]} : vector<16x16xbf16>, memref<16x16xbf16, strided<[16, 1]>>
    tt.return
  }
}
