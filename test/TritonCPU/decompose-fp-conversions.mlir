// RUN: triton-opt %s -split-input-file -triton-cpu-decompose-fp-conversions="decompose-bf16-conversions=true" | FileCheck %s

// fp32 -> bf16 rounds to nearest even by default and keeps NaNs quiet.

// CHECK-LABEL: @fp32_to_bf16_rne
// CHECK-DAG:     %[[C16:.+]] = arith.constant dense<16> : vector<8xi32>
// CHECK-DAG:     %[[C1:.+]] = arith.constant dense<1> : vector<8xi32>
// CHECK-DAG:     %[[BIAS:.+]] = arith.constant dense<32767> : vector<8xi32>
// CHECK:         %[[SRC:.+]] = arith.bitcast %{{.+}} : vector<8xf32> to vector<8xi32>
// CHECK:         %[[HI:.+]] = arith.shrui %[[SRC]], %[[C16]]
// CHECK:         %[[LSB:.+]] = arith.andi %[[HI]], %[[C1]]
// CHECK:         %[[ADD:.+]] = arith.addi %[[LSB]], %[[BIAS]]
// CHECK:         %[[ROUNDED:.+]] = arith.addi %[[SRC]], %[[ADD]]
// CHECK:         %[[ISNAN:.+]] = arith.cmpi ugt
// CHECK:         %[[QNAN:.+]] = arith.ori %[[SRC]]
// CHECK:         %[[SEL:.+]] = arith.select %[[ISNAN]], %[[QNAN]], %[[ROUNDED]]
// CHECK:         arith.shrui %[[SEL]]
// CHECK:         arith.trunci {{.+}} : vector<8xi32> to vector<8xi16>
// CHECK-NOT:     arith.truncf
tt.func @fp32_to_bf16_rne(%arg0: vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 : vector<8xf32> to vector<8xbf16>
  tt.return %0 : vector<8xbf16>
}

// -----

// Round toward zero keeps plain truncation.

// CHECK-LABEL: @fp32_to_bf16_rtz
// CHECK:         %[[C16:.+]] = arith.constant dense<16> : vector<8xi32>
// CHECK-NEXT:    %[[SRC:.+]] = arith.bitcast %{{.+}} : vector<8xf32> to vector<8xi32>
// CHECK-NEXT:    %[[HI:.+]] = arith.shrui %[[SRC]], %[[C16]]
// CHECK-NEXT:    arith.trunci %[[HI]] : vector<8xi32> to vector<8xi16>
// CHECK-NOT:     arith.truncf
tt.func @fp32_to_bf16_rtz(%arg0: vector<8xf32>) -> vector<8xbf16> {
  %0 = arith.truncf %arg0 toward_zero : vector<8xf32> to vector<8xbf16>
  tt.return %0 : vector<8xbf16>
}
