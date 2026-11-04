// RUN: triton-opt %s -inline -canonicalize | FileCheck %s
// RUN: triton-opt %s -gluon-inline -canonicalize | FileCheck %s
// RUN: triton-opt %s -inline -canonicalize -test-print-alignment -o /dev/null 2>&1 | FileCheck %s --check-prefix=AXIS

// Hints remain attached to the correct value after inlining. Annotations fold
// into single-result definitions and remain explicit for other values.
tt.func private @get_ptrs(%ptr: tensor<128x!tt.ptr<f32>>, %offsets: tensor<128xi32>) -> tensor<128x!tt.ptr<f32>> {
  %0 = tt.addptr %ptr, %offsets : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
  tt.return %0 : tensor<128x!tt.ptr<f32>>
}

// CHECK-LABEL: @caller
// CHECK-NEXT: %[[PTR:.*]] = tt.addptr %arg0, %arg1 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[PTR]]
// AXIS: tt.addptr
// AXIS-SAME: => contiguity = [16], divisibility = [16], constancy = [1]
tt.func public @caller(%arg0: tensor<128x!tt.ptr<f32>>, %arg1: tensor<128xi32>) -> tensor<128x!tt.ptr<f32>> {
  %0 = tt.call @get_ptrs(%arg0, %arg1) {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>} : (tensor<128x!tt.ptr<f32>>, tensor<128xi32>) -> tensor<128x!tt.ptr<f32>>
  tt.return %0 : tensor<128x!tt.ptr<f32>>
}

tt.func private @get_offsets(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0 = arith.addi %arg, %arg : tensor<128xi32>
  %1 = arith.muli %arg, %arg {tt.contiguity = dense<2> : tensor<1xi32>} : tensor<128xi32>
  tt.return %0, %1 : tensor<128xi32>, tensor<128xi32>
}

// Hinting one result preserves the other result's own hint.
// CHECK-LABEL: @first_result
// CHECK-NEXT: %[[A:.*]] = arith.addi %arg0, %arg0 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<32> : tensor<1xi32>}
// CHECK-NEXT: %[[B:.*]] = arith.muli %arg0, %arg0 {tt.contiguity = dense<2> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[A]], %[[B]]
// AXIS: arith.addi
// AXIS-SAME: => contiguity = [16], divisibility = [32], constancy = [1]
tt.func public @first_result(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @get_offsets(%arg) : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  %a = tt.annotate %0#0 {tt.constancy = dense<1> : tensor<1xi32>, tt.contiguity = dense<16> : tensor<1xi32>, tt.divisibility = dense<32> : tensor<1xi32>} : tensor<128xi32>
  tt.return %a, %0#1 : tensor<128xi32>, tensor<128xi32>
}

// CHECK-LABEL: @second_result
// CHECK-NEXT: %[[A:.*]] = arith.addi %arg0, %arg0 :
// CHECK-NEXT: %[[B:.*]] = arith.muli %arg0, %arg0 {tt.constancy = dense<4> : tensor<1xi32>, tt.contiguity = dense<8> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[A]], %[[B]]
// AXIS: arith.muli {{.*}}tt.constancy = dense<4>
// AXIS-SAME: => contiguity = [8], divisibility = [16], constancy = [4]
tt.func public @second_result(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @get_offsets(%arg) : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  %b = tt.annotate %0#1 {tt.constancy = dense<4> : tensor<1xi32>, tt.contiguity = dense<8> : tensor<1xi32>, tt.divisibility = dense<16> : tensor<1xi32>} : tensor<128xi32>
  tt.return %0#0, %b : tensor<128xi32>, tensor<128xi32>
}

tt.func private @pass_through(%a: tensor<128xi32>, %b: tensor<8x16xi32>) -> (tensor<8x16xi32>, tensor<128xi32>) {
  tt.return %b, %a : tensor<8x16xi32>, tensor<128xi32>
}

// Returned arguments can have different ranks and appear in any order.
// CHECK-LABEL: @pass_through_caller(
// CHECK-NEXT: %[[B:.*]] = tt.annotate %arg1 {tt.contiguity = dense<[1, 8]> : tensor<2xi32>}
// CHECK-NEXT: %[[A:.*]] = tt.annotate %arg0 {tt.contiguity = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[B]], %[[A]]
// AXIS: tt.annotate %arg1
// AXIS-SAME: => contiguity = [1, 8], divisibility = [1, 1], constancy = [1, 1]
// AXIS: tt.annotate %arg0
// AXIS-SAME: => contiguity = [16], divisibility = [32], constancy = [1]
tt.func public @pass_through_caller(%a: tensor<128xi32> {tt.divisibility = dense<32> : tensor<1xi32>}, %b: tensor<8x16xi32>) -> (tensor<8x16xi32>, tensor<128xi32>) {
  %0:2 = tt.call @pass_through(%a, %b) : (tensor<128xi32>, tensor<8x16xi32>) -> (tensor<8x16xi32>, tensor<128xi32>)
  %x = tt.annotate %0#0 {tt.contiguity = dense<[1, 8]> : tensor<2xi32>} : tensor<8x16xi32>
  // Unspecified hints are inherited from the annotated value.
  %y = tt.annotate %0#1 {tt.contiguity = dense<16> : tensor<1xi32>} : tensor<128xi32>
  tt.return %x, %y : tensor<8x16xi32>, tensor<128xi32>
}

tt.func private @split(%arg: tensor<128x2xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0, %1 = tt.split %arg : tensor<128x2xi32> -> tensor<128xi32>
  tt.return %1, %0 : tensor<128xi32>, tensor<128xi32>
}

// The replacement can itself belong to a multi-result operation.
// CHECK-LABEL: @split_caller
// CHECK-NEXT: %[[A:.*]], %[[B:.*]] = tt.split %arg0 :
// CHECK-NEXT: %[[HINT:.*]] = tt.annotate %[[B]] {tt.contiguity = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[HINT]], %[[A]]
tt.func public @split_caller(%arg: tensor<128x2xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @split(%arg) : (tensor<128x2xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  %x = tt.annotate %0#0 {tt.contiguity = dense<16> : tensor<1xi32>} : tensor<128xi32>
  tt.return %x, %0#1 : tensor<128xi32>, tensor<128xi32>
}

tt.func private @nested(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @get_offsets(%arg) : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  tt.return %0#1, %0#0 : tensor<128xi32>, tensor<128xi32>
}

// CHECK-LABEL: @nested_caller
// CHECK-NEXT: %[[A:.*]] = arith.addi %arg0, %arg0 {tt.divisibility = dense<32> : tensor<1xi32>}
// CHECK-NEXT: %[[B:.*]] = arith.muli %arg0, %arg0 {tt.contiguity = dense<8> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[B]], %[[A]]
tt.func public @nested_caller(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @nested(%arg) : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  %x = tt.annotate %0#0 {tt.contiguity = dense<8> : tensor<1xi32>} : tensor<128xi32>
  %y = tt.annotate %0#1 {tt.divisibility = dense<32> : tensor<1xi32>} : tensor<128xi32>
  tt.return %x, %y : tensor<128xi32>, tensor<128xi32>
}

tt.func private @duplicate(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  tt.return %arg, %arg : tensor<128xi32>, tensor<128xi32>
}

// Even repeated return operands keep independent annotations.
// CHECK-LABEL: @duplicate_caller
// CHECK-NEXT: %[[X:.*]] = tt.annotate %arg0 {tt.contiguity = dense<8> : tensor<1xi32>}
// CHECK-NEXT: %[[Y:.*]] = tt.annotate %arg0 {tt.contiguity = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[X]], %[[Y]]
tt.func public @duplicate_caller(%arg: tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>) {
  %0:2 = tt.call @duplicate(%arg) : (tensor<128xi32>) -> (tensor<128xi32>, tensor<128xi32>)
  %x = tt.annotate %0#0 {tt.contiguity = dense<8> : tensor<1xi32>} : tensor<128xi32>
  %y = tt.annotate %0#1 {tt.contiguity = dense<16> : tensor<1xi32>} : tensor<128xi32>
  tt.return %x, %y : tensor<128xi32>, tensor<128xi32>
}

tt.func private @single_split(%arg: tensor<128x2xi32>) -> tensor<128xi32> {
  %0, %1 = tt.split %arg : tensor<128x2xi32> -> tensor<128xi32>
  tt.return %1 : tensor<128xi32>
}

// A single call result can be replaced by one result of a multi-result op.
// CHECK-LABEL: @single_split_caller
// CHECK-NEXT: %[[A:.*]], %[[B:.*]] = tt.split %arg0 :
// CHECK-NEXT: %[[HINT:.*]] = tt.annotate %[[B]] {tt.contiguity = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[HINT]]
tt.func public @single_split_caller(%arg: tensor<128x2xi32>) -> tensor<128xi32> {
  %0 = tt.call @single_split(%arg) {tt.contiguity = dense<16> : tensor<1xi32>} : (tensor<128x2xi32>) -> tensor<128xi32>
  tt.return %0 : tensor<128xi32>
}

tt.func private @identity(%arg: tensor<128xi32>) -> tensor<128xi32> {
  tt.return %arg : tensor<128xi32>
}

// The single-result call hook also preserves hints on returned arguments.
// CHECK-LABEL: @identity_caller
// CHECK-NEXT: %[[HINT:.*]] = tt.annotate %arg0 {tt.contiguity = dense<16> : tensor<1xi32>}
// CHECK-NEXT: tt.return %[[HINT]]
tt.func public @identity_caller(%arg: tensor<128xi32>) -> tensor<128xi32> {
  %0 = tt.call @identity(%arg) {tt.contiguity = dense<16> : tensor<1xi32>} : (tensor<128xi32>) -> tensor<128xi32>
  tt.return %0 : tensor<128xi32>
}
