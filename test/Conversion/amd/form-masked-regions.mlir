// RUN: triton-opt %s -split-input-file --triton-amdgpu-form-masked-regions | FileCheck %s

// CHECK-LABEL: llvm.func @two_loads_same_mask
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[ONE:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK: %{{.*}}:2 = amdg.masked_region %[[ACTIVE]] else(%[[ZERO]], %[[ONE]]) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: } : i32, i32 -> i32, i32
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @two_loads_same_mask(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %active, %one : (!llvm.ptr, i1, i32) -> i32
    %sum = llvm.add %a, %b : i32
    llvm.store %sum, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @load_load_store_same_mask
// CHECK: amdg.masked_region %{{.*}} {
// CHECK:   %[[A:.*]] = llvm.load
// CHECK:   %[[B:.*]] = llvm.load
// CHECK:   %[[SUM:.*]] = llvm.add %[[A]], %[[B]] : i32
// CHECK:   llvm.store %[[SUM]],
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @load_load_store_same_mask(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %sum = llvm.add %a, %b : i32
    amdg.masked_store %dst, %sum, %active : !llvm.ptr, i32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @unrelated_pure_op_between_loads_can_stay_outside
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK-SAME: %[[X:[A-Za-z0-9_]+]]: i32, %[[Y:[A-Za-z0-9_]+]]: i32
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %[[ACTIVE]]
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: } : i32, i32 -> i32, i32
// CHECK: %[[TMP:.*]] = llvm.add %[[X]], %[[Y]] : i32
// CHECK: %[[SUM:.*]] = llvm.add %[[REGION]]#0, %[[REGION]]#1 : i32
// CHECK: llvm.add %[[SUM]], %[[TMP]] : i32
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @unrelated_pure_op_between_loads_can_stay_outside(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %x: i32, %y: i32) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %tmp = llvm.add %x, %y : i32
    %b = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %sum = llvm.add %a, %b : i32
    %out = llvm.add %sum, %tmp : i32
    llvm.store %out, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @escaped_extracts_between_loads
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %[[ACTIVE]] else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xf32>
// CHECK: } : vector<4xf32>, vector<4xf32> -> vector<4xf32>, vector<4xf32>
// CHECK: llvm.extractelement %[[REGION]]#0[%[[C0]] : i32] : vector<4xf32>
// CHECK: llvm.extractelement %[[REGION]]#1[%[[C0]] : i32] : vector<4xf32>
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @escaped_extracts_between_loads(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<4xf32>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %v0 = amdg.masked_load %src0, %active, %zero_vec : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e0 = llvm.extractelement %v0[%c0 : i32] : vector<4xf32>
    %v1 = amdg.masked_load %src1, %active, %zero_vec : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e1 = llvm.extractelement %v1[%c0 : i32] : vector<4xf32>
    %sum = llvm.fadd %e0, %e1 : f32
    llvm.store %sum, %dst : f32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @hoist_false_values_defined_between_loads
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: %[[FALSE0:.*]] = llvm.insertelement
// CHECK: %[[FALSE1:.*]] = llvm.insertelement
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %[[ACTIVE]] else(%[[FALSE0]], %[[FALSE1]]) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xf32>
// CHECK: } : vector<4xf32>, vector<4xf32> -> vector<4xf32>, vector<4xf32>
// CHECK: llvm.extractelement %[[REGION]]#0
// CHECK: llvm.extractelement %[[REGION]]#1
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @hoist_false_values_defined_between_loads(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %other0: f32, %other1: f32) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %undef0 = llvm.mlir.undef : vector<4xf32>
    %false0 = llvm.insertelement %other0, %undef0[%c0 : i32] : vector<4xf32>
    %v0 = amdg.masked_load %src0, %active, %false0 : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e0 = llvm.extractelement %v0[%c0 : i32] : vector<4xf32>
    %undef1 = llvm.mlir.undef : vector<4xf32>
    %false1 = llvm.insertelement %other1, %undef1[%c0 : i32] : vector<4xf32>
    %v1 = amdg.masked_load %src1, %active, %false1 : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e1 = llvm.extractelement %v1[%c0 : i32] : vector<4xf32>
    %sum = llvm.fadd %e0, %e1 : f32
    llvm.store %sum, %dst : f32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @aggregate_masks_from_same_value_group
// CHECK: amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<4xf32>, vector<4xf32>
// CHECK: } : vector<4xf32>, vector<4xf32> -> vector<4xf32>, vector<4xf32>
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @aggregate_masks_from_same_value_group(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<4xf32>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %agg0 = llvm.insertvalue %active, %undef[0] : !llvm.struct<(i1, i1)>
    %agg1 = llvm.insertvalue %active, %agg0[1] : !llvm.struct<(i1, i1)>
    %m0 = llvm.extractvalue %agg1[0] : !llvm.struct<(i1, i1)>
    %m1 = llvm.extractvalue %agg1[1] : !llvm.struct<(i1, i1)>
    %v0 = amdg.masked_load %src0, %m0, %zero_vec : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e0 = llvm.extractelement %v0[%c0 : i32] : vector<4xf32>
    %v1 = amdg.masked_load %src1, %m1, %zero_vec : (!llvm.ptr, i1, vector<4xf32>) -> vector<4xf32>
    %e1 = llvm.extractelement %v1[%c0 : i32] : vector<4xf32>
    %sum = llvm.fadd %e0, %e1 : f32
    llvm.store %sum, %dst : f32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @aggregate_mask_materialized_between_loads
// CHECK: amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xi32>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xi32>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<1xi32>, vector<1xi32>
// CHECK: } : vector<1xi32>, vector<1xi32> -> vector<1xi32>, vector<1xi32>
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @aggregate_mask_materialized_between_loads(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<1xi32>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %agg0 = llvm.insertvalue %active, %undef[0] : !llvm.struct<(i1, i1)>
    %m0 = llvm.extractvalue %agg0[0] : !llvm.struct<(i1, i1)>
    %v0 = amdg.masked_load %src0, %m0, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    %e0 = llvm.extractelement %v0[%c0 : i32] : vector<1xi32>
    %agg1 = llvm.insertvalue %active, %agg0[1] : !llvm.struct<(i1, i1)>
    %m1 = llvm.extractvalue %agg1[1] : !llvm.struct<(i1, i1)>
    %v1 = amdg.masked_load %src1, %m1, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    %e1 = llvm.extractelement %v1[%c0 : i32] : vector<1xi32>
    %sum = llvm.add %e0, %e1 : i32
    llvm.store %sum, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @overlapping_aggregate_insert_blocks_grouping
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @overlapping_aggregate_insert_blocks_grouping(%old: i1, %new: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %zero_vec: vector<1xi32>) {
    %undef_inner = llvm.mlir.undef : !llvm.struct<(i1)>
    %undef_outer = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(i1)>)>
    %old_inner = llvm.insertvalue %old, %undef_inner[0] : !llvm.struct<(i1)>
    %old_outer = llvm.insertvalue %old_inner, %undef_outer[0] : !llvm.struct<(!llvm.struct<(i1)>)>
    %new_inner = llvm.insertvalue %new, %undef_inner[0] : !llvm.struct<(i1)>
    %new_outer = llvm.insertvalue %new_inner, %old_outer[0] : !llvm.struct<(!llvm.struct<(i1)>)>
    %m0 = llvm.extractvalue %old_outer[0, 0] : !llvm.struct<(!llvm.struct<(i1)>)>
    %m1 = llvm.extractvalue %new_outer[0, 0] : !llvm.struct<(!llvm.struct<(i1)>)>
    %a = amdg.masked_load %src0, %m0, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    %b = amdg.masked_load %src1, %m1, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @different_scalarized_predicates_do_not_group
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @different_scalarized_predicates_do_not_group(%row: i1, %pid_n: i32, %N: i32, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %c7 = llvm.mlir.constant(7 : i32) : i32
    %c64 = llvm.mlir.constant(64 : i32) : i32
    %base = llvm.mul %pid_n, %c64 : i32
    %col0 = llvm.add %base, %zero : i32
    %col7 = llvm.add %base, %c7 : i32
    %col0_in_bounds = llvm.icmp "slt" %col0, %N : i32
    %col7_in_bounds = llvm.icmp "slt" %col7, %N : i32
    %mask0 = llvm.and %row, %col0_in_bounds : i1
    %mask7 = llvm.and %row, %col7_in_bounds : i1
    %a = amdg.masked_load %src0, %mask0, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %mask7, %zero : (!llvm.ptr, i1, i32) -> i32
    %sum = llvm.add %a, %b : i32
    llvm.store %sum, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @store_only_same_mask
// CHECK: amdg.masked_region %{{.*}} {
// CHECK:   llvm.store
// CHECK:   llvm.store
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @store_only_same_mask(%active: i1, %src: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr) {
    %v = llvm.load %src : !llvm.ptr -> i32
    amdg.masked_store %dst0, %v, %active : !llvm.ptr, i32, i1
    amdg.masked_store %dst1, %v, %active : !llvm.ptr, i32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @cv_load_wt_store_form_volatile_region
// CHECK: amdg.masked_region %{{.*}} {
// CHECK:   %[[VALUE:.*]] = llvm.load volatile
// CHECK:   llvm.store volatile %[[VALUE]],
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @cv_load_wt_store_form_volatile_region(%active: i1, %src: !llvm.ptr, %dst: !llvm.ptr, %zero: i32) {
    %value = amdg.masked_load %src, %active, %zero cacheModifier = cv : (!llvm.ptr, i1, i32) -> i32
    amdg.masked_store %dst, %value, %active cacheModifier = wt : !llvm.ptr, i32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @different_masks_do_not_group
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @different_masks_do_not_group(%mask0: i1, %mask1: i1, %src0: !llvm.ptr, %src1: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %mask0, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %mask1, %zero : (!llvm.ptr, i1, i32) -> i32
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @different_aggregate_masks_do_not_group
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @different_aggregate_masks_do_not_group(%mask0: i1, %mask1: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %zero_vec: vector<1xi32>) {
    %undef = llvm.mlir.undef : !llvm.struct<(i1, i1)>
    %agg0 = llvm.insertvalue %mask0, %undef[0] : !llvm.struct<(i1, i1)>
    %agg1 = llvm.insertvalue %mask1, %agg0[1] : !llvm.struct<(i1, i1)>
    %m0 = llvm.extractvalue %agg1[0] : !llvm.struct<(i1, i1)>
    %m1 = llvm.extractvalue %agg1[1] : !llvm.struct<(i1, i1)>
    %a = amdg.masked_load %src0, %m0, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    %b = amdg.masked_load %src1, %m1, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @intervening_store_blocks_grouping
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: llvm.store
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @intervening_store_blocks_grouping(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %side: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    llvm.store %a, %side : i32, !llvm.ptr
    %b = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @readfirstlane_blocks_grouping
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: rocdl.readfirstlane
// CHECK: amdg.masked_load
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @readfirstlane_blocks_grouping(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %lane_value: i32, %zero_vec: vector<1xi32>) {
    %v0 = amdg.masked_load %src0, %active, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    %lane = rocdl.readfirstlane %lane_value : i32
    %v1 = amdg.masked_load %src1, %active, %zero_vec : (!llvm.ptr, i1, vector<1xi32>) -> vector<1xi32>
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @multicast_loads_do_not_form_region
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @multicast_loads_do_not_form_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %multicast = llvm.mlir.constant(3 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero, %multicast : (!llvm.ptr, i1, i32, i32) -> i32
    %b = amdg.masked_load %src1, %active, %zero, %multicast : (!llvm.ptr, i1, i32, i32) -> i32
    llvm.return
  }
}
