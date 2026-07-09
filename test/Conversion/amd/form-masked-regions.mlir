// RUN: triton-opt %s -split-input-file --triton-amdgpu-form-masked-regions | FileCheck %s

// CHECK-LABEL: llvm.func @two_loads_same_mask
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK-NOT: amdg.masked_region
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

// CHECK-LABEL: llvm.func @loads_packed_to_buffer_store
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %[[ACTIVE]] else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xf16>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xf16>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<1xf16>, vector<1xf16>
// CHECK: } : vector<1xf16>, vector<1xf16> -> vector<1xf16>, vector<1xf16>
// CHECK: %[[E0:.*]] = llvm.extractelement %[[REGION]]#0
// CHECK: %[[E1:.*]] = llvm.extractelement %[[REGION]]#1
// CHECK: %[[PACK0:.*]] = llvm.insertelement %[[E0]],
// CHECK: %[[PACK1:.*]] = llvm.insertelement %[[E1]], %[[PACK0]]
// CHECK: %[[BITS:.*]] = llvm.bitcast %[[PACK1]]
// CHECK: rocdl.raw.ptr.buffer.store %[[BITS]],
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @loads_packed_to_buffer_store(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %rsrc: !llvm.ptr<8>, %offset: i32, %soffset: i32, %aux: i32, %zero_vec: vector<1xf16>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %v0 = amdg.masked_load %src0, %active, %zero_vec : (!llvm.ptr, i1, vector<1xf16>) -> vector<1xf16>
    %v1 = amdg.masked_load %src1, %active, %zero_vec : (!llvm.ptr, i1, vector<1xf16>) -> vector<1xf16>
    %e0 = llvm.extractelement %v0[%c0 : i32] : vector<1xf16>
    %e1 = llvm.extractelement %v1[%c0 : i32] : vector<1xf16>
    %undef = llvm.mlir.undef : vector<2xf16>
    %pack0 = llvm.insertelement %e0, %undef[%c0 : i32] : vector<2xf16>
    %pack1 = llvm.insertelement %e1, %pack0[%c1 : i32] : vector<2xf16>
    %bits = llvm.bitcast %pack1 : vector<2xf16> to i32
    rocdl.raw.ptr.buffer.store %bits, %rsrc, %offset, %soffset, %aux : i32
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @loads_to_llvm_stores
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: } : i32, i32 -> i32, i32
// CHECK: llvm.store %[[REGION]]#0,
// CHECK: llvm.store %[[REGION]]#1,
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @loads_to_llvm_stores(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %active, %one : (!llvm.ptr, i1, i32) -> i32
    llvm.store %a, %dst0 : i32, !llvm.ptr
    llvm.store %b, %dst1 : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @loads_to_later_masked_stores
// CHECK-SAME: (%[[LOAD_MASK:.*]]: i1, %[[STORE_MASK:.*]]: i1
// CHECK: %[[LOADS:.*]]:2 = amdg.masked_region %[[LOAD_MASK]] else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : i32, i32
// CHECK: } : i32, i32 -> i32, i32
// CHECK: amdg.masked_region %[[STORE_MASK]] {
// CHECK:   llvm.store %[[LOADS]]#0,
// CHECK:   llvm.store %[[LOADS]]#1,
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @loads_to_later_masked_stores(%load_mask: i1, %store_mask: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %a = amdg.masked_load %src0, %load_mask, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %load_mask, %one : (!llvm.ptr, i1, i32) -> i32
    amdg.masked_store %dst0, %a, %store_mask : !llvm.ptr, i32, i1
    amdg.masked_store %dst1, %b, %store_mask : !llvm.ptr, i32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @loads_cast_to_stores
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> i16
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> f16
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : i16, f16
// CHECK: } : i16, f16 -> i16, f16
// CHECK: %[[ZEXT:.*]] = llvm.zext %[[REGION]]#0 : i16 to i32
// CHECK: %[[FPEXT:.*]] = llvm.fpext %[[REGION]]#1 : f16 to f32
// CHECK: llvm.store %[[ZEXT]],
// CHECK: llvm.store %[[FPEXT]],
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @loads_cast_to_stores(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr, %zero_i16: i16, %zero_f16: f16) {
    %a = amdg.masked_load %src0, %active, %zero_i16 : (!llvm.ptr, i1, i16) -> i16
    %b = amdg.masked_load %src1, %active, %zero_f16 : (!llvm.ptr, i1, f16) -> f16
    %zext = llvm.zext %a : i16 to i32
    %fpext = llvm.fpext %b : f16 to f32
    llvm.store %zext, %dst0 : i32, !llvm.ptr
    llvm.store %fpext, %dst1 : f32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @loads_packed_with_insertvalue
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32)>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> !llvm.struct<(i32)>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : !llvm.struct<(i32)>, !llvm.struct<(i32)>
// CHECK: } : !llvm.struct<(i32)>, !llvm.struct<(i32)> -> !llvm.struct<(i32)>, !llvm.struct<(i32)>
// CHECK: %[[E0:.*]] = llvm.extractvalue %[[REGION]]#0[0]
// CHECK: %[[E1:.*]] = llvm.extractvalue %[[REGION]]#1[0]
// CHECK: %[[PACK0:.*]] = llvm.insertvalue %[[E0]],
// CHECK: %[[PACK1:.*]] = llvm.insertvalue %[[E1]], %[[PACK0]]
// CHECK: llvm.store %[[PACK1]],
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @loads_packed_with_insertvalue(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero: !llvm.struct<(i32)>) {
    %v0 = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, !llvm.struct<(i32)>) -> !llvm.struct<(i32)>
    %v1 = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, !llvm.struct<(i32)>) -> !llvm.struct<(i32)>
    %e0 = llvm.extractvalue %v0[0] : !llvm.struct<(i32)>
    %e1 = llvm.extractvalue %v1[0] : !llvm.struct<(i32)>
    %undef = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %pack0 = llvm.insertvalue %e0, %undef[0] : !llvm.struct<(i32, i32)>
    %pack1 = llvm.insertvalue %e1, %pack0[1] : !llvm.struct<(i32, i32)>
    llvm.store %pack1, %dst : !llvm.struct<(i32, i32)>, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @loads_shuffled_to_store
// CHECK: %[[REGION:.*]]:2 = amdg.masked_region %{{.*}} else(%{{.*}}, %{{.*}}) {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<2xi32>
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<2xi32>
// CHECK:   amdg.masked_yield %{{.*}}, %{{.*}} : vector<2xi32>, vector<2xi32>
// CHECK: } : vector<2xi32>, vector<2xi32> -> vector<2xi32>, vector<2xi32>
// CHECK: %[[SHUFFLE:.*]] = llvm.shufflevector %[[REGION]]#0, %[[REGION]]#1 [0, 2] : vector<2xi32>
// CHECK: llvm.store %[[SHUFFLE]],
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @loads_shuffled_to_store(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<2xi32>) {
    %v0 = amdg.masked_load %src0, %active, %zero_vec : (!llvm.ptr, i1, vector<2xi32>) -> vector<2xi32>
    %v1 = amdg.masked_load %src1, %active, %zero_vec : (!llvm.ptr, i1, vector<2xi32>) -> vector<2xi32>
    %shuffle = llvm.shufflevector %v0, %v1 [0, 2] : vector<2xi32>
    llvm.store %shuffle, %dst : vector<2xi32>, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @load_as_extract_index_blocks_region
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: llvm.extractelement
// CHECK: rocdl.raw.ptr.buffer.store
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @load_as_extract_index_blocks_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %rsrc: !llvm.ptr<8>, %offset: i32, %soffset: i32, %aux: i32, %vec: vector<4xf32>) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %idx = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %unused = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %value = llvm.extractelement %vec[%idx : i32] : vector<4xf32>
    rocdl.raw.ptr.buffer.store %value, %rsrc, %offset, %soffset, %aux : f32
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @load_as_buffer_store_offset_blocks_region
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: rocdl.raw.ptr.buffer.store
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @load_as_buffer_store_offset_blocks_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %rsrc: !llvm.ptr<8>, %soffset: i32, %aux: i32, %value: i32) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %offset = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %unused = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    rocdl.raw.ptr.buffer.store %value, %rsrc, %offset, %soffset, %aux : i32
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
// CHECK-NOT: amdg.masked_load
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

// CHECK-LABEL: llvm.func @pure_op_stays_outside_store_region
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK-SAME: %[[X:[A-Za-z0-9_]+]]: i32, %[[Y:[A-Za-z0-9_]+]]: i32
// CHECK: amdg.masked_region %[[ACTIVE]] {
// CHECK:   %[[A:.*]] = llvm.load
// CHECK:   %[[B:.*]] = llvm.load
// CHECK:   %[[SUM:.*]] = llvm.add %[[A]], %[[B]] : i32
// CHECK:   llvm.store %[[SUM]],
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK: %[[TMP:.*]] = llvm.add %[[X]], %[[Y]] : i32
// CHECK: llvm.store %[[TMP]],
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @pure_op_stays_outside_store_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %side: !llvm.ptr, %x: i32, %y: i32) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %tmp = llvm.add %x, %y : i32
    %b = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %sum = llvm.add %a, %b : i32
    amdg.masked_store %dst, %sum, %active : !llvm.ptr, i32, i1
    llvm.store %tmp, %side : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @masked_store_does_not_hide_load_escape
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: amdg.masked_load
// CHECK: amdg.masked_store
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @masked_store_does_not_hide_load_escape(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr, %value: i32) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_load %src0, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    %b = amdg.masked_load %src1, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    amdg.masked_store %dst0, %value, %active : !llvm.ptr, i32, i1
    %sum = llvm.add %a, %b : i32
    llvm.store %sum, %dst1 : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @extracted_loads_to_arithmetic_block_region
// CHECK-NOT: amdg.masked_region
// CHECK: amdg.masked_load
// CHECK: llvm.extractelement
// CHECK: amdg.masked_load
// CHECK: llvm.extractelement
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @extracted_loads_to_arithmetic_block_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<4xf32>) {
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

// CHECK-LABEL: llvm.func @interleaved_false_values_store_region
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: amdg.masked_region %[[ACTIVE]] {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.fadd
// CHECK:   llvm.store
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @interleaved_false_values_store_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %other0: f32, %other1: f32) {
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
    amdg.masked_store %dst, %sum, %active : !llvm.ptr, f32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @aggregate_masks_from_same_value_group
// CHECK: amdg.masked_region %{{.*}} {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<4xf32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.fadd
// CHECK:   llvm.store
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
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
    amdg.masked_store %dst, %sum, %m1 : !llvm.ptr, f32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @materialized_aggregate_mask_store_region
// CHECK: amdg.masked_region %{{.*}} {
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xi32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.load %{{.*}} : !llvm.ptr -> vector<1xi32>
// CHECK:   llvm.extractelement
// CHECK:   llvm.add
// CHECK:   llvm.store
// CHECK:   amdg.masked_yield
// CHECK: }
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @materialized_aggregate_mask_store_region(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr, %zero_vec: vector<1xi32>) {
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
    amdg.masked_store %dst, %sum, %m1 : !llvm.ptr, i32, i1
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
