// RUN: triton-opt %s -split-input-file --triton-amdgpu-masked-ops-to-llvm | FileCheck %s
// RUN: triton-opt %s -split-input-file --triton-amdgpu-masked-ops-to-llvm='gfx-arch=gfx1250' | FileCheck %s --check-prefix=GFX1250

// CHECK-LABEL: llvm.func @region_load_load_store
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: llvm.cond_br %[[ACTIVE]], ^[[TRUE:.*]], ^[[JOIN:.*]](%{{.*}}, %{{.*}} : i32, i32)
// CHECK: ^[[TRUE]]:
// CHECK:   %[[A:.*]] = llvm.load
// CHECK:   %[[B:.*]] = llvm.load
// CHECK:   %[[SUM:.*]] = llvm.add %[[A]], %[[B]] : i32
// CHECK:   llvm.store %[[SUM]],
// CHECK:   llvm.br ^[[JOIN]](%[[A]], %[[B]] : i32, i32)
// CHECK: ^[[JOIN]](
// CHECK-NOT: llvm.cond_br %[[ACTIVE]]
// CHECK-NOT: amdg.masked_region
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @region_load_load_store(%active: i1, %src0: !llvm.ptr, %src1: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a:2 = amdg.masked_region %active else (%zero, %zero) {
      %v0 = llvm.load %src0 : !llvm.ptr -> i32
      %v1 = llvm.load %src1 : !llvm.ptr -> i32
      %sum = llvm.add %v0, %v1 : i32
      llvm.store %sum, %dst : i32, !llvm.ptr
      amdg.masked_yield %v0, %v1 : i32, i32
    } : i32, i32 -> i32, i32
    %sum = llvm.add %a#0, %a#1 : i32
    llvm.store %sum, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @store_only_region
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: llvm.cond_br %[[ACTIVE]], ^[[TRUE:.*]], ^[[JOIN:.*]]
// CHECK: ^[[TRUE]]:
// CHECK:   llvm.store
// CHECK:   llvm.store
// CHECK:   llvm.br ^[[JOIN]]
// CHECK: ^[[JOIN]]:
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @store_only_region(%active: i1, %src: !llvm.ptr, %dst0: !llvm.ptr, %dst1: !llvm.ptr) {
    %v = llvm.load %src : !llvm.ptr -> i32
    amdg.masked_region %active {
      llvm.store %v, %dst0 : i32, !llvm.ptr
      llvm.store %v, %dst1 : i32, !llvm.ptr
      amdg.masked_yield
    }
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @volatile_region
// CHECK-SAME: (%[[ACTIVE:.*]]: i1
// CHECK: llvm.cond_br %[[ACTIVE]], ^[[TRUE:.*]], ^[[JOIN:.*]](%{{.*}} : i32)
// CHECK: ^[[TRUE]]:
// CHECK:   %[[V:.*]] = llvm.load volatile
// CHECK:   llvm.store volatile %[[V]],
// CHECK:   llvm.br ^[[JOIN]](%[[V]] : i32)
// CHECK: ^[[JOIN]](
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @volatile_region(%active: i1, %src: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_region %active else (%zero) {
      %v = llvm.load volatile %src : !llvm.ptr -> i32
      llvm.store volatile %v, %dst : i32, !llvm.ptr
      amdg.masked_yield %v : i32
    } : i32 -> i32
    llvm.store %a, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @constant_true_region
// CHECK-NOT: llvm.cond_br
// CHECK: llvm.load
// CHECK-NOT: amdg.masked_region
// CHECK: llvm.return
module {
  llvm.func @constant_true_region(%src: !llvm.ptr, %dst: !llvm.ptr) {
    %true = llvm.mlir.constant(true) : i1
    %zero = llvm.mlir.constant(0 : i32) : i32
    %a = amdg.masked_region %true else (%zero) {
      %v = llvm.load %src : !llvm.ptr -> i32
      amdg.masked_yield %v : i32
    } : i32 -> i32
    llvm.store %a, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @leftover_masked_ops
// CHECK: llvm.cond_br %{{.*}}, ^{{.*}}, ^{{.*}}(%{{.*}} : i32)
// CHECK: llvm.load
// CHECK: llvm.cond_br %{{.*}}, ^{{.*}}, ^{{.*}}
// CHECK: llvm.store
// CHECK-NOT: amdg.masked_load
// CHECK-NOT: amdg.masked_store
// CHECK: llvm.return
module {
  llvm.func @leftover_masked_ops(%active: i1, %src: !llvm.ptr, %dst: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %value = amdg.masked_load %src, %active, %zero : (!llvm.ptr, i1, i32) -> i32
    amdg.masked_store %dst, %value, %active : !llvm.ptr, i32, i1
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @regular_noalias_load
// CHECK: llvm.load %{{.*}} {alias_scopes = {{.*}}noalias_scopes = {{.*}}} : !llvm.ptr -> i32
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
module {
  llvm.func @regular_noalias_load(%src: !llvm.ptr, %dst: !llvm.ptr) {
    %true = llvm.mlir.constant(true) : i1
    %zero = llvm.mlir.constant(0 : i32) : i32
    %value = amdg.masked_load %src, %true, %zero forceNoAlias true : (!llvm.ptr, i1, i32) -> i32
    llvm.store %value, %dst : i32, !llvm.ptr
    llvm.return
  }
}

// -----

// CHECK-LABEL: llvm.func @multicast_lowering
// CHECK: llvm.load
// CHECK-NOT: llvm.call_intrinsic "llvm.amdgcn.cluster.load.b32"
// CHECK-NOT: amdg.masked_load
// CHECK: llvm.return
// GFX1250-LABEL: llvm.func @multicast_lowering
// GFX1250: llvm.call_intrinsic "llvm.amdgcn.cluster.load.b32"
// GFX1250-NOT: amdg.masked_load
// GFX1250: llvm.return
module {
  llvm.func @multicast_lowering(%src: !llvm.ptr<1>, %dst: !llvm.ptr<1>) {
    %true = llvm.mlir.constant(true) : i1
    %zero = llvm.mlir.constant(0 : i32) : i32
    %multicast = llvm.mlir.constant(3 : i32) : i32
    %value = amdg.masked_load %src, %true, %zero, %multicast cacheModifier = ca : (!llvm.ptr<1>, i1, i32, i32) -> i32
    llvm.store %value, %dst : i32, !llvm.ptr<1>
    llvm.return
  }
}
