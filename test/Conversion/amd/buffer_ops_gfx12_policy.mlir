// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1250 | FileCheck %s --check-prefixes=CHECK,CU,CDNA5
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm="gfx-arch=gfx1201 cu-mode=true" | FileCheck %s --check-prefixes=CHECK,CU,RDNA4
// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1201 | FileCheck %s --check-prefixes=CHECK,WGP,RDNA4

// A buffer descriptor packs its base and num_records into the resource;
// soffset feeds the instruction's scalar offset, scaled from elements to bytes.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_desc_soffset
  tt.func @buffer_desc_soffset(%arg0: !tt.ptr<f32>, %offsets: tensor<32xi32, #blocked>, %s: i32, %n: i32) {
    // CHECK: %[[STRIDE:.*]] = llvm.mlir.constant(0 : i16) : i16
    // CHECK: %[[NUM:.*]] = llvm.zext %{{.*}} : i32 to i64
    // CHECK: %[[RSRC:.*]] = rocdl.make.buffer.rsrc %{{.*}}, %[[STRIDE]], %[[NUM]], %{{.*}}
    %desc = amdg.make_buffer_desc %arg0, %n : !tt.ptr<f32> -> !amdg.buffer_desc<f32>
    // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: %[[SOFF:.*]] = llvm.mul %[[C4]], %{{.*}} : i32
    // CHECK: rocdl.raw.ptr.buffer.load %[[RSRC]], %{{.*}}, %[[SOFF]], 0 : f32
    %0 = amdg.buffer_load %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %[[RSRC]], %{{.*}}, %{{.*}}, 0 : f32
    amdg.buffer_store %0, %desc[%offsets] soffset = %s : i32 : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}

// -----

// scope selects SCOPE and the gfx12 policy selects TH, independently. The cta
// scope is SCOPE_CU, except in RDNA4's WGP mode, where it is SCOPE_SE.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_scope_policy
  tt.func @buffer_scope_policy(%arg0: !tt.ptr<f32>, %offsets: tensor<32xi32, #blocked>) {
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 1 : f32
    %0 = amdg.buffer_load %arg0[%offsets] {cachePolicy = #amdg.cache_policy_gfx12<temporal = none>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CU: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 0 : f32
    // WGP: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 8 : f32
    %1 = amdg.buffer_load %arg0[%offsets] {scope = 2 : i32} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %{{.*}}, %{{.*}}, 19 : f32
    %2 = amdg.buffer_load %arg0[%offsets] {scope = 1 : i32, cachePolicy = #amdg.cache_policy_gfx12<temporal = last_use>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, 26 : f32
    amdg.buffer_store %0, %arg0[%offsets] {scope = 3 : i32, cachePolicy = #amdg.cache_policy_gfx12<temporal = high>} : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_pointer_mask
  tt.func @buffer_pointer_mask(%base: !tt.ptr<f32>, %offsets: tensor<32xi32, #blocked>, %mask: tensor<32xi1, #blocked>) {
    // CHECK: %[[POOB:.*]] = llvm.mlir.constant(-2147483648 : i32) : i32
    // CHECK: %[[POFF:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[POOB]] : i1, i32
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %[[POFF]], %{{.*}}, 0 : f32
    %x = amdg.buffer_load %base[%offsets], %mask : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    amdg.buffer_store %x, %base[%offsets], %mask : !tt.ptr<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}

// -----

// The descriptor can be a block/function argument. Choose the sentinel from
// its current extent without tracing its producer or changing that extent.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: buffer_descriptor_mask
  tt.func @buffer_descriptor_mask(%desc: !amdg.buffer_desc<f32>, %offsets: tensor<32xi32, #blocked>, %mask: tensor<32xi1, #blocked>) {
    // CHECK: %[[BITS:.*]] = llvm.ptrtoint %{{.*}} : !llvm.ptr<8> to i128
    // CDNA5: %[[SHIFT:.*]] = llvm.mlir.constant(57 : i128) : i128
    // RDNA4: %[[SHIFT:.*]] = llvm.mlir.constant(64 : i128) : i128
    // CHECK: %[[HIGH:.*]] = llvm.lshr %[[BITS]], %[[SHIFT]] : i128
    // CHECK: %[[BOUND:.*]] = llvm.trunc %[[HIGH]] : i128 to i32
    // CHECK: %[[ALIGNED:.*]] = llvm.mlir.constant(-16 : i32) : i32
    // CHECK: %[[FITS:.*]] = llvm.icmp "ule" %[[BOUND]], %[[ALIGNED]] : i32
    // CHECK: %[[DOOB:.*]] = llvm.select %[[FITS]], %[[ALIGNED]], %{{.*}} : i1, i32
    // CHECK: %[[DOFF:.*]] = llvm.select %{{.*}}, %{{.*}}, %[[DOOB]] : i1, i32
    // CHECK: rocdl.raw.ptr.buffer.load %{{.*}}, %[[DOFF]], %{{.*}}, 0 : f32
    %x = amdg.buffer_load %desc[%offsets], %mask : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    amdg.buffer_store %x, %desc[%offsets], %mask : !amdg.buffer_desc<f32> -> tensor<32xf32, #blocked>
    tt.return
  }
}
