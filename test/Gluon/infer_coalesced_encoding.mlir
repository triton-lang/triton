// RUN: triton-opt %s -split-input-file --gluon-infer-coalesced-encodings | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_efficient(%in_ptr : !tt.ptr<f32>, %out_ptr : !tt.ptr<f32>) {
    // CHECK: [[BLOCKED:#.+]] = #ttg.blocked
    // CHECK: %[[IN_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_IN:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: %[[VALUE:.+]] = tt.load %[[IN_PTRS]], %[[MASK_IN]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %mask = arith.constant dense<0> : tensor<128x256xi1, #gluon.auto_encoding>
    %in_ptrs_1 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs_2 = gluon.set_auto_layout %in_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_in = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    %value = tt.load %in_ptrs_2, %mask_in : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>

    // CHECK: %[[SIN:.+]] = math.sin %[[VALUE]] : tensor<128x256xf32, [[BLOCKED]]>
    // CHECK: %[[MAX:.+]] = arith.maxnumf %[[SIN]], {{.*}} : tensor<128x256xf32, [[BLOCKED]]>
    %value_2 = math.sin %value : tensor<128x256xf32, #gluon.coalesced_encoding>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #gluon.coalesced_encoding>
    %value_3 = arith.maxnumf %value_2, %cst : tensor<128x256xf32, #gluon.coalesced_encoding>

    // CHECK: %[[OUT_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_OUT:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: tt.store %[[OUT_PTRS]], %[[MAX]], %[[MASK_OUT]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %out_ptrs_1 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %out_ptrs_2 = gluon.set_auto_layout %out_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_out = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    tt.store %out_ptrs_2, %value_3, %mask_out : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    tt.return
  }
}



// -----


// CHECK: [[$ATOMIC:#.*]] = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @atomic_load_store_coalesced
  tt.func @atomic_load_store_coalesced(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) {
    %offsets = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #gluon.coalesced_encoding>
    %base = tt.splat %ptr : !tt.ptr<i8> -> tensor<2048x!tt.ptr<i8>, #gluon.coalesced_encoding>
    %ptrs = tt.addptr %base, %offsets : tensor<2048x!tt.ptr<i8>, #gluon.coalesced_encoding>, tensor<2048xi32, #gluon.coalesced_encoding>
    // CHECK: %[[LOADED:.*]] = tt.atomic_load acquire, gpu, %{{.*}} : (tensor<2048x!tt.ptr<i8>, [[$ATOMIC]]>) -> tensor<2048xi8, [[$ATOMIC]]>
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<2048x!tt.ptr<i8>, #gluon.coalesced_encoding>) -> tensor<2048xi8, #gluon.coalesced_encoding>
    // CHECK: tt.atomic_store release, gpu, %{{.*}}, %[[LOADED]] : tensor<2048x!tt.ptr<i8>, [[$ATOMIC]]>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<2048x!tt.ptr<i8>, #gluon.coalesced_encoding>
    tt.return
  }
}


// -----

// The mask restriction is available before any physical layout is assigned.
// CHECK: [[$MASKED_COALESCED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @masked_coalesced_odd_tail
  tt.func @masked_coalesced_odd_tail(%ptr: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %start: i32 {tt.divisibility = 1024 : i32}) {
    %range = tt.make_range {start = 0 : i32, end = 1024 : i32} : tensor<1024xi32, #gluon.coalesced_encoding>
    %base = tt.splat %ptr : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>, #gluon.coalesced_encoding>
    %ptrs = tt.addptr %base, %range : tensor<1024x!tt.ptr<bf16>, #gluon.coalesced_encoding>, tensor<1024xi32, #gluon.coalesced_encoding>
    %tile = tt.splat %start : i32 -> tensor<1024xi32, #gluon.coalesced_encoding>
    %offsets = arith.addi %tile, %range : tensor<1024xi32, #gluon.coalesced_encoding>
    %end = arith.constant dense<1023> : tensor<1024xi32, #gluon.coalesced_encoding>
    %mask = arith.cmpi slt, %offsets, %end : tensor<1024xi32, #gluon.coalesced_encoding>
    // CHECK: tt.load {{.*}} : tensor<1024x!tt.ptr<bf16>, [[$MASKED_COALESCED]]>
    %value = tt.load %ptrs, %mask : tensor<1024x!tt.ptr<bf16>, #gluon.coalesced_encoding>
    // CHECK: tt.store {{.*}} : tensor<1024x!tt.ptr<bf16>, [[$MASKED_COALESCED]]>
    tt.store %ptrs, %value, %mask : tensor<1024x!tt.ptr<bf16>, #gluon.coalesced_encoding>
    tt.return
  }
}
