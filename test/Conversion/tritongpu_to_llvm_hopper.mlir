// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm 2>&1 | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 4], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 4], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_no_broadcast
  tt.func @tma_multicast_no_broadcast(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                        %dim0: i64, %dim1: i64,
                                        %stride0: i64, %stride1: i64,
                                        %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: nvgpu.tma_load_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} {operand_segment_sizes = array<i32: 1, 1, 1, 1, 1, 2, 0>} : !llvm.ptr<i8, 3>, !llvm.ptr<i64, 3>, !llvm.ptr<i8, 1>, i64, i1, i32, i32
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_const_mask
  tt.func @tma_multicast_const_mask(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                      %dim0: i64, %dim1: i64,
                                      %stride0: i64, %stride1: i64,
                                      %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: %[[C15:.*]] = llvm.mlir.constant(15 : i16) : i16
    // CHECK: nvgpu.tma_load_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %[[C15]]
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 2], CTAOrder = [0, 1]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 4], CTASplitNum = [1, 2], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 4 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @tma_multicast_variable_mask
  tt.func @tma_multicast_variable_mask(%basePtr: !tt.ptr<f16> {tt.divisibility = 8 : i32},
                                         %dim0: i64, %dim1: i64,
                                         %stride0: i64, %stride1: i64,
                                         %coord0: i32, %coord1: i32) {
    %mbar = triton_nvidia_gpu.alloc_mbarrier { count = 128 : i32 } : !tt.ptr<i64, 3>
    %dst = triton_gpu.alloc_tensor : tensor<1x64x64xf16, #shared>
    %c0 = arith.constant 0 : i32
    %src = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>
    // CHECK: nvgpu.cluster_id
    // CHECK: nvgpu.tma_load_tiled
    %res = triton_nvidia_gpu.insert_slice_async_v2 %src, %dst, %c0, %mbar {axis = 0 : i32, cache = 1 : i32, evict = 1 : i32, isVolatile = false, operand_segment_sizes = array<i32: 1, 1, 1, 1, 0, 0>} : !tt.ptr<tensor<64x64xf16, #blocked>, 1>, tensor<1x64x64xf16, #shared>, i32, !tt.ptr<i64, 3> -> tensor<1x64x64xf16, #shared>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: @tma_store
  tt.func @tma_store(%basePtr: !tt.ptr<f32> {tt.divisibility = 8 : i32},
                       %dim0: i64, %dim1: i64,
                       %stride0: i64, %stride1: i64,
                       %coord0: i32, %coord1: i32) {
    %src = triton_gpu.alloc_tensor : tensor<64x64xf32, #shared>
    %c0 = arith.constant 0 : i32
    %dst = tt.make_tensor_ptr %basePtr, [%dim0, %dim1], [%stride0, %stride1], [%coord0, %coord1] {order = array<i32: 1, 0>} : !tt.ptr<tensor<64x64xf32, #blocked>, 1>
    // CHECK: nvgpu.tma_store_tiled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : !llvm.ptr<i8, 1>, !llvm.ptr<i8, 3>, i1, i32, i32
    triton_nvidia_gpu.store_async %dst, %src {cache = 1 : i32} : !tt.ptr<tensor<64x64xf32, #blocked>, 1>, tensor<64x64xf32, #shared>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_high_precision_acc
  tt.func @dot_high_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 32 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_low_precision_acc
  tt.func @dot_low_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: llvm.return
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 129 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 256, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: @dot_mix_precision_acc
  tt.func @dot_mix_precision_acc(%a: tensor<128x128xf8E5M2, #shared>, %b: tensor<128x256xf8E5M2, #shared1>, %c: tensor<128x256xf32, #mma>) {
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-NOT: llvm.fadd
    // CHECK: nvgpu.wgmma
    // CHECK-COUNT-128: llvm.fadd
    // CHECK: llvm.return
    %m = triton_nvidia_gpu.dot_async %a, %b, %c
      {maxNumImpreciseAcc = 64 : i32, allowTF32 = true} :
      tensor<128x128xf8E5M2, #shared> * tensor<128x256xf8E5M2, #shared1> -> tensor<128x256xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_zero_acc
  // Generate a wgmma with 2 sources.
  // CHECK: nvgpu.wgmma %{{.*}}, %{{.*}} {
  tt.func @dot_zero_acc(%a: tensor<128x64xf16, #shared>, %b: tensor<64x64xf16, #shared1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %m = triton_nvidia_gpu.dot_async %a, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<128x64xf16, #shared> * tensor<64x64xf16, #shared1> -> tensor<128x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #triton_gpu.mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_reg_operand_A
  // Generate a wgmma where the first operand is a struct.
  // CHECK: nvgpu.wgmma {{.*}} : (!llvm.struct<(i32, i32, i32, i32)>, i64, !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>) -> !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)>
  tt.func @dot_reg_operand_A(%a: tensor<128x64xf16, #mma>, %b: tensor<64x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %opA = triton_gpu.convert_layout %a : (tensor<128x64xf16, #mma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
    %m = tt.dot %opA, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} :
      tensor<128x64xf16,  #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<64x64xf16, #shared> -> tensor<128x64xf32, #mma>
    tt.return
  }
}
