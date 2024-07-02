// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32_scalar
  tt.func @atomic_add_f32_scalar(%arg0 : !tt.ptr<f32>, %arg1 : i1, %arg2 : f32) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.br
    // CHECK: rocdl.barrier
    // CHECK: llvm.load
    // CHECK: rocdl.barrier
    // CHECK: rocdl.raw.ptr.buffer.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %0 : !tt.ptr<f32>
    tt.return
  }
}

// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: atomic_add_f32
  tt.func @atomic_add_f32(%arg0 : tensor<256x!tt.ptr<f32>, #blocked0>, %arg1 : tensor<256xi1, #blocked0>, %arg2 : tensor<256xf32, #blocked0>) {
    // CHECK: llvm.cond_br
    // CHECK: llvm.atomicrmw
    // CHECK: llvm.atomicrmw
    // CHECK: %[[ADDR1:.*]] = llvm.extractvalue
    // CHECK: %[[ADDR2:.*]] = llvm.extractvalue
    // CHECK: rocdl.raw.ptr.buffer.store
    // CHECK: rocdl.raw.ptr.buffer.store
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg0, %arg2, %arg1 : (tensor<256x!tt.ptr<f32>, #blocked0>, tensor<256xf32, #blocked0>, tensor<256xi1, #blocked0>) -> tensor<256xf32, #blocked0>
    tt.store %arg0, %0 : tensor<256x!tt.ptr<f32>, #blocked0>
    tt.return
  }
}

// -----

#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: vector_add_with_buffer_ops
  tt.func @vector_add_with_buffer_ops(%arg0 : tensor<256x!tt.ptr<f32>, #blocked1>, %arg1 : tensor<256xi1, #blocked1>) {
    // CHECK: %[[outOfBound:.*]] = llvm.mlir.constant(-2147483648 : i32) : i32
    // CHECK: %[[ptr0:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK: %[[ptr1:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK: %[[pred0:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1)>
    // CHECK: %[[pred1:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1)>
    // CHECK: %[[bufferSrc0:.*]] = rocdl.make.buffer.rsrc %[[ptr0]]
    // CHECK: %[[select0:.*]] = llvm.select %[[pred0]]
    // CHECK: %[[data0:.*]] = rocdl.raw.ptr.buffer.load %[[bufferSrc0]], %[[select0]]

    // CHECK: %[[bufferSrc1:.*]] = rocdl.make.buffer.rsrc %[[ptr1]]
    // CHECK: %[[select1:.*]] = llvm.select %[[pred1]]
    // CHECK: %[[data1:.*]] = rocdl.raw.ptr.buffer.load %[[bufferSrc1]], %[[select1]]
    %a = tt.load %arg0, %arg1 : tensor<256x!tt.ptr<f32>, #blocked1>
    // CHECK: %[[add0:.*]] = llvm.fadd
    // CHECK: %[[add1:.*]] = llvm.fadd
    %b = arith.addf %a, %a : tensor<256xf32, #blocked1>
    // CHECK: %[[ptr2:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK: %[[ptr3:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(ptr<1>, ptr<1>)>
    // CHECK: %[[pred2:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(i1, i1)>
    // CHECK: %[[pred3:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(i1, i1)>
    // CHECK: %[[pred4:.*]] = llvm.and {{.*}}, %[[pred2]] : i1
    // CHECK: %[[bufferSrc2:.*]] = rocdl.make.buffer.rsrc %[[ptr2]]
    // CHECK: %[[select2:.*]] = llvm.select %[[pred4]]
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %[[select2]]
    // CHECK: %[[pred5:.*]] = llvm.and {{.*}}, %[[pred3]] : i1
    // CHECK: %[[bufferSrc3:.*]] = rocdl.make.buffer.rsrc %[[ptr3]]
    // CHECK: %[[select3:.*]] = llvm.select %[[pred5]]
    // CHECK: rocdl.raw.ptr.buffer.store %{{.*}}, %{{.*}}, %[[select3]]
    tt.store %arg0, %b, %arg1 : tensor<256x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
