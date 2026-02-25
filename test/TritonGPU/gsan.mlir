// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-global-sanitizer | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func @instrumented
  tt.func @instrumented(%ptrs: tensor<128x!tt.ptr<f32>, #blocked>,
                        %mask: tensor<128xi1, #blocked>,
                        %other: tensor<128xf32, #blocked>,
                        %vals: tensor<128xf32, #blocked>) {
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, 4, false, %{{.*}}
    // CHECK-NEXT: %[[LD:.*]] = tt.load
    %0 = tt.load %ptrs, %mask, %other : tensor<128x!tt.ptr<f32>, #blocked>
    // CHECK: tti.experimental_gsan_tensor_access %{{.*}}, 4, true, %{{.*}}
    // CHECK-NEXT: tt.store
    tt.store %ptrs, %vals, %mask : tensor<128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
