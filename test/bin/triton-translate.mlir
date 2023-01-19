// RUN: triton-translate %s | FileCheck %s
// RUN: triton-translate --num-warps=4 %s | FileCheck %s --check-prefix=NW4
// RUN: triton-translate --target=ptx %s | FileCheck %s --check-prefix=PTX

func @f(%arg: !tt.ptr<i32>) {
  %barg = tt.splat %arg : (!tt.ptr<i32>) -> tensor<256x!tt.ptr<i32>>
  %ofs = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
  %ptrs = tt.addptr %barg, %ofs : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
  tt.store %ptrs, %ofs : tensor<256xi32>
  return
}

// CHECK: asm sideeffect
// CHECK-SAME: st.global.b32

// If we have 4 warps, every thread should do 2 elements
// NW4: st.global.b32
// NW4: st.global.b32

// PTX-NOT: asm sideeffect
// PTX: st.global.b32
