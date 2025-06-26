// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix | FileCheck %s

// CHECK: %pid{{(_[0-9]*)?}} = tt.get_program_id x
// CHECK-NEXT: %block_start{{(_[0-9]*)?}} = arith.muli %pid{{(_[0-9]*)?}}
// CHECK: %offsets{{(_[0-9])?}} = tt.splat %block_start{{(_\d*)?}}
// CHECK-NEXT: %offsets{{(_[0-9])?}} = arith.addi %offsets{{(_[0-9]*)?}}
// CHECK: %mask{{(_[0-9]*)?}} = tt.splat %arg3
// CHECK-NEXT: %mask{{(_[0-9]*)?}} = arith.cmpi slt, %offsets{{(_[0-9]*)?}}, %mask{{(_[0-9]*)?}}
// CHECK: %x{{(_[0-9]*)?}} = amdgpu.buffer_load %{{.*}}, %mask{{(_[0-9]*)?}}
// CHECK: %y{{(_[0-9]*)?}} = amdgpu.buffer_load %{{.*}}, %mask{{(_[0-9]*)?}}
// CHECK: %output{{(_[0-9]*)?}} = arith.addf %x{{(_[0-9]*)?}}, %y{{(_[0-9]*)?}}

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#loc = loc("vector-add.py":30:0)
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("vector-add.py":30:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %name1 = tt.get_program_id x : i32 loc(#loc14)
    %name2 = arith.muli %name1, %c1024_i32 : i32 loc(#loc15)
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked> loc(#loc4)
    %name3 = tt.splat %name2 : i32 -> tensor<1024xi32, #blocked> loc(#loc16)
    %name4 = arith.addi %name3, %0 : tensor<1024xi32, #blocked> loc(#loc16)
    %name5 = tt.splat %arg3 : i32 -> tensor<1024xi32, #blocked> loc(#loc17)
    %name6 = arith.cmpi slt, %name4, %name5 : tensor<1024xi32, #blocked> loc(#loc17)
    %1 = tt.addptr %arg0, %name2 : !tt.ptr<f32>, i32 loc(#loc7)
    %name7 = amdgpu.buffer_load %1[%0], %name6 : tensor<1024xf32, #blocked> loc(#loc18)
    %2 = tt.addptr %arg1, %name2 : !tt.ptr<f32>, i32 loc(#loc9)
    %name8 = amdgpu.buffer_load %2[%0], %name6 : tensor<1024xf32, #blocked> loc(#loc19)
    %name9 = arith.addf %name7, %name8 : tensor<1024xf32, #blocked> loc(#loc20)
    %3 = tt.addptr %arg2, %name2 : !tt.ptr<f32>, i32 loc(#loc12)
    amdgpu.buffer_store %name9, %3[%0], %name6 : tensor<1024xf32, #blocked> loc(#loc13)
    tt.return loc(#loc1)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("vector-add.py":39:10)
#loc3 = loc("vector-add.py":44:18)
#loc4 = loc("vector-add.py":45:28)
#loc5 = loc("vector-add.py":45:14)
#loc6 = loc("vector-add.py":47:11)
#loc7 = loc("vector-add.py":50:16)
#loc8 = loc("vector-add.py":50:8)
#loc9 = loc("vector-add.py":51:16)
#loc10 = loc("vector-add.py":51:8)
#loc11 = loc("vector-add.py":52:13)
#loc12 = loc("vector-add.py":54:13)
#loc13 = loc("vector-add.py":54:4)
// CHECK: #loc14 = loc("pid"(#loc2))
#loc14 = loc("pid"(#loc2))
// CHECK-NEXT: #loc15 = loc("block_start"(#loc3))
#loc15 = loc("block_start"(#loc3))
// CHECK-NEXT: #loc16 = loc("offsets"(#loc5))
#loc16 = loc("offsets"(#loc5))
// CHECK-NEXT: #loc17 = loc("mask"(#loc6))
#loc17 = loc("mask"(#loc6))
// CHECK-NEXT: #loc18 = loc("x"(#loc8))
#loc18 = loc("x"(#loc8))
// CHECK-NEXT: #loc19 = loc("y"(#loc10))
#loc19 = loc("y"(#loc10))
// CHECK-NEXT: #loc20 = loc("output"(#loc11))
#loc20 = loc("output"(#loc11))
