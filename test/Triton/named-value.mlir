// RUN: triton-opt %s -o - --mlir-print-debuginfo --mlir-use-nameloc-as-prefix | FileCheck %s
// While we only check whether the var name is preserved, it should generate exactly the same file

// CHECK: %pid{{(_[0-9]*)?}} = tt.get_program_id x
// CHECK-NEXT: %block_start{{(_[0-9]*)?}} = arith.muli %pid{{(_[0-9]*)?}}
// CHECK: %offsets{{(_[0-9])?}} = tt.splat %block_start{{(_\d*)?}}
// CHECK-NEXT: %offsets{{(_[0-9])?}} = arith.addi %offsets{{(_[0-9]*)?}}
// CHECK: %mask{{(_[0-9]*)?}} = tt.splat %arg3
// CHECK-NEXT: %mask{{(_[0-9]*)?}} = arith.cmpi slt, %offsets{{(_[0-9]*)?}}, %mask{{(_[0-9]*)?}}
// CHECK: %x{{(_[0-9]*)?}} = tt.load %{{.*}}, %mask{{(_[0-9]*)?}}
// CHECK: %y{{(_[0-9]*)?}} = tt.load %{{.*}}, %mask{{(_[0-9]*)?}}
// CHECK: %output{{(_[0-9]*)?}} = arith.addf %x{{(_[0-9]*)?}}, %y{{(_[0-9]*)?}}

#loc = loc("vector-add.py":30:0)
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32} loc("vector-add.py":30:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("vector-add.py":30:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc14)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc15)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc16)
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc16)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc17)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc17)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc7)
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc18)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc9)
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc19)
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc20)
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
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
