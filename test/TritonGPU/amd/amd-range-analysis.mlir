// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -test-tritonamdgpu-range-analysis -verify-diagnostics=only-expected | FileCheck %s

// CHECK-LABEL:   tt.func @conversion1
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    // expected-remark@+2 {{unsigned : [1024, 1024] signed : [1024, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %numps = tt.get_num_programs x : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cmpule_programs = arith.cmpi ule, %numps, %c65536_i32 : i32
    llvm.intr.assume %cmpule_programs : i1
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @assumepid
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @assumepid(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c0 = arith.constant 0 : i32
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %pid = tt.get_program_id x : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmpsle = arith.cmpi sle, %pid, %c1024_i32 : i32
    llvm.intr.assume %cmpsle : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmpsge = arith.cmpi sge, %pid, %c0 : i32
    llvm.intr.assume %cmpsge : i1
    // expected-remark@+2 {{unsigned : [0, 1048576] signed : [0, 1048576]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %pid, %c1024_i32 : i32
    %2 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.load %3 : tensor<1024x!tt.ptr<f32>>
    tt.return %4 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion2
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion2(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion3
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion3(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 2048] signed : [0, 2048]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %6, %4 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @conversion4
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion4(%arg0: !tt.ptr<f32> {tt.pointer_range = 32 : i32}) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.addptr %3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 2048] signed : [0, 2048]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.addi %2, %2 : tensor<1024xi32>
    %6 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %7 = tt.addptr %6, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
    tt.return %8 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+3 {{result 1: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %5:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOp2
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+3 {{result 1: unsigned : [0, 130048] signed : [0, 130048]}}
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %10 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %11 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
      // expected-remark@+1 {{non-neg}}
      %12 = arith.addi %11, %arg4 : tensor<1024xi64>
      %13 = tt.splat %10 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %14 = tt.addptr %13, %12 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %15 = tt.load %14 : tensor<1024x!tt.ptr<f32>>
      %16 = arith.addf %15, %arg5 : tensor<1024xf32>
      scf.yield %10, %12, %16 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forNested
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forNested(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+3 {{result 1: unsigned : [0, 15360] signed : [0, 15360]}}
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 16}}
    %3:3 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      // expected-remark@+3 {{result 1: unsigned : [0, 261120] signed : [0, 261120]}}
      // expected-remark@+2 {{result 1: non-neg}}
      // expected-remark@+1 {{inferred total trip count: 256}}
      %10:3 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
        %11 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
        // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
        // expected-remark@+1 {{non-neg}}
        %12 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
        // expected-remark@+2 {{unsigned : [0, 262144] signed : [0, 262144]}}
        // expected-remark@+1 {{non-neg}}
        %13 = arith.addi %12, %arg8 : tensor<1024xi64>
        %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
        %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
        %17 = arith.addf %16, %arg9 : tensor<1024xf32>
        scf.yield %11, %13, %17 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
      }
      scf.yield %10#0, %10#1, %10#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 16384] signed : [0, 16384]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forNestedOverMaxTripCount
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forNestedOverMaxTripCount(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+2 {{result 1: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %3:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg0, %arg4 = %cst, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      // expected-remark@+2 {{result 1: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
      // expected-remark@+1 {{inferred total trip count: 16384}}
      %10:3 = scf.for %arg6 = %c0 to %c128 step %c1 iter_args(%arg7 = %arg3, %arg8 = %arg4, %arg9 = %arg5) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
        %11 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
        // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
        // expected-remark@+1 {{non-neg}}
        %12 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
        // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
        %13 = arith.addi %12, %arg8 : tensor<1024xi64>
        %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
        %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
        %17 = arith.addf %16, %arg9 : tensor<1024xf32>
        scf.yield %11, %13, %17 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
      }
      scf.yield %10#0, %10#1, %10#2 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %4 = tt.addptr %3#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %6 = arith.addi %5, %3#1 : tensor<1024xi64>
    %7 = tt.splat %4 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %9 = tt.load %8 : tensor<1024x!tt.ptr<f32>>
    tt.return %9 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @ifOp
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+2 {{result 1: unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{result 1: non-neg}}
    %3:2 = scf.if %arg2 -> (!tt.ptr<f32>, tensor<1024xi64>) {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      scf.yield %8, %9 : !tt.ptr<f32>, tensor<1024xi64>
    } else {
      %8 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
      scf.yield %8, %cst : !tt.ptr<f32>, tensor<1024xi64>
    }
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.trunci %3#1 : tensor<1024xi64> to tensor<1024xi32>
    %5 = tt.splat %3#0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    tt.return %7 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @condBranch
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 1: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @condBranch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    cf.cond_br %arg1, ^bb1(%arg0, %cst : !tt.ptr<f32>, tensor<1024xi64>), ^bb2(%3, %4 : !tt.ptr<f32>, tensor<1024xi64>)
  ^bb1(%5: !tt.ptr<f32>, %6: tensor<1024xi64>):  // pred: ^bb0
    // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  ^bb2(%11: !tt.ptr<f32>, %12: tensor<1024xi64>):  // pred: ^bb0
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %13 = arith.trunci %12 : tensor<1024xi64> to tensor<1024xi32>
    %14 = tt.splat %11 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %13 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %16 = tt.load %15 : tensor<1024x!tt.ptr<f32>>
    tt.return %16 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @branch
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 1: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @branch(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.addptr %4, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %6 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    tt.return %6 : tensor<1024xf32>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+2 {{arg 1: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  // expected-remark@+1 {{arg 2: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func @tile_offset(%arg0: !tt.ptr<f16>, %arg1: i32, %arg2: i32) -> tensor<16x256xf16, #blocked> {
    %c256_i32 = arith.constant 256 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 16776960] signed : [0, 16776960]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.expand_dims %3 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %5 = tt.splat %arg2 : i32 -> tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %6 = arith.muli %4, %5 : tensor<16x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %7 = tt.broadcast %6 : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %8 = tt.expand_dims %2 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.broadcast %8 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %10 = arith.addi %7, %9 : tensor<16x256xi32, #blocked>
    %11 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    %13 = tt.addptr %12, %10 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    %14 = tt.load %13 : tensor<16x256x!tt.ptr<f16>, #blocked>
    tt.return %14 : tensor<16x256xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 1: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16>, %arg1: i32) -> tensor<128x16xf16, #blocked> {
    %c128_i32 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 8388480] signed : [0, 8388480]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %4 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %5 = arith.muli %1, %arg1 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %6 = tt.splat %arg1 : i32 -> tensor<128x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %7 = arith.muli %4, %6 : tensor<128x1xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %9 = tt.expand_dims %3 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %10 = tt.broadcast %9 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %11 = arith.addi %8, %10 : tensor<128x16xi32, #blocked>
    %12 = tt.addptr %arg0, %5 : !tt.ptr<f16>, i32
    %13 = tt.splat %12 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %11 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %15 = tt.load %14 : tensor<128x16x!tt.ptr<f16>, #blocked>
    tt.return %15 : tensor<128x16xf16, #blocked>
  }
}

// -----

// CHECK-LABEL:   tt.func @select
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 1: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @select(%arg0: !tt.ptr<f32>, %arg1: i1) -> tensor<1024xf32> {
    %cst = arith.constant dense<0> : tensor<1024xi64>
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = arith.select %arg1, %arg0, %3 : !tt.ptr<f32>
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.select %arg1, %cst, %4 : tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.trunci %6 : tensor<1024xi64> to tensor<1024xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @where_kernel
module attributes {"ttg.num-ctas" = 1 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 255] signed : [-128, 127]}}
  tt.func @where_kernel(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>, %arg2: i8) -> tensor<1024xi64> {
    %c0_i8 = arith.constant 0 : i8
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %3 = arith.cmpi ne, %arg2, %c0_i8 : i8
    %4 = arith.select %3, %arg0, %arg1 : !tt.ptr<i64>
    %5 = tt.addptr %4, %1 : !tt.ptr<i64>, i32
    %6 = tt.splat %5 : !tt.ptr<i64> -> tensor<1024x!tt.ptr<i64>>
    %7 = tt.addptr %6, %2 : tensor<1024x!tt.ptr<i64>>, tensor<1024xi32>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %8 = tt.load %7 : tensor<1024x!tt.ptr<i64>>
    tt.return %8 : tensor<1024xi64>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOpWithHints
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOpWithHints(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.addptr %arg0, %0 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %3 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+3 {{result 1: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+2 {{result 1: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %4:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %2, %arg4 = %3, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      // expected-remark@+2 {{unsigned : [0, 131072] signed : [0, 131072]}}
      // expected-remark@+1 {{non-neg}}
      %11 = arith.trunci %arg4 : tensor<1024xi64> to tensor<1024xi32>
      %12 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %13 = tt.addptr %12, %11 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %14 = tt.load %13 : tensor<1024x!tt.ptr<f32>>
      %15 = tt.addptr %arg3, %0 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %16 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %17 = arith.addi %16, %arg4 : tensor<1024xi64>
      %18 = tt.addptr %15, %0 : !tt.ptr<f32>, i32
      %19 = arith.addf %14, %arg5 : tensor<1024xf32>
      scf.yield %18, %17, %19 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    } {tt.divisibility_arg1 = dense<16> : tensor<1xi32>, tt.divisibility_arg2 = dense<16> : tensor<1xi32>}
    %5 = tt.addptr %4#0, %0 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %1 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %6, %4#1 : tensor<1024xi64>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    tt.return %10 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func public @scalar_pointers
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func public @scalar_pointers(%arg0: !tt.ptr<i64>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i64>, i32
    // expected-remark@+1 {{inferred total trip count: 99}}
    %1 = scf.for %arg1 = %c1_i32 to %c100_i32 step %c1_i32 iter_args(%arg2 = %0) -> (!tt.ptr<i64>)  : i32 {
      tt.store %arg2, %c0_i64 : !tt.ptr<i64>
      %2 = tt.addptr %arg2, %c1_i32 : !tt.ptr<i64>, i32
      scf.yield %2 : !tt.ptr<i64>
    }
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_if
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @scalar_if(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %arg2: i1) -> f32 {
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = scf.if %arg2 -> (!tt.ptr<f32>) {
      %3 = tt.addptr %0, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    } else {
      %3 = tt.addptr %0, %c100_i32 : !tt.ptr<f32>, i32
      scf.yield %3 : !tt.ptr<f32>
    }
    %2 = tt.load %1 : !tt.ptr<f32>
    tt.return %2 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @scalar_cond_branch
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 1] signed : [-1, 0]}}
  tt.func @scalar_cond_branch(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i1) -> f32 {
    cf.cond_br %arg2, ^bb1(%arg0 : !tt.ptr<f32>), ^bb2(%arg1 : !tt.ptr<f32>)
  ^bb1(%0: !tt.ptr<f32>):  // pred: ^bb0
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.return %1 : f32
  ^bb2(%2: !tt.ptr<f32>):  // pred: ^bb0
    %3 = tt.load %2 : !tt.ptr<f32>
    tt.return %3 : f32
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpSimple
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpSimple(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+5 {{result 1: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+4 {{result 3: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+3 {{result 1: non-neg}}
    // expected-remark@+2 {{result 3: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %7:5 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %5, %arg4 = %6, %arg5 = %3, %arg6 = %4, %arg7 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %14 = tt.addptr %arg5, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %16 = arith.addi %15, %arg6 : tensor<1024xi64>
      %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
      %20 = arith.addf %19, %arg7 : tensor<1024xf32>
      scf.yield %14, %16, %arg3, %arg4, %20 : !tt.ptr<f32>, tensor<1024xi64>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    tt.return %13 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @flipFlopForOpComplex
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @flipFlopForOpComplex(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: tensor<1024xf32>) -> (tensor<1024xf32>, tensor<1024xf32>) {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+5 {{result 1: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+4 {{result 4: unsigned : [0, 131072] signed : [0, 131072]}}
    // expected-remark@+3 {{result 1: non-neg}}
    // expected-remark@+2 {{result 4: non-neg}}
    // expected-remark@+1 {{inferred total trip count: 128}}
    %7:6 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %3, %arg5 = %4, %arg6 = %arg2, %arg7 = %5, %arg8 = %6, %arg9 = %arg2) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %20 = tt.addptr %arg4, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %21 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %22 = arith.addi %21, %arg5 : tensor<1024xi64>
      %23 = tt.splat %20 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %24 = tt.addptr %23, %22 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %25 = tt.load %24 : tensor<1024x!tt.ptr<f32>>
      %26 = arith.addf %25, %arg6 : tensor<1024xf32>
      %27 = tt.addptr %arg7, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %28 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
      // expected-remark@+1 {{non-neg}}
      %29 = arith.addi %28, %arg8 : tensor<1024xi64>
      %30 = tt.splat %27 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %31 = tt.addptr %30, %29 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %32 = tt.load %31 : tensor<1024x!tt.ptr<f32>>
      %33 = arith.addf %32, %arg9 : tensor<1024xf32>
      scf.yield %27, %29, %33, %20, %22, %26 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>, !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %8 = tt.addptr %7#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %9 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %9, %7#1 : tensor<1024xi64>
    %11 = tt.splat %8 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %12 = tt.addptr %11, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %13 = tt.load %12 : tensor<1024x!tt.ptr<f32>>
    %14 = tt.addptr %7#3, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %15 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{unsigned : [0, 132096] signed : [0, 132096]}}
    // expected-remark@+1 {{non-neg}}
    %16 = arith.addi %15, %7#4 : tensor<1024xi64>
    %17 = tt.splat %14 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %19 = tt.load %18 : tensor<1024x!tt.ptr<f32>>
    tt.return %13, %19 : tensor<1024xf32>, tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @forOpDynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
  tt.func @forOpDynamicKBound(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>, %K: index) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid : i1
    // expected-remark@+2 {{unsigned : [0, 67107840] signed : [0, 67107840]}}
    // expected-remark@+1 {{non-neg}}
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+2 {{result 1: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    // expected-remark@+1 {{inferred total trip count: 1025}}
    %5:3 = scf.for %arg2 = %c0 to %c128 step %K iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
      // expected-remark@+1 {{non-neg}}
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    // expected-remark@+2 {{unsigned : [0, 1024] signed : [0, 1024]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}

// -----

// CHECK-LABEL:   tt.func @DynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
  tt.func @DynamicKBound(%K: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %c128 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp = arith.cmpi sle, %K, %c128 : i32
    llvm.intr.assume %cmp : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %condtest = arith.cmpi sle, %K, %c1024_i32 : i32
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @unsupportedAssumption
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [0, 128] signed : [0, 128]}}
  tt.func @unsupportedAssumption(%K: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %c128 = arith.constant 128 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp = arith.cmpi ule, %K, %c128 : i32
    llvm.intr.assume %cmp : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %condtest = arith.cmpi sle, %K, %c1024_i32 : i32
    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @moreDynamicKBound
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @moreDynamicKBound(
        // expected-remark@+1 {{arg 0: unsigned : [128, 128] signed : [128, 128]}}
        %Keqlhs: i32,
        // expected-remark@+1 {{arg 1: unsigned : [128, 2147483647] signed : [128, 2147483647]}}
        %Ksgelhs: i32,
        // expected-remark@+1 {{arg 2: unsigned : [129, 2147483647] signed : [129, 2147483647]}}
        %Ksgtlhs: i32,
        // expected-remark@+1 {{arg 3: unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
        %Kslelhs: i32,
        // expected-remark@+1 {{arg 4: unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
        %Ksltlhs: i32,
        // expected-remark@+1 {{arg 5: unsigned : [64, 64] signed : [64, 64]}}
        %Keqrhs: i32,
        // expected-remark@+1 {{arg 6: unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
        %Ksgerhs: i32,
        // expected-remark@+1 {{arg 7: unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
        %Ksgtrhs: i32,
        // expected-remark@+1 {{arg 8: unsigned : [128, 2147483647] signed : [128, 2147483647]}}
        %Kslerhs: i32,
        // expected-remark@+1 {{arg 9: unsigned : [129, 2147483647] signed : [129, 2147483647]}}
        %Ksltrhs: i32
    ) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1024_i32 = arith.constant 1024 : i32

    //// eq comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeeqlhs = arith.cmpi eq, %Keqlhs, %c128 : i32
    llvm.intr.assume %assumeeqlhs : i1
    // expected-remark@+2 {{unsigned : [128, 128] signed : [128, 128]}}
    // expected-remark@+1 {{non-neg}}
    %testeqlhs1 = arith.addi %Keqlhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testeqlhs2 = arith.cmpi ne, %Keqlhs, %c256 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeeqrhs = arith.cmpi eq, %c64, %Keqrhs : i32
    llvm.intr.assume %assumeeqrhs : i1
    // expected-remark@+2 {{unsigned : [64, 64] signed : [64, 64]}}
    // expected-remark@+1 {{non-neg}}
    %testeqrhs1 = arith.addi %Keqrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testeqrhs2 = arith.cmpi ne, %Keqrhs, %c256 : i32

    //// sge comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgelhs = arith.cmpi sge, %Ksgelhs, %c128 : i32
    llvm.intr.assume %assumesgelhs : i1
    // expected-remark@+2 {{unsigned : [128, 2147483647] signed : [128, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsgelhs1 = arith.addi %Ksgelhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testsgelhs2 = arith.cmpi sge, %Ksgelhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgerhs = arith.cmpi sge, %c128, %Ksgerhs  : i32
    llvm.intr.assume %assumesgerhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
    %testsgerhs1 = arith.addi %Ksgerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsgerhs2 = arith.cmpi sge, %c1024_i32, %Ksgerhs : i32

    //// sgt comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgtlhs = arith.cmpi sgt, %Ksgtlhs, %c128 : i32
    llvm.intr.assume %assumesgtlhs : i1
    // expected-remark@+2 {{unsigned : [129, 2147483647] signed : [129, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsgtlhs1 = arith.addi %Ksgtlhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testsgtlhs2 = arith.cmpi sgt, %Ksgtlhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesgtrhs = arith.cmpi sgt, %c128, %Ksgtrhs  : i32
    llvm.intr.assume %assumesgtrhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
    %testsgtrhs1 = arith.addi %Ksgtrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsgtrhs2 = arith.cmpi sgt, %c1024_i32, %Ksgtrhs : i32

    //// sle comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeslelhs = arith.cmpi sle, %Kslelhs, %c128 : i32
    llvm.intr.assume %assumeslelhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 128]}}
    %testslelhs1 = arith.addi %Kslelhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testslelhs2 = arith.cmpi sle, %Kslelhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeslerhs = arith.cmpi sle, %c128, %Kslerhs  : i32
    llvm.intr.assume %assumeslerhs : i1
    // expected-remark@+2 {{unsigned : [128, 2147483647] signed : [128, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testslerhs1 = arith.addi %Kslerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testslerhs2 = arith.cmpi sle, %c64, %Kslerhs : i32

    //// slt comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesltlhs = arith.cmpi slt, %Ksltlhs, %c128 : i32
    llvm.intr.assume %assumesltlhs : i1
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 127]}}
    %testsltlhs1 = arith.addi %Ksltlhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsltlhs2 = arith.cmpi slt, %Ksltlhs, %c1024_i32 : i32

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumesltrhs = arith.cmpi slt, %c128, %Ksltrhs  : i32
    llvm.intr.assume %assumesltrhs : i1
    // expected-remark@+2 {{unsigned : [129, 2147483647] signed : [129, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %testsltrhs1 = arith.addi %Ksltrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testsltrhs2 = arith.cmpi slt, %c64, %Ksltrhs : i32

    tt.return
  }
}

// -----

// CHECK-LABEL:   tt.func @moreDynamicKBoundUnsigned
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @moreDynamicKBoundUnsigned(
        // expected-remark@+1 {{arg 0: unsigned : [128, 4294967295] signed : [-2147483648, 2147483647]}}
        %Kugelhs: i32,
        // expected-remark@+1 {{arg 1: unsigned : [129, 4294967295] signed : [-2147483648, 2147483647]}}
        %Kugtlhs: i32,
        // expected-remark@+1 {{arg 2: unsigned : [0, 128] signed : [0, 128]}}
        %Kulelhs: i32,
        // expected-remark@+1 {{arg 3: unsigned : [0, 127] signed : [0, 127]}}
        %Kultlhs: i32,
        // expected-remark@+1 {{arg 4: unsigned : [0, 128] signed : [0, 128]}}
        %Kugerhs: i32,
        // expected-remark@+1 {{arg 5: unsigned : [0, 127] signed : [0, 127]}}
        %Kugtrhs: i32,
        // expected-remark@+1 {{arg 6: unsigned : [128, 4294967295] signed : [-2147483648, 2147483647]}}
        %Kulerhs: i32,
        // expected-remark@+1 {{arg 7: unsigned : [129, 4294967295] signed : [-2147483648, 2147483647]}}
        %Kultrhs: i32
    ) {
    %c0 = arith.constant 0 : i32
    %c16 = arith.constant 16 : i32
    %c32 = arith.constant 32 : i32
    %c64 = arith.constant 64 : i32
    %c128 = arith.constant 128 : i32
    %c256 = arith.constant 256 : i32
    %c1024_i32 = arith.constant 1024 : i32

    //// uge comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeugelhs = arith.cmpi uge, %Kugelhs, %c128 : i32
    llvm.intr.assume %assumeugelhs : i1
    // expected-remark@+1 {{unsigned : [128, 4294967295] signed : [-2147483648, 2147483647]}}
    %testugelhs1 = arith.addi %Kugelhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testugelhs2 = arith.cmpi uge, %Kugelhs, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeugerhs = arith.cmpi uge, %c128, %Kugerhs  : i32
    llvm.intr.assume %assumeugerhs : i1
    // expected-remark@+2 {{unsigned : [0, 128] signed : [0, 128]}}
    // expected-remark@+1 {{non-neg}}
    %testugerhs1 = arith.addi %Kugerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testugerhs2 = arith.cmpi uge, %c1024_i32, %Kugerhs : i32

    //// ugt comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeugtlhs = arith.cmpi ugt, %Kugtlhs, %c128 : i32
    llvm.intr.assume %assumeugtlhs : i1
    // expected-remark@+1 {{unsigned : [129, 4294967295] signed : [-2147483648, 2147483647]}}
    %testugtlhs1 = arith.addi %Kugtlhs, %c0 : i32
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %testugtlhs2 = arith.cmpi ugt, %Kugtlhs, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeugtrhs = arith.cmpi ugt, %c128, %Kugtrhs  : i32
    llvm.intr.assume %assumeugtrhs : i1
    // expected-remark@+2 {{unsigned : [0, 127] signed : [0, 127]}}
    // expected-remark@+1 {{non-neg}}
    %testugtrhs1 = arith.addi %Kugtrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testugtrhs2 = arith.cmpi ugt, %c1024_i32, %Kugtrhs : i32

    //// ule comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeulelhs = arith.cmpi ule, %Kulelhs, %c128 : i32
    llvm.intr.assume %assumeulelhs : i1
    // expected-remark@+2 {{unsigned : [0, 128] signed : [0, 128]}}
    // expected-remark@+1 {{non-neg}}
    %testulelhs1 = arith.addi %Kulelhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testulelhs2 = arith.cmpi ule, %Kulelhs, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeulerhs = arith.cmpi ule, %c128, %Kulerhs  : i32
    llvm.intr.assume %assumeulerhs : i1
    // expected-remark@+1 {{unsigned : [128, 4294967295] signed : [-2147483648, 2147483647]}}
    %testulerhs1 = arith.addi %Kulerhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testulerhs2 = arith.cmpi ule, %c64, %Kulerhs : i32

    //// ult comparison

    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeultlhs = arith.cmpi ult, %Kultlhs, %c128 : i32
    llvm.intr.assume %assumeultlhs : i1
    // expected-remark@+2 {{unsigned : [0, 127] signed : [0, 127]}}
    // expected-remark@+1 {{non-neg}}
    %testultlhs1 = arith.addi %Kultlhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testultlhs2 = arith.cmpi ult, %Kultlhs, %c1024_i32 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumeultrhs = arith.cmpi ult, %c128, %Kultrhs  : i32
    llvm.intr.assume %assumeultrhs : i1
    // expected-remark@+1 {{unsigned : [129, 4294967295] signed : [-2147483648, 2147483647]}}
    %testultrhs1 = arith.addi %Kultrhs, %c0 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %testultrhs2 = arith.cmpi ult, %c64, %Kultrhs : i32

    tt.return
  }
}

// -----


// CHECK-LABEL: join_cat_transitive_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @join_cat_transitive_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32>
    // expected-remark@+2 {{unsigned : [0, 10] signed : [0, 10]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.join %0, %1 : tensor<8xi32> -> tensor<8x2xi32>
    %3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %4 = tt.make_range {end = 8 : i32, start = 4 : i32} : tensor<4xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %5 = tt.join %3, %4 : tensor<4xi32> -> tensor<4x2xi32>
    // expected-remark@+2 {{unsigned : [0, 8] signed : [0, 8]}}
    // expected-remark@+1 {{non-neg}}
    %6 = tt.cat %5, %5 : tensor<4x2xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %7 = arith.addi %2, %6 : tensor<8x2xi32>
    %zeros = arith.constant dense<0> : tensor<8x1xi32>
    %ones = arith.constant dense<1> : tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %8 = tt.gather %7[%zeros] {axis = 1 : i32} : (tensor<8x2xi32>, tensor<8x1xi32>) -> tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 18] signed : [0, 18]}}
    // expected-remark@+1 {{non-neg}}
    %9 = tt.gather %7[%ones] {axis = 1 : i32} : (tensor<8x2xi32>, tensor<8x1xi32>) -> tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 36] signed : [0, 36]}}
    // expected-remark@+1 {{non-neg}}
    %10 = arith.addi %8, %9 : tensor<8x1xi32>
    // expected-remark@+2 {{unsigned : [0, 36] signed : [0, 36]}}
    // expected-remark@+1 {{non-neg}}
    %11 = tt.reshape %10 allow_reorder : tensor<8x1xi32> -> tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: histo_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func @histo_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2 : tensor<256xi32>) {
    // expected-remark@+2 {{unsigned : [0, 4294967295] signed : [0, -1]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.histogram %arg2 : tensor<256xi32> -> tensor<8xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: get_num_prog_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{arg 2: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func @get_num_prog_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2 : i32) {
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_num_programs x : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cmpule_num_program0 = arith.cmpi ule, %0, %c65536_i32 : i32
    llvm.intr.assume %cmpule_num_program0 : i1
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.get_num_programs y : i32
    %cmpule_num_program1 = arith.cmpi ule, %1, %c65536_i32 : i32
    llvm.intr.assume %cmpule_num_program1 : i1
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.get_num_programs z : i32
    %cmpule_num_program2 = arith.cmpi ule, %2, %c65536_i32 : i32
    llvm.intr.assume %cmpule_num_program2 : i1
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %3 = arith.minsi %0, %1 : i32
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %4 = arith.minsi %2, %3 : i32
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %5 = arith.maxsi %arg2, %4 : i32
    // expected-remark@+2 {{unsigned : [0, 2147483647] signed : [0, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %6 = tt.splat %5 : i32 -> tensor<8xi32>
    %7 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    // expected-remark@+1 {{unsigned : [0, 2147483655] signed : [-2147483648, 2147483647]}}
    %8 = arith.addi %6, %7 : tensor<8xi32>
    tt.return
  }
}

// -----

// CHECK-LABEL: unary_triton_ops_transitive_nonneg
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @unary_triton_ops_transitive_nonneg(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) {
    %c10_i32 = arith.constant 5 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %2 = tt.reshape %1 allow_reorder : tensor<1x16xi32> -> tensor<8x2xi32>
    %3 = tt.reshape %1 allow_reorder : tensor<1x16xi32> -> tensor<2x8xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %4 = tt.trans %3 {order = array<i32: 1, 0>} : tensor<2x8xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 16] signed : [0, 16]}}
    // expected-remark@+1 {{non-neg}}
    %5 = ttg.convert_layout %4 : tensor<8x2xi32> -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 32] signed : [0, 32]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %2 : tensor<8x2xi32>
    %7 = tt.make_range {end = 10 : i32, start = 2 : i32} : tensor<8xi32>
    // expected-remark@+2 {{unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+1 {{non-neg}}
    %8 = ttg.convert_layout %7 : tensor<8xi32> -> tensor<8xi32>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %10 = tt.broadcast %9 : tensor<1x8xi32> -> tensor<2x8xi32>
    %11 = tt.reshape %10 allow_reorder : tensor<2x8xi32> -> tensor<8x2xi32>
    %12 = tt.splat %c10_i32 : i32 -> tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [7, 15] signed : [7, 15]}}
    // expected-remark@+1 {{non-neg}}
    %13 = arith.addi %11, %12 : tensor<8x2xi32>
    // expected-remark@+2 {{unsigned : [0, 15] signed : [0, 15]}}
    // expected-remark@+1 {{non-neg}}
    %14 = arith.minsi %13, %5 : tensor<8x2xi32>
    // expected-remark@+4 {{result 0: unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+3 {{result 1: unsigned : [2, 10] signed : [2, 10]}}
    // expected-remark@+2 {{result 0: non-neg}}
    // expected-remark@+1 {{result 1: non-neg}}
    %15, %16 = tt.split %11: tensor<8x2xi32> -> tensor<8xi32>
    %17 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %18 = tt.addptr %17, %15 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %19 = tt.load %18 : tensor<8x!tt.ptr<bf16>>
    %20 = tt.addptr %17, %16 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    %21 = tt.load %20 : tensor<8x!tt.ptr<bf16>>
    %22 = arith.addf %19, %21 : tensor<8xbf16>
    %23 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<8x!tt.ptr<bf16>>
    %24 = tt.addptr %23, %7 : tensor<8x!tt.ptr<bf16>>, tensor<8xi32>
    tt.store %24, %22 : tensor<8x!tt.ptr<bf16>>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // expected-remark@+3 {{arg 0: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
  // expected-remark@+2 {{arg 1: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
  // expected-remark@+1 {{arg 2: unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
  tt.func @assume_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16>, %arg4: !tt.ptr<f16>) -> tensor<128x128xf32, #mma> {
    // expected-remark@+1 {{unsigned : [18446744073709551615, 18446744073709551615] signed : [-1, -1]}}
    %c-1 = arith.constant -1 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // expected-remark@+1 {{unsigned : [1, 1] signed : [-1, -1]}}
    %true = arith.constant true
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #blocked>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    %11 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %12 = arith.cmpi slt, %arg0, %arg1 : index
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %13 = tt.splat %12 : i1 -> tensor<128x32xi1, #blocked1>
    %14 = tt.load %4, %13 : tensor<128x32x!tt.ptr<f16>, #blocked1>
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %15 = tt.splat %12 : i1 -> tensor<32x128xi1, #blocked>
    %16 = tt.load %9, %15, %cst_3 : tensor<32x128x!tt.ptr<f16>, #blocked>
    %17 = ttg.memdesc_index %10[%c0_i32] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    ttg.local_store %14, %17 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
    %18 = ttg.memdesc_index %11[%c0_i32] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    ttg.local_store %16, %18 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %19 = arith.subi %arg1, %arg2 : index
    // expected-remark@+1 {{inferred total trip count: 0}}
    %20:6 = scf.for %arg5 = %arg0 to %19 step %arg2 iter_args(%arg6 = %4, %arg7 = %9, %arg8 = %cst_2, %arg9 = %c0_i32, %arg10 = %17, %arg11 = %18) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>) {
      %33 = tt.addptr %arg6, %cst_1 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %34 = tt.addptr %arg7, %cst_0 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      llvm.intr.assume %true : i1
      %35 = tt.load %33 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %36 = ttg.local_load %arg10 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %37 = tt.load %34 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %38 = ttg.local_load %arg11 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %39 = arith.mulf %38, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %40 = tt.dot %36, %39, %arg8 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %41 = arith.addi %arg9, %c1_i32 : i32
      %42 = arith.cmpi slt, %41, %c1_i32 : i32
      // expected-remark@+2 {{unsigned : [0, 0] signed : [0, 0]}}
      // expected-remark@+1 {{non-neg}}
      %43 = arith.select %42, %41, %c0_i32 : i32
      %44 = ttg.memdesc_index %10[%43] : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      ttg.local_store %35, %44 : tensor<128x32xf16, #blocked1> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable>
      %45 = ttg.memdesc_index %11[%43] : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      ttg.local_store %37, %45 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
      scf.yield %33, %34, %40, %43, %44, %45 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>, i32, !ttg.memdesc<128x32xf16, #shared, #smem, mutable>, !ttg.memdesc<32x128xf16, #shared1, #smem, mutable>
    }
    // expected-remark@+1 {{unsigned : [0, 1] signed : [-1, 0]}}
    %21 = arith.cmpi slt, %arg2, %c0 : index
    // expected-remark@+1 {{unsigned : [1, 18446744073709551615] signed : [-1, 1]}}
    %22 = arith.select %21, %c1, %c-1 : index
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %23 = arith.subi %arg1, %arg0 : index
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %24 = arith.addi %23, %arg2 : index
    // expected-remark@+1 {{unsigned : [0, 18446744073709551615] signed : [-9223372036854775808, 9223372036854775807]}}
    %25 = arith.addi %24, %22 : index
    // expected-remark@+2 {{unsigned : [1, 9223372036854775807] signed : [1, 9223372036854775807]}}
    // expected-remark@+1 {{non-neg}}
    %26 = arith.divsi %25, %arg2 : index
    %28 = ttg.local_load %20#4 : !ttg.memdesc<128x32xf16, #shared, #smem, mutable> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %29 = ttg.local_load %20#5 : !ttg.memdesc<32x128xf16, #shared1, #smem, mutable> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %30 = arith.mulf %29, %cst : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %27 = arith.cmpi sge, %26, %c1 : index
    llvm.intr.assume %27 : i1
    %31 = scf.if %27 -> (tensor<128x128xf32, #mma>) {
      %33 = tt.dot %28, %30, %20#2 : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      scf.yield %33 : tensor<128x128xf32, #mma>
    } else {
      scf.yield %20#2 : tensor<128x128xf32, #mma>
    }
    %32 = arith.select %27, %31, %20#2 : tensor<128x128xf32, #mma>
    ttg.local_dealloc %10 : !ttg.memdesc<1x128x32xf16, #shared, #smem, mutable>
    ttg.local_dealloc %11 : !ttg.memdesc<1x32x128xf16, #shared1, #smem, mutable>
    tt.return %32 : tensor<128x128xf32, #mma>
  }
}

// -----

// CHECK-LABEL:   tt.func @assume_func_args
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [1024, 2147483647] signed : [1024, 2147483647]}}
  tt.func @assume_func_args(%arg0: i32) -> i1 {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assumege = arith.cmpi sge, %arg0, %c1024_i32 : i32
    llvm.intr.assume %assumege : i1
    %c256_i32 = arith.constant 256 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmpge = arith.cmpi sge, %arg0, %c256_i32 : i32
    tt.return %cmpge : i1
  }
}

// -----

// CHECK-LABEL:   tt.func @assume_func_args_two_bounds
module attributes {"ttg.num-warps" = 4 : i32} {
  // expected-remark@+1 {{unsigned : [256, 1024] signed : [256, 1024]}}
  tt.func @assume_func_args_two_bounds(%arg0: i32) -> i1 {
    %c1024_i32 = arith.constant 1024 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assume_sle_1024 = arith.cmpi sle, %arg0, %c1024_i32 : i32
    llvm.intr.assume %assume_sle_1024 : i1
    %c256_i32 = arith.constant 256 : i32
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assume_sge_256 = arith.cmpi sge, %arg0, %c256_i32 : i32
    llvm.intr.assume %assume_sge_256 : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assume_ule_1024 = arith.cmpi ule, %arg0, %c1024_i32 : i32
    llvm.intr.assume %assume_ule_1024 : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %assume_uge_256 = arith.cmpi uge, %arg0, %c256_i32 : i32
    llvm.intr.assume %assume_uge_256 : i1

    tt.return %assume_sge_256 : i1
  }
}

// -----

// CHECK-LABEL: buffer_stride
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // expected-remark@+7 {{arg 3: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  // expected-remark@+6 {{arg 4: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  // expected-remark@+5 {{arg 5: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  // expected-remark@+4 {{arg 6: unsigned : [1, 2147483647] signed : [1, 2147483647]}}
  // expected-remark@+3 {{arg 7: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  // expected-remark@+2 {{arg 8: unsigned : [1, 2147483647] signed : [1, 1023]}}
  // expected-remark@+1 {{arg 9: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func public @buffer_stride(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1024_i32 = arith.constant 1024 : i32
    %c48_i32 = arith.constant 48 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp = arith.cmpi sgt, %arg6, %c0_i32 : i32
    llvm.intr.assume %cmp : i1
    // expected-remark@+2 {{unsigned : [1, 2147483647] signed : [1, 2147483647]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.splat %arg6 : i32 -> tensor<256x1xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %3 = arith.muli %1, %2 : tensor<256x1xi32, #blocked>
    %4 = tt.addptr %arg0, %c32_i32 : !tt.ptr<f16>, i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %5 = tt.broadcast %3 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %6 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 64] signed : [0, 64]}}
    // expected-remark@+1 {{non-neg}}
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 64] signed : [0, 64]}}
    // expected-remark@+1 {{non-neg}}
    %8 = tt.broadcast %7 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %9 = arith.addi %8, %5 : tensor<256x64xi32, #blocked>
    %10 = tt.splat %4 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %11 = tt.addptr %10, %9 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
    %12 = tt.load %11 : tensor<256x64x!tt.ptr<f16>, #blocked>
    %13 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    // expected-remark@+2 {{unsigned : [0, 256] signed : [0, 256]}}
    // expected-remark@+1 {{non-neg}}
    %15 = tt.expand_dims %13 {axis = 1 : i32} : tensor<256xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 64] signed : [0, 64]}}
    // expected-remark@+1 {{non-neg}}
    %16 = tt.expand_dims %14 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp1 = arith.cmpi sgt, %arg8, %c0_i32 : i32
    llvm.intr.assume %cmp1 : i1
    // expected-remark@+2 {{unsigned : [1, 1] signed : [-1, -1]}}
    // expected-remark@+1 {{result is true}}
    %cmp2 = arith.cmpi slt, %arg8, %c1024_i32 : i32
    llvm.intr.assume %cmp2 : i1
    // expected-remark@+2 {{unsigned : [1, 2147483647] signed : [1, 1023]}}
    // expected-remark@+1 {{non-neg}}
    %17 = tt.splat %arg8 : i32 -> tensor<256x1xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 261888] signed : [0, 261888]}}
    // expected-remark@+1 {{non-neg}}
    %18 = arith.muli %17, %15 : tensor<256x1xi32, #blocked>
    %19 = tt.addptr %arg2, %c48_i32 : !tt.ptr<f16>, i32
    // expected-remark@+2 {{unsigned : [0, 261888] signed : [0, 261888]}}
    // expected-remark@+1 {{non-neg}}
    %20 = tt.broadcast %18 : tensor<256x1xi32, #blocked> -> tensor<256x64xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 64] signed : [0, 64]}}
    // expected-remark@+1 {{non-neg}}
    %21 = tt.broadcast %16 : tensor<1x64xi32, #blocked> -> tensor<256x64xi32, #blocked>
    %22 = tt.addptr %19, %c48_i32 : !tt.ptr<f16>, i32
    // expected-remark@+2 {{unsigned : [0, 261952] signed : [0, 261952]}}
    // expected-remark@+1 {{non-neg}}
    %23 = arith.addi %21, %20 : tensor<256x64xi32, #blocked>
    %24 = tt.splat %22 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %25 = tt.addptr %24, %23 : tensor<256x64x!tt.ptr<f16>, #blocked>, tensor<256x64xi32, #blocked>
    tt.store %25, %12 : tensor<256x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: zero_divisor_for_loop_step
  // expected-remark@+1 {{arg 2: unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
  tt.func public @zero_divisor_for_loop_step(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
    %c127_i32 = arith.constant 127 : i32
    %c128_i32 = arith.constant 128 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0xFF800000> : tensor<32xf32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %0 = tt.get_program_id x : i32
    %c65535_i32 = arith.constant 65535 : i32
    %cmpule_pid0 = arith.cmpi ule, %0, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid0 : i1
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %1 = tt.get_program_id y : i32
    %cmpule_pid1 = arith.cmpi ule, %1, %c65535_i32 : i32
    llvm.intr.assume %cmpule_pid1 : i1
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %2 = tt.get_num_programs y : i32
    %c65536_i32 = arith.constant 65536 : i32
    %cmpule_num_program1 = arith.cmpi ule, %2, %c65536_i32 : i32
    llvm.intr.assume %cmpule_num_program1 : i1
    // expected-remark@+2 {{unsigned : [0, 2097120] signed : [0, 2097120]}}
    // expected-remark@+1 {{non-neg}}
    %3 = arith.muli %0, %c32_i32 : i32
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 2097120] signed : [0, 2097120]}}
    // expected-remark@+1 {{non-neg}}
    %5 = tt.splat %3 : i32 -> tensor<32xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 2097152] signed : [0, 2097152]}}
    // expected-remark@+1 {{non-neg}}
    %6 = arith.addi %5, %4 : tensor<32xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %7 = arith.addi %arg2, %c127_i32 : i32
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-16777216, 16777215]}}
    %8 = arith.divsi %7, %c128_i32 : i32
    %9 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 2097152] signed : [0, 2097152]}}
    // expected-remark@+1 {{non-neg}}
    %10 = ttg.convert_layout %6 : tensor<32xi32, #blocked> -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    // expected-remark@+2 {{unsigned : [0, 2097152] signed : [0, 2097152]}}
    // expected-remark@+1 {{non-neg}}
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    // expected-remark@+2 {{unsigned : [0, 2097152] signed : [0, 2097152]}}
    // expected-remark@+1 {{non-neg}}
    %12 = ttg.convert_layout %11 : tensor<32x1xi32, #blocked1> -> tensor<32x1xi32, #blocked2>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %13 = tt.splat %arg2 : i32 -> tensor<32x1xi32, #blocked2>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %14 = arith.muli %12, %13 : tensor<32x1xi32, #blocked2>
    %15 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>, #blocked2>
    %16 = tt.addptr %15, %14 : tensor<32x1x!tt.ptr<f32>, #blocked2>, tensor<32x1xi32, #blocked2>
    %17 = tt.broadcast %16 : tensor<32x1x!tt.ptr<f32>, #blocked2> -> tensor<32x128x!tt.ptr<f32>, #blocked2>
    %18 = ttg.convert_layout %17 : tensor<32x128x!tt.ptr<f32>, #blocked2> -> tensor<32x128x!tt.ptr<f32>, #blocked3>
    // expected-remark@+1 {{inferred total trip count: 16711680}}
    %19 = scf.for %arg3 = %1 to %8 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #blocked>)  : i32 {
      // expected-remark@+2 {{unsigned : [0, 2147483392] signed : [0, 2147483392]}}
      // expected-remark@+1 {{non-neg}}
      %26 = arith.muli %arg3, %c128_i32 : i32
      // expected-remark@+2 {{unsigned : [0, 2147483392] signed : [0, 2147483392]}}
      // expected-remark@+1 {{non-neg}}
      %27 = tt.splat %26 : i32 -> tensor<128xi32, #blocked>
      // expected-remark@+2 {{unsigned : [0, 2147483520] signed : [0, 2147483520]}}
      // expected-remark@+1 {{non-neg}}
      %28 = arith.addi %27, %9 : tensor<128xi32, #blocked>
      // expected-remark@+2 {{unsigned : [0, 2147483520] signed : [0, 2147483520]}}
      // expected-remark@+1 {{non-neg}}
      %29 = ttg.convert_layout %28 : tensor<128xi32, #blocked> -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>>
      // expected-remark@+2 {{unsigned : [0, 2147483520] signed : [0, 2147483520]}}
      // expected-remark@+1 {{non-neg}}
      %30 = tt.expand_dims %29 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked4}>> -> tensor<1x128xi32, #blocked4>
      // expected-remark@+2 {{unsigned : [0, 2147483520] signed : [0, 2147483520]}}
      // expected-remark@+1 {{non-neg}}
      %31 = ttg.convert_layout %30 : tensor<1x128xi32, #blocked4> -> tensor<1x128xi32, #blocked3>
      // expected-remark@+2 {{unsigned : [0, 2147483520] signed : [0, 2147483520]}}
      // expected-remark@+1 {{non-neg}}
      %32 = tt.broadcast %31 : tensor<1x128xi32, #blocked3> -> tensor<32x128xi32, #blocked3>
      %33 = tt.addptr %18, %32 : tensor<32x128x!tt.ptr<f32>, #blocked3>, tensor<32x128xi32, #blocked3>
      %34 = tt.load %33 : tensor<32x128x!tt.ptr<f32>, #blocked3>
      %35 = "tt.reduce"(%34) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %38 = arith.maxnumf %arg5, %arg6 : f32
        tt.reduce.return %38 : f32
      }) : (tensor<32x128xf32, #blocked3>) -> tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked3}>>
      %36 = ttg.convert_layout %35 : tensor<32xf32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<32xf32, #blocked>
      %37 = arith.maxnumf %arg4, %36 : tensor<32xf32, #blocked>
      scf.yield %37 : tensor<32xf32, #blocked>
    }
    // expected-remark@+2 {{unsigned : [0, 65536] signed : [0, 65536]}}
    // expected-remark@+1 {{non-neg}}
    %20 = tt.splat %2 : i32 -> tensor<32xi32, #blocked>
    // expected-remark@+1 {{unsigned : [0, 4294967295] signed : [-2147483648, 2147483647]}}
    %21 = arith.muli %6, %20 : tensor<32xi32, #blocked>
    %22 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %23 = tt.addptr %22, %21 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    // expected-remark@+2 {{unsigned : [0, 65535] signed : [0, 65535]}}
    // expected-remark@+1 {{non-neg}}
    %24 = tt.splat %1 : i32 -> tensor<32xi32, #blocked>
    %25 = tt.addptr %23, %24 : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %25, %19 : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
