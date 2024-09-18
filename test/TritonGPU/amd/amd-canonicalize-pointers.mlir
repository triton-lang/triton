// RUN: triton-opt %s -split-input-file -tritonamdgpu-canonicalize-pointers | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @conversion1
  tt.func @conversion1(%arg0: !tt.ptr<f32>)-> tensor<1024xf32, #blocked>{
     %c1024_i32 = arith.constant 1024 : i32
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
     // CHECK: %[[scalarOffset:.*]] = arith.muli{{.*}} : i32
     // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]] : !tt.ptr<f32>, i32
     // CHECK: %[[offset_32bit:.*]] = arith.trunci %{{.*}} : tensor<1024xi64, #blocked> to tensor<1024xi32, #blocked>
     // CHECK: %[[basePtr:.*]] = tt.splat %[[scalarPtr]]
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[offset_32bit]]
     // CHECK: tt.load %[[newPtr]]
     %6 = tt.addptr %5, %3 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %7 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @conversion2
  tt.func @conversion2(%arg0: !tt.ptr<f32>)-> tensor<1024xf32, #blocked>{
     %c1024_i32 = arith.constant 1024 : i32
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
     %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
     %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
     // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
     // CHECK: %[[baseOffset064bit:.*]] = tt.splat {{.*}} : i64
     // CHECK: %[[newScalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]]
     // CHECK: %[[offset064bit:.*]] = arith.extsi {{.*}}
     // CHECK: %[[offset164bit:.*]] = arith.addi %[[offset064bit]], %[[baseOffset064bit]]
     // CHECK: %[[offset132bit:.*]] = arith.trunci %[[offset164bit]] : tensor<1024xi64, #blocked> to tensor<1024xi32, #blocked>
     // CHECK: %[[basePtr:.*]] = tt.splat %[[newScalarPtr]]
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[offset132bit]]
     // CHECK: tt.load %[[newPtr]]
     %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %7 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @conversion3
  tt.func @conversion3(%arg0: !tt.ptr<f32>)-> tensor<1024xf32, #blocked>{
     %c1024_i32 = arith.constant 1024 : i32
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
     %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
     %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>

     //CHECK: %0 = tt.get_program_id x : i32
     //CHECK: %[[pid:.*]] = arith.muli %0, {{.*}} : i32
     //CHECK: %[[makerange:.*]] = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
     //CHECK: %[[uniformOffset1:.*]] = arith.addi %[[pid]], {{.*}} : i32
     //CHECK: %[[tensorOffset1:.*]] = arith.addi %{{.*}}, %[[makerange]] : tensor<1024xi32, #blocked>
     //CHECK: %[[uniformOffset0:.*]] = arith.addi %[[pid:.*]], %{{.*}} : i32
     //CHECK: %[[tensorOffset3:.*]] = arith.addi %{{.*}}, %[[makerange]] : tensor<1024xi32, #blocked>
     //CHECK: %[[zero:.*]] = tt.splat %{{.*}} : i64 -> tensor<1024xi64, #blocked>
     //CHECK: %[[uniformPtr0:.*]] = tt.addptr %arg0, %[[uniformOffset0:.*]] : !tt.ptr<f32>, i32
     //CHECK: %[[tensorOffset3ext:.*]] = arith.extsi %[[tensorOffset3]] : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
     //CHECK: %[[tensorOffset0:.*]]= arith.addi %[[tensorOffset3ext]], %[[zero]] : tensor<1024xi64, #blocked>
     //CHECK: %[[uniformPtr1:.*]] = tt.addptr %[[uniformPtr0]], %[[uniformOffset1]] : !tt.ptr<f32>, i32
     //CHECK: %[[tensorOffset1ext:.*]] = arith.extsi %[[tensorOffset1]] : tensor<1024xi32, #blocked> to tensor<1024xi64, #blocked>
     //CHECK: %[[tensorOffset2:.*]] = arith.addi %[[tensorOffset1ext]], %[[tensorOffset0]]: tensor<1024xi64, #blocked>
     //CHECK: %[[scalarPtr:.*]] = tt.splat %[[uniformPtr1]] : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
     //CHECK: %[[newPtr:.*]] = tt.addptr %[[scalarPtr]], %[[tensorOffset2]] : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi64, #blocked>
     //CHECK: tt.load %[[newPtr]]

     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
     %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %8 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @forOp
  tt.func @forOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffsetLoop:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset1:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor
    // CHECK: %[[scalarOffset1:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[scalarOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor
    // CHECK: %[[scalarPtrUpdate:.*]] = tt.addptr %arg0, %[[scalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %{{.*}} : tensor<1024xi64, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[loop:.*]]:4 = scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[loopScalarPtr:.*]] = %{{.*}}, %[[loopOffset:.*]] = %[[offset1]]) -> {{.*}} {
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %6, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[scalarPtrUpdateLoop:.*]] = tt.addptr %[[loopScalarPtr]], %[[scalarOffsetLoop]]
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[variableOffset1]]
        // CHECK: %[[offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtrUpdateLoop]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[offset_i]]
        // CHECK: tt.load %[[newPtr]]
        // CHECK: scf.yield {{.*}}, {{.*}}, %[[scalarPtrUpdateLoop]], %[[offset_i]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    // CHECK: tt.addptr %[[loop]]#2, %[[scalarOffset1]] : !tt.ptr<f32>, i32
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @forOp2
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[variableOffset0:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[finalScalarOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[variableOffset1:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[forOut:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr:.*]] = %arg0, %[[loopOffset:.*]] = %[[base_offset]])
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[scalarPtrUpdate:.*]] = tt.addptr %[[scalarPtr]], %[[scalarOffset]]
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[variableOffset0]]
        // CHECK: %[[ext_offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtrUpdate]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[ext_offset_i]]
        // CHECK: tt.load %[[newPtr]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    // CHECK: %[[scalarPtrFinalUpdate:.*]] = tt.addptr %[[forOut]]#2, %[[finalScalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset1]]
    // CHECK: %[[tailOffset:.*]] = arith.addi %[[ext_offset0]], %[[forOut]]#3
    // CHECK: %[[tail_base_ptr:.*]] = tt.splat %[[scalarPtrFinalUpdate]]
    // CHECK: %[[tailPtr:.*]] = tt.addptr %[[tail_base_ptr]], %[[tailOffset]]
    // CHECK: tt.load %[[tailPtr]]
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @forNested
  tt.func @forNested(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>

    // CHECK: %[[forOut0:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr0:.*]] = %arg0, %[[loopOffset0:.*]] = %[[base_offset]]){{.*}}{
    // CHECK: %[[forOut1:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr1:.*]] = %[[scalarPtr0]], %[[loopOffset1:.*]] = %[[loopOffset0]]){{.*}}{
    // CHECK: %[[scalarPtrUpdate:.*]] = tt.addptr %[[scalarPtr1]], %{{.*}}
    // CHECK: %[[ext_loop_offset1:.*]] = arith.extsi %[[variableOffset]]
    // CHECK: %[[offset_i:.*]] = arith.addi %[[ext_loop_offset1]], %[[loopOffset1]]
    // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtrUpdate]]
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[offset_i]]
    // CHECK: tt.load %[[newPtr]]
    // CHECK: scf.yield %{{.*}}, {{.*}}, %[[scalarPtrUpdate]], %[[offset_i]]
    // CHECK: scf.yield %{{.*}}, {{.*}}, %[[forOut1]]#2, %[[forOut1]]#3

    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        %53:2 = scf.for %arg10 = %c0 to %c128 step %c1 iter_args(%arg3 = %arg1, %arg4 = %arg2) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
            %11 = tt.addptr %arg3, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
            %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
            %10 = arith.addf %9, %arg4 : tensor<1024xf32, #blocked>
            scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
        }
        scf.yield %53#0, %53#1: tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @ifOp
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>, %cond : i1)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[baseOffsetVariable:.*]] = tt.splat {{.*}} : i64 -> tensor<1024xi64, #blocked>
    // CHECK: %[[ifOut:.*]]:3 = scf.if {{.*}} -> (tensor<1024x!tt.ptr<f32>, #blocked>, !tt.ptr<f32>, tensor<1024xi64, #blocked>)
    %6 = scf.if %cond -> (tensor<1024x!tt.ptr<f32>, #blocked>){
        // CHECK: %[[scalarOffsetUpdate:.*]] = tt.addptr %arg0, %[[scalarOffset]]
        // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
        // CHECK: %[[if_offset:.*]] = arith.addi %[[ext_offset0]], %[[baseOffsetVariable]]
        // CHECK: scf.yield %{{.*}}, %[[scalarOffsetUpdate]], %[[if_offset]]
        %true = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        scf.yield %true : tensor<1024x!tt.ptr<f32>, #blocked>
    } else {
        // CHECK: %[[new_scalar_ptr:.*]] = tt.addptr %arg0, {{.*}}
        // CHECK: scf.yield %{{.*}}, %[[new_scalar_ptr]], %[[baseOffsetVariable]]
        %false = tt.addptr %5, %3 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        scf.yield %false : tensor<1024x!tt.ptr<f32>, #blocked>
    }
    // CHECK: %[[trunc_offset:.*]] = arith.trunci %[[ifOut]]#2
    // CHECK: %[[base_ptr:.*]] = tt.splat %[[ifOut]]#1
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[trunc_offset]]
    // CHECK: tt.load %[[newPtr]]
    %11 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL  tt.func @whileOp
  tt.func @whileOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>, %cond : i1)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[whileOut:.*]]:3 = scf.while ({{.*}}, %[[loopPtr:.*]] = %arg0, %[[loopOffset:.*]] = %[[base_offset]])
    %6 = scf.while (%arg1 = %5, %arg2 = %cond) : (tensor<1024x!tt.ptr<f32>, #blocked>, i1) -> (tensor<1024x!tt.ptr<f32>, #blocked>) {
        // CHECK: scf.condition({{.*}}) %{{.*}}, %[[loopPtr]], %[[loopOffset]]
        scf.condition(%arg2) %arg1 : tensor<1024x!tt.ptr<f32>, #blocked>
        } do {
        // CHECK: ^bb{{.*}}(%{{.*}}, %[[blockPtr:.*]]: !tt.ptr<f32>, %[[blockOffset:.*]]: tensor<1024xi64, #blocked>):
        ^bb0(%arg1: tensor<1024x!tt.ptr<f32>, #blocked>):
        // CHECK: scf.yield {{.*}}, %[[blockPtr]], %[[blockOffset]]
        scf.yield %arg1, %cond : tensor<1024x!tt.ptr<f32>, #blocked>, i1
        }
    // CHECK: %[[trunc_offset:.*]] = arith.trunci %[[whileOut]]#2
    // CHECK: %[[base_ptr:.*]] = tt.splat %[[whileOut]]#1
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[trunc_offset]]
    // CHECK: tt.load %[[newPtr]]
    %11 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL  tt.func @condBranch
  tt.func @condBranch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: cf.cond_br {{.*}}, ^bb1(%{{.*}}, %arg0, %[[base_offset]] : {{.*}}), ^bb2(%{{.*}}, %[[scalarPtr]], %[[offset1]] : {{.*}})
    cf.cond_br %i1, ^bb1(%5 : tensor<1024x!tt.ptr<f32>, #blocked>), ^bb2(%6 : tensor<1024x!tt.ptr<f32>, #blocked>)
  // CHECK: ^bb1({{.*}}, %[[block1ScalarPtr:.*]]: !tt.ptr<f32>, %[[block1Offset:.*]]: tensor<1024xi64, #blocked>)
  ^bb1(%arg1 : tensor<1024x!tt.ptr<f32>, #blocked>):
    // CHECK: %[[trunc_offset_1:.*]] = arith.trunci %[[block1Offset]]
    // CHECK: %[[basePtr1:.*]] = tt.splat %[[block1ScalarPtr]]
    // CHECK: %[[newPtr1:.*]] = tt.addptr %[[basePtr1]], %[[trunc_offset_1]]
    // CHECK: tt.load %[[newPtr1]]
    %out1 = tt.load %arg1 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %out1 : tensor<1024xf32, #blocked>
  // CHECK: ^bb2({{.*}}, %[[block2ScalarPtr:.*]]: !tt.ptr<f32>, %[[block2Offset:.*]]: tensor<1024xi64, #blocked>)
  ^bb2(%arg2 : tensor<1024x!tt.ptr<f32>, #blocked>):  // 2 preds: ^bb0, ^bb1
    // CHECK: %[[trunc_offset_2:.*]] = arith.trunci %[[block2Offset]]
    // CHECK: %[[basePtr2:.*]] = tt.splat %[[block2ScalarPtr]]
    // CHECK: %[[newPtr2:.*]] = tt.addptr %[[basePtr2]], %[[trunc_offset_2]]
    // CHECK: tt.load %[[newPtr2]]
    %out2 = tt.load %arg2 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %out2 : tensor<1024xf32, #blocked>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: tt.func @branch
  tt.func @branch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: cf.br ^bb1(%{{.*}}, %[[scalarPtr]], %[[offset1]] : {{.*}})
    // CHECK: ^bb1({{.*}}, %[[block1ScalarPtr:.*]]: {{.*}}, %[[block1Offset:.*]]: {{.*}})
    cf.br ^bb1(%6 : tensor<1024x!tt.ptr<f32>, #blocked>)
  ^bb1(%arg1 : tensor<1024x!tt.ptr<f32>, #blocked>):
    // CHECK: %[[trunc_offset_1:.*]] = arith.trunci %[[block1Offset]]
    // CHECK: %[[basePtr1:.*]] = tt.splat %[[block1ScalarPtr]]
    // CHECK: %[[newPtr1:.*]] = tt.addptr %[[basePtr1]], %[[trunc_offset_1]]
    // CHECK: tt.load %[[newPtr1]]
    %out1 = tt.load %arg1 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %out1 : tensor<1024xf32, #blocked>
  }
}

// -----

// The following is a simple case of a tile offset like: (A*B + C + D) where B,C are Uniform and A,D are not. So
// we expect that the Uniform offset (which can be added to the scalar pointer) will be simply C and the NonUniform
// offset will be A*B+D
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @tile_offset
  tt.func @tile_offset(%arg1: !tt.ptr<f16>,  %arg5: i32 , %arg7: i32 )  {
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = tt.get_program_id x : i32
    %20 = arith.muli %1, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %24 = tt.splat %20 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %26 = arith.addi %24, %22 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %36 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %37 = tt.expand_dims %36 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %38 = tt.splat %arg7 : i32 -> tensor<16x1xi32, #blocked>
    %39 = arith.muli %37, %38 : tensor<16x1xi32, #blocked>
    %41 = tt.expand_dims %26 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %42 = tt.broadcast %39 : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %43 = tt.broadcast %41 : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    %44 = arith.addi %42, %43 : tensor<16x256xi32, #blocked>
    %45 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    %46 = tt.addptr %45, %44 : tensor<16x256x!tt.ptr<f16>, #blocked>, tensor<16x256xi32, #blocked>
    // CHECK: %[[uniformOffset1:.*]] = arith.muli %c0_i32_0, %arg2 : i32
    // CHECK: {{.*}} = tt.expand_dims %{{.*}} {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    // CHECK: %[[tensorOffset6:.*]] = tt.expand_dims %{{.*}} {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    // CHECK: {{.*}} = tt.broadcast %{{.*}} : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[tensorOffset3:.*]] = tt.broadcast %{{.*}} : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[tensorOffset4:.*]] = tt.broadcast %{{.*}} : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[tensorOffset5:.*]] = tt.broadcast %[[tensorOffset6]] : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[uniformOffset:.*]] = arith.addi %[[uniformOffset1]], %{{.*}}: i32
    // CHECK: %[[tensorOffset2:.*]] = arith.addi %[[tensorOffset3]], %[[tensorOffset5]] : tensor<16x256xi32, #blocked>
    // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[uniformOffset]] : !tt.ptr<f16>, i32
    // CHECK: %[[tensorOffset2ext:.*]] = arith.extsi %[[tensorOffset2]] : tensor<16x256xi32, #blocked> to tensor<16x256xi64, #blocked>
    // CHECK: %[[tensorOffset1:.*]] = arith.addi %[[tensorOffset2ext]], %{{.*}} : tensor<16x256xi64, #blocked>
    // CHECK: %[[tensorOffset:.*]] = arith.trunci %[[tensorOffset1:.*]] : tensor<16x256xi64, #blocked> to tensor<16x256xi32, #blocked>
    // CHECK: %[[ptr:.*]] = tt.splat %[[scalarPtr]] : !tt.ptr<f16> -> tensor<16x256x!tt.ptr<f16>, #blocked>
    // CHECK: tt.addptr %[[ptr]], %[[tensorOffset]] : tensor<16x256x!tt.ptr<f16>, #block
    %61 = tt.load %46 : tensor<16x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// The following is a more complex case where also a multiplication is involved. It's useful to walk through the case.
// We have that the offset to the pointer is the following:
//   %12 = %10 + 11
// This can be transformed in:
//  = %7 + %9
//  = %5*%6 + %8
//  = %4*%arg1 + %8
//  = (%3+%2)*%arg1 + %8
//  = (%1 + %2) * %arg1 + %8
//  = (U + N)*U + N
// Where U means uniform (e.g., a splat) and N means NonUniform (e.g., a make_range)
// The scalar offset we want is (%1*%arg1), while the variable offset should be (%2*%arg1 + %8)
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.splat %1 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %4 = arith.addi %3, %2 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %6 = tt.splat %arg1 : i32 -> tensor<128x1xi32, #blocked>
    %7 = arith.muli %5, %6 : tensor<128x1xi32, #blocked>
    %8 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %10 = tt.broadcast %7 : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %11 = tt.broadcast %9 : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    %12 = arith.addi %10, %11 : tensor<128x16xi32, #blocked>
    %13 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    %14 = tt.addptr %13, %12 : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    %15 = tt.load %14 : tensor<128x16x!tt.ptr<f16>, #blocked>
    // CHECK: %[[pid:.*]] = tt.get_program_id x : i32
    // CHECK: %[[uniformOffset3:.*]] = arith.muli %[[pid]], %{{.*}} : i32
    // CHECK: %[[uniformOffset2:.*]] = arith.addi %[[uniformOffset3]], %{{.*}} : i32
    // CHECK: %[[uniformOffset1:.*]] = arith.muli %[[uniformOffset2]], %arg1 : i32
    // CHECK: %[[makerange:.*]] = tt.make_range
    // CHECK: %{{.*}} = tt.expand_dims %[[makerange]] {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    // CHECK: %[[tensorOffset6:.*]] = tt.expand_dims %[[makerange]] {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    // CHECK: %{{.*}} = tt.broadcast %{{.*}} : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // CHECK: %[[tensorOffset3:.*]] = tt.broadcast %{{.*}} : tensor<128x1xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // CHECK: %{{.*}} = tt.broadcast %{{.*}} : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // CHECK: %[[tensorOffset4:.*]] = tt.broadcast %[[tensorOffset6]] : tensor<1x16xi32, #blocked> -> tensor<128x16xi32, #blocked>
    // CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : tensor<128x16xi32, #blocked>
    // CHECK: %[[uniformOffset:.*]] = arith.addi %[[uniformOffset1]], %{{.*}} : i32
    // CHECK: %[[tensorOffset2:.*]] = arith.addi %[[tensorOffset3]], %[[tensorOffset4]] : tensor<128x16xi32, #blocked>
    // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[uniformOffset]] : !tt.ptr<f16>, i32
    // CHECK: %[[tensorOffset1Ext:.*]] = arith.extsi %[[tensorOffset2]] : tensor<128x16xi32, #blocked> to tensor<128x16xi64, #blocked>
    // CHECK: %[[tensorOffset:.*]] = arith.addi %[[tensorOffset1Ext]], %{{.*}} : tensor<128x16xi64, #blocked>
    // CHECK: %[[tensorOffsetTrunc:.*]] = arith.trunci %[[tensorOffset]] : tensor<128x16xi64, #blocked> to tensor<128x16xi32, #blocked>
    // CHECK: %[[ptr:.*]] = tt.splat %[[scalarPtr]] : !tt.ptr<f16> -> tensor<128x16x!tt.ptr<f16>, #blocked>
    // CHECK: tt.addptr %[[ptr]], %[[tensorOffsetTrunc]] : tensor<128x16x!tt.ptr<f16>, #blocked>, tensor<128x16xi32, #blocked>
    tt.return
  }
}


// -----
#blocked = #triton_gpu.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL  tt.func @condBranch
  tt.func @select(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[variableOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<1024xi32, #blocked>
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: %[[offset2:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    // CHECK: %[[scalarPtr1:.*]] = arith.select %arg1, %arg0, %[[scalarPtr]]
    // CHECK: %[[offset0:.*]] = arith.select %arg1, {{.*}}, %[[offset2]]
    // CHECK: %[[offset1:.*]] = arith.trunci %[[offset0]]
    // CHECK: %[[ptr:.*]] = tt.splat %[[scalarPtr1]]
    // CHECK: tt.addptr %[[ptr]], %[[offset1]]
    %7 = arith.select %i1, %5 , %6 : tensor<1024x!tt.ptr<f32>, #blocked>
    %out = tt.load %7: tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %out : tensor<1024xf32, #blocked>
  }
}
