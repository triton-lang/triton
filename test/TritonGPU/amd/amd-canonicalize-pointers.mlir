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
     // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
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
     // CHECK: %[[baseOffset064bit:.*]] = tt.splat {{.*}} : i64
     // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
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
     // CHECK: %[[offset0:.*]] = tt.make_range
     // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
     // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
     // CHECK: %[[newScalarPtr:.*]] = tt.addptr %arg0, %[[scalarOffset]]
     // CHECK: %[[ext_offset00:.*]] = arith.extsi %[[offset0]]
     // CHECK: %[[newOffset0:.*]] = arith.addi %[[ext_offset00]], %[[base_offset]]
     // CHECK: %[[newScalarPtr0:.*]] = tt.addptr %[[newScalarPtr]]
     // CHECK: %[[ext_offset01:.*]] = arith.extsi %[[offset0]]
     // CHECK: %[[newOffset1:.*]] = arith.addi %[[ext_offset01]], %[[newOffset0]]
     // CHECK: %[[basePtr:.*]] = tt.splat %[[newScalarPtr0]] : !tt.ptr<f32>
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[newOffset1]]
     // CHECK: {{.*}} = tt.load %[[newPtr]]
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
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
    // CHECK: %[[scalarPtrUpdate:.*]] = tt.addptr %arg0, %[[scalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %{{.*}} : tensor<1024xi64, #blocked>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: {{.*}}:4 = scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[loopScalarPtr:.*]] = %{{.*}}, %[[loopOffset:.*]] = %[[offset1]]) -> {{.*}} {
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %6, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[scalarOffsetLoop:.*]] = arith.addi {{.*}}, {{.*}} : i32
        // CHECK: %[[scalarPtrUpdateLoop:.*]] = tt.addptr %[[loopScalarPtr]], %[[scalarOffsetLoop]]
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[variableOffset]]
        // CHECK: %[[offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtrUpdateLoop]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[offset_i]]
        // CHECK: {{.*}} = tt.load %[[newPtr]]
        // CHECK: scf.yield {{.*}}, {{.*}}, %[[scalarPtrUpdateLoop]], %[[offset_i]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
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
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[forOut:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr:.*]] = %arg0, %[[loopOffset:.*]] = %[[base_offset]])
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[scalarOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
        // CHECK: %[[scalarPtrUpdate:.*]] = tt.addptr %[[scalarPtr]], %[[scalarOffset]]
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[variableOffset]]
        // CHECK: %[[ext_offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtrUpdate]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[ext_offset_i]]
        // CHECK: tt.load %[[newPtr]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    // CHECK: %[[finalScalarOffset:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
    // CHECK: %[[scalarPtrFinalUpdate:.*]] = tt.addptr %[[forOut]]#2, %[[finalScalarOffset]]
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[variableOffset]]
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
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>

    // CHECK: %[[forOut0:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr0:.*]] = %arg0, %[[loopOffset0:.*]] = %[[base_offset]]){{.*}}{
    // CHECK: %[[forOut1:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr1:.*]] = %[[scalarPtr0]], %[[loopOffset1:.*]] = %[[loopOffset0]]){{.*}}{
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
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
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[baseOffsetVariable:.*]] = tt.splat {{.*}} : i64 -> tensor<1024xi64, #blocked>
    // CHECK: %[[ifOut:.*]]:3 = scf.if {{.*}} -> (tensor<1024x!tt.ptr<f32>, #blocked>, !tt.ptr<f32>, tensor<1024xi64, #blocked>)
    %6 = scf.if %cond -> (tensor<1024x!tt.ptr<f32>, #blocked>){
        // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
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
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
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
  // CHECK-LABEL  tt.func @branch
  tt.func @branch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[variableOffset:.*]] = tt.make_range
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[scalarOffset:.*]] = arith.addi {{.*}}, {{.*}} : i32
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

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @tile_ptr(%arg1: !tt.ptr<f16>,  %arg5: i32 , %arg7: i32 )  {
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
    // CHECK: %[[pid:.*]] = tt.get_program_id x : i32
    // CHECK: %[[pidOffset:.*]] = arith.muli %[[pid]], {{.*}}
    // CHECK: %[[variableOffset0:.*]] = tt.make_range
    // CHECK: %[[variableOffset1:.*]] = arith.muli %{{.*}}, %{{.*}} : tensor<16x1xi32, #blocked>
    // CHECK: %[[variableOffset00:.*]] = tt.expand_dims %[[variableOffset0]] {{.*}} -> tensor<1x256xi32, #blocked>
    // CHECK: %[[variableOffset10:.*]] = tt.broadcast %[[variableOffset1]] : tensor<16x1xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[variableOffset01:.*]] = tt.broadcast %[[variableOffset00]] : tensor<1x256xi32, #blocked> -> tensor<16x256xi32, #blocked>
    // CHECK: %[[variableOffset2:.*]] = arith.addi %[[variableOffset10]], %[[variableOffset01]]
    // CHECK: %[[tilePtrTotalOffset:.*]] = arith.addi %{{.*}}, %[[pidOffset]] : i32
    // CHECK: %[[tilePtr:.*]] = tt.addptr %arg0, %[[tilePtrTotalOffset]] : !tt.ptr<f16>, i32
    // CHECK: %[[variableOffset3:.*]] = arith.extsi %[[variableOffset2]]
    // CHECK: %[[variableOffset4:.*]] = arith.addi %[[variableOffset3]], {{.*}}
    // CHECK: %[[variableOffset5:.*]] = arith.trunci %[[variableOffset4]]
    // CHECK: %[[tilePtrSplat:.*]] = tt.splat %[[tilePtr]]
    // CHECK: %[[ptr:.*]] = tt.addptr %[[tilePtrSplat]], %[[variableOffset5]]
    // CHECK: tt.load %[[ptr]]
    %61 = tt.load %46 : tensor<16x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
