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
     // CHECK: %[[scalarPtr:.*]] = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
     // CHECK: %[[offset_32bit:.*]] = arith.trunci %{{.*}} : tensor<1024xi64, #blocked> to tensor<1024xi32, #blocked>
     // CHECK: %[[basePtr:.*]] = tt.splat %[[scalarPtr]]
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[offset_32bit]]
     // CHECK  tt.load %[[newPtr]]
     %6 = tt.addptr %5, %3 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %7 : tensor<1024xf32, #blocked>
  }

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
     // CHECK: %[[offset064bit:.*]] = arith.extsi {{.*}}
     // CHECK: %[[offset164bit:.*]] = arith.addi %[[offset064bit]], %[[baseOffset064bit]]
     // CHECK: %[[offset132bit:.*]] = arith.trunci %[[offset164bit]] : tensor<1024xi64, #blocked> to tensor<1024xi32, #blocked>
     // CHECK: %[[basePtr:.*]] = tt.splat %arg0
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[offset132bit]]
     // CHECK: tt.load %[[newPtr]]
     %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %7 : tensor<1024xf32, #blocked>
  }

  // CHECK-LABEL: tt.func @conversion3
  tt.func @conversion3(%arg0: !tt.ptr<f32>)-> tensor<1024xf32, #blocked>{
     %c1024_i32 = arith.constant 1024 : i32
     %0 = tt.get_program_id x : i32
     %1 = arith.muli %0, %c1024_i32 : i32
     %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
     %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
     %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
     // CHECK  %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
     // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
     // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
     // CHECK: %[[ext_offset00:.*]] = arith.extsi %[[offset0]]
     // CHECK: %[[newOffset0:.*]] = arith.addi %[[ext_offset00]], %[[base_offset]]
     // CHECK: %[[ext_offset01:.*]] = arith.extsi %[[offset0]]
     // CHECK: %[[newOffset1:.*]] = arith.addi %[[ext_offset01]], %[[newOffset0]]
     // CHECK: %[[basePtr:.*]] = tt.splat %arg0 : !tt.ptr<f32>
     // CHECK: %[[newPtr:.*]] = tt.addptr %[[basePtr]], %[[newOffset1]]
     // CHECK: {{.*}} = tt.load %[[newPtr]]
     %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
     %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %7 = tt.addptr %6, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
     %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>, #blocked>
     tt.return %8 : tensor<1024xf32, #blocked>
  }

  // CHECK-LABEL: tt.func @forOp
  tt.func @forOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}}
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[offset0]]
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: {{.*}}:4 = scf.for {{.*}} iter_args({{.*}}, {{.*}}, %[[scalarPtr:.*]] = %arg0, %[[loopOffset:.*]] = %[[offset1]]
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %6, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[offset0]]
        // CHECK: %[[offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtr]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[offset_i]]
        // CHECK: {{.*}} = tt.load %[[newPtr]]
        // CHECK: scf.yield {{.*}}, {{.*}}, %[[scalarPtr]], %[[offset_i]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }

  // CHECK-LABEL: tt.func @forOp2
  tt.func @forOp2(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[forOut:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr:.*]] = %arg0, %[[loopOffset:.*]] = %[[base_offset]])
    %52:2 = scf.for %arg9 = %c0 to %c128 step %c1 iter_args(%arg1 = %5, %arg2 = %init) -> (tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>){
        // CHECK: %[[ext_offset0i:.*]] = arith.extsi %[[offset0]]
        // CHECK: %[[ext_offset_i:.*]] = arith.addi %[[ext_offset0i]], %[[loopOffset]]
        // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtr]]
        // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[ext_offset_i]]
        // CHECK: tt.load %[[newPtr]]
        %11 = tt.addptr %arg1, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        %9 = tt.load %11 : tensor<1024x!tt.ptr<f32>, #blocked>
        %10 = arith.addf %9, %arg2 : tensor<1024xf32, #blocked>
        scf.yield %11, %10 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xf32, #blocked>
    }
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[offset0]]
    // CHECK: %[[tailOffset:.*]] = arith.addi %[[ext_offset0]], %[[forOut]]#3
    // CHECK: %[[tail_base_ptr:.*]] = tt.splat %[[forOut]]#2
    // CHECK: %[[tailPtr:.*]] = tt.addptr %[[tail_base_ptr]], %[[tailOffset]]
    // CHECK: tt.load %[[tailPtr]]
    %8 = tt.addptr %52#0, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    %11 = tt.load %8 : tensor<1024x!tt.ptr<f32>, #blocked>
    tt.return %11 : tensor<1024xf32, #blocked>
  }

  // CHECK-LABEL: tt.func @forNested
  tt.func @forNested(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0:.*]] = arith.addi {{.*}}, {{.*}}
    // CHECK: %[[base_offset:.*]] = tt.splat {{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>

    // CHECK: %[[forOut0:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr0:.*]] = %arg0, %[[loopOffset0:.*]] = %[[base_offset]]){{.*}}{
    // CHECK: %[[forOut1:.*]]:4 = scf.for {{.*}} iter_args(%{{.*}}, {{.*}}, %[[scalarPtr1:.*]] = %[[scalarPtr0]], %[[loopOffset1:.*]] = %[[loopOffset0]]){{.*}}{
    // CHECK: %[[ext_loop_offset1:.*]] = arith.extsi %[[offset0]]
    // CHECK: %[[offset_i:.*]] = arith.addi %[[ext_loop_offset1]], %[[loopOffset1]]
    // CHECK: %[[base_ptr:.*]] = tt.splat %[[scalarPtr1]]
    // CHECK: %[[newPtr:.*]] = tt.addptr %[[base_ptr]], %[[offset_i]]
    // CHECK: tt.load %[[newPtr]]
    // CHECK: scf.yield %{{.*}}, {{.*}}, %[[scalarPtr1]], %[[offset_i]]
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

  // CHECK-LABEL: tt.func @ifOp
  tt.func @ifOp(%arg0: !tt.ptr<f32>, %init : tensor<1024xf32, #blocked>, %cond : i1)-> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0]] = arith.addi
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[ifOut:.*]]:3 = scf.if {{.*}} -> (tensor<1024x!tt.ptr<f32>, #blocked>, !tt.ptr<f32>, tensor<1024xi64, #blocked>)
    %6 = scf.if %cond -> (tensor<1024x!tt.ptr<f32>, #blocked>){
        // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[offset0]]
        // CHECK: %[[if_offset:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
        // CHECK: scf.yield %{{.*}}, %arg0, %[[if_offset]]
        %true = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
        scf.yield %true : tensor<1024x!tt.ptr<f32>, #blocked>
    } else {
        // CHECK: %[[new_scalar_ptr:.*]] = tt.addptr %arg0, {{.*}}
        // CHECK: scf.yield %{{.*}}, %[[new_scalar_ptr]], %[[base_offset]]
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

  // CHECK-LABEL  tt.func @condBranch
  tt.func @condBranch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0]] = arith.addi
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[offset0]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: cf.cond_br {{.*}}, ^bb1(%{{.*}}, %arg0, %[[base_offset]] : {{.*}}), ^bb2(%{{.*}}, %arg0, %[[offset1]] : {{.*}})
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

  // CHECK-LABEL  tt.func @branch
  tt.func @branch(%arg0 : !tt.ptr<f32>, %i1 : i1) -> tensor<1024xf32, #blocked>{
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0: index
    %c128 = arith.constant 128: index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked>
    // CHECK: %[[offset0]] = arith.addi
    // CHECK: %[[base_offset:.*]] = tt.splat %{{.*}} : i64
    // CHECK: %[[ext_offset0:.*]] = arith.extsi %[[offset0]]
    %3 = tt.splat %1 : i32 -> tensor<1024xi32, #blocked>
    %4 = arith.addi %3, %2 : tensor<1024xi32, #blocked>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>, #blocked>
    // CHECK: %[[offset1:.*]] = arith.addi %[[ext_offset0]], %[[base_offset]]
    %6 = tt.addptr %5, %4 : tensor<1024x!tt.ptr<f32>, #blocked>, tensor<1024xi32, #blocked>
    // CHECK: cf.br ^bb1(%{{.*}}, %arg0, %[[offset1]] : {{.*}})
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
