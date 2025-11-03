// RUN: triton-opt %s -split-input-file --tritonamdgpu-update-async-wait-count=arch-generation-name=gfx950 | FileCheck %s

// The number in SSA symbolic names represents the number of generated async load operation at assembly level a ttg.async_copy_global_to_local will generate, which is counted by this pass.
// For example `ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst ..` will generate two global_load_async_to_lds_b128 assembly instruction

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: simple_waitcnt
  tt.func public @simple_waitcnt(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emit 1 instruction
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    // Emits 2 instructions
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    // CHECK: amdgpu.async_wait {num_inst = 0
    ttg.async_wait {num = 0 : i32}
    // CHECK: amdgpu.async_wait {num_inst = 2
    ttg.async_wait {num = 1 : i32}
    // Check we stop at function boundary
    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 2 : i32}
    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 3 : i32}

    tt.return
  }

  // CHECK-LABEL: simple_waitcnt_non_committed_async_ops
  tt.func public @simple_waitcnt_non_committed_async_ops(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Emit 1 instruction
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>

    // We expect 1 because the async copy above has not been committed yet
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 0 : i32}
    // -1 can be used to wait on all, even non committed async ops
    // CHECK: amdgpu.async_wait {num_inst = 0
    ttg.async_wait {num = -1 : i32}

    tt.return
  }

  // CHECK-LABEL: wait_if_without_else
  tt.func public @wait_if_without_else(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    // Ensure we look into then but also skip the if if no else is present

    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    scf.if %cond {
      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
    }
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 1: i32}

    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    scf.if %cond {
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      scf.yield
    }
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 1: i32}

    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 2: i32}


    tt.return
  }

  // CHECK-LABEL wait_if_with_else
  tt.func public @wait_if_with_else(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    scf.if %cond {
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      scf.yield
    } else {
      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      scf.yield
    }
    ttg.async_commit_group
    // Ensure we use the branch with less instructions (then)
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 1: i32}
    // Check we do not loop in an if but instead continue upwards
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 2: i32}

    scf.if %cond {
      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      scf.yield
    } else {
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      scf.yield
    }
    ttg.async_commit_group
    // Ensure we use the branch with less instructions (else)
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 1: i32}

    tt.return
  }

  // CHECK-LABEL: check_wait_nested_ifs
  tt.func public @check_wait_nested_ifs(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {
    scf.if %cond {
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      scf.if %cond {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        scf.yield
      } else {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        scf.yield
      }
      ttg.async_commit_group
      scf.yield
    } else {
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      scf.if %cond {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        scf.yield
      } else {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
        scf.yield
      }
      ttg.async_commit_group
      scf.yield
    }
    // The shortest path (else->then) contains 2 async ops -> instruction count 2
    // CHECK: amdgpu.async_wait {num_inst = 2
    ttg.async_wait {num = 1: i32}

    tt.return
  }

  //CHECK-LABEL: for_without_async_ops
  tt.func public @for_without_async_ops(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 iter_args() -> () : i32 {
      // CHECK: amdgpu.async_wait {num_inst = 1
      ttg.async_wait {num = 1: i32}
      scf.yield
    }
    // CHECK: amdgpu.async_wait {num_inst = 1
    ttg.async_wait {num = 1: i32}

    tt.return
  }

  //CHECK-LABEL: for_with_async_ops
  tt.func public @for_with_async_ops(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    // CHECK: amdgpu.async_wait {num_inst = 6
    ttg.async_wait {num = 3: i32}

    scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 : i32 {
      // The minimum it waits are 3 loop iteration with 1 instructions per iteration. Note the prologue would lead to 6
      // CHECK: amdgpu.async_wait {num_inst = 3
      ttg.async_wait {num = 3: i32}
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      scf.yield
    }
    // The minimum it waits are 3 loop iteration with 1 instructions per iteration. Note the prologue would lead to 6
    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 3: i32}

    tt.return
  }

  //CHECK-LABEL: for_nested_control_flow
  tt.func public @for_nested_control_flow(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    // Prologue: 2 instructions per commit group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group

    // The loop has 3 commits group which produce 2,1,1 (in program order) async instructions
    scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 : i32 {
      // 2 full loop iterations => 8
      // CHECK: amdgpu.async_wait {num_inst = 8
      ttg.async_wait {num = 6: i32}

      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group

      // Wait on 1 full loop iteration (4) + the commit group above (2)
      // CHECK: amdgpu.async_wait {num_inst = 6
      ttg.async_wait {num = 4: i32}

      scf.if %cond {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      } else {
        ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      }
      ttg.async_commit_group

      scf.if %cond {
        ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      } else {
        ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      }
      ttg.async_commit_group

      // Wait on 1 full loop iteration (4) + the commit group above (1)
      // CHECK: amdgpu.async_wait {num_inst = 5
      ttg.async_wait {num = 4: i32}

      scf.yield
    }
    // 2 Full loop iterations (2 * 4)
    // CHECK: amdgpu.async_wait {num_inst = 8
    ttg.async_wait {num = 6: i32}

    tt.return
  }

  // CHECK-LABEL: while_without_async_ops
  tt.func public @while_without_async_ops(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    // Check we are not getting stuck in loops with no async ops
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    %69 = scf.while (%arg10 = %cond) : (i1) -> (i1) {
      // CHECK: amdgpu.async_wait {num_inst = 2
      ttg.async_wait {num = 1: i32}
      scf.condition(%arg10) %arg10 : i1
    } do {
    ^bb0(%arg12: i1):
      // CHECK: amdgpu.async_wait {num_inst = 2
      ttg.async_wait {num = 1: i32}
      scf.yield %arg12 : i1
    }
    // CHECK: amdgpu.async_wait {num_inst = 2
    ttg.async_wait {num = 1: i32}

    tt.return
  }

  // CHECK-LABEL: while_async_op_in_before_block
  tt.func public @while_async_op_in_before_block(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    // Check we are following control flow and count inside the before block
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    // CHECK: amdgpu.async_wait {num_inst = 6
    ttg.async_wait {num = 3: i32}

    %70 = scf.while (%arg10 = %cond) : (i1) -> (i1) {
      // Count before block 3 times
      // CHECK: amdgpu.async_wait {num_inst = 3
      ttg.async_wait {num = 3: i32}
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      scf.condition(%arg10) %arg10 : i1
    } do {
    ^bb0(%arg12: i1):
      // Count before block 3 times
      // CHECK: amdgpu.async_wait {num_inst = 3
      ttg.async_wait {num = 3: i32}
      scf.yield %arg12 : i1
    }
    // Count before block 3 times
    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 3: i32}

    tt.return
  }

  // CHECK-LABEL: while_async_op_in_after_block
  tt.func public @while_async_op_in_after_block(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    // Check we are following control flow and count inside the after block
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    // CHECK: amdgpu.async_wait {num_inst = 6
    ttg.async_wait {num = 3: i32}

    %71 = scf.while (%arg10 = %cond) : (i1) -> (i1) {
      // Count after block 3 times
      // CHECK: amdgpu.async_wait {num_inst = 3
      ttg.async_wait {num = 3: i32}
      scf.condition(%arg10) %arg10 : i1
    } do {
    ^bb0(%arg12: i1):
      ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      // Count after block 4 times
      // CHECK: amdgpu.async_wait {num_inst = 4
      ttg.async_wait {num = 4: i32} // 4 because we moved the wait after the next prefetch
      scf.yield %arg12 : i1
    }
    // Count after block 3 times
    // CHECK: amdgpu.async_wait {num_inst = 3
    ttg.async_wait {num = 3: i32}

    tt.return
  }

  //CHECK-LABEL: nested_loops_and_if
  tt.func public @nested_loops_and_if(
        %cond: i1,
        %arg0: i32,
        %memDesc2Inst: !ttg.memdesc<128x16xf16, #shared, #smem, mutable>,
        %ptr2Inst: tensor<128x16x!tt.ptr<f16>, #blocked>  {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>},
        %memDesc1Inst: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
        %ptr1Inst: tensor<64x16x!tt.ptr<f16>, #blocked> {tt.divisibility = dense<[16, 16]> : tensor<2xi32>, tt.contiguity = dense<[16, 16]> : tensor<2xi32>}) {

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    // CHECK: amdgpu.async_wait {num_inst = 6
    ttg.async_wait {num = 6: i32}

    %70 = scf.while (%arg10 = %cond) : (i1) -> (i1) {
      // Escape while and count prologue = 6
      // CHECK: amdgpu.async_wait {num_inst = 6
      ttg.async_wait {num = 6: i32}
      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      // 2 Instructions
      scf.condition(%arg10) %arg10 : i1
    } do {
    ^bb0(%arg12: i1):
      // 1 commit group in Before-block + 5 commits groups in prologue = 7
      // CHECK: amdgpu.async_wait {num_inst = 7
      ttg.async_wait {num = 6: i32}
      ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
      ttg.async_commit_group
      // 2 Instructions

      scf.for %arg14 = %c0_i32 to %arg0 step %c1_i32 : i32 {
        ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
        // 2 Instructions
        ttg.async_commit_group
        // 1 commit group(2) to escape for, 1 commits group(2) in rest of while after block, 1 commit group (2) in while before block and 3 commits group in prologue = 9
        // CHECK: amdgpu.async_wait {num_inst = 9
        ttg.async_wait {num = 6: i32}

        scf.if %cond {
          ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
          ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>

          // Same as above but we also have to count the 2 async_copies above = 9+3
          // CHECK: amdgpu.async_wait {num_inst = 12
          ttg.async_wait {num = 6: i32}
        } else {
          ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
        }
        // 2 Instructions (else)
        ttg.async_commit_group

        scf.if %cond {
          ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
          ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
          // 3 Instructions
          ttg.async_commit_group
          // 1 commit group (3) in this block, 2 commits group in the rest of the for body (2+2), 1 commits group(2) in rest of while after block, 1 commit group (2) in while before block, 1 commit group (1) in epilogue = 12
          // CHECK: amdgpu.async_wait {num_inst = 12
          ttg.async_wait {num = 6: i32}
        }
        // Same as above but skips the if (first commit group(3)) and instead counts one more in the prologue (1) = 10
        // CHECK: amdgpu.async_wait {num_inst = 10
        ttg.async_wait {num = 6: i32}
        scf.for %arg15 = %c0_i32 to %arg0 step %c1_i32 : i32 {
          ttg.async_copy_global_to_local %ptr1Inst, %memDesc1Inst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
          // 1 Instruction
          ttg.async_commit_group
          ttg.async_copy_global_to_local %ptr2Inst, %memDesc2Inst : tensor<128x16x!tt.ptr<f16>, #blocked> -> <128x16xf16, #shared, #smem, mutable>
          // 2 Instructions
          ttg.async_commit_group
          // Just staying in the loop is the lowest path (3 per iteration and we do 3 iterations)
          // CHECK: amdgpu.async_wait {num_inst = 9
          ttg.async_wait {num = 6: i32}
          scf.yield
        }
        // Just stay in the inner loop for the lowest path
        // CHECK: amdgpu.async_wait {num_inst = 9
        ttg.async_wait {num = 6: i32}
        scf.yield
      }
      scf.yield %arg12 : i1
    }
    // While before-body (2) + 5 prologue groups = 7
    // CHECK: amdgpu.async_wait {num_inst = 7
    ttg.async_wait {num = 6: i32}

    tt.return
  }

}
