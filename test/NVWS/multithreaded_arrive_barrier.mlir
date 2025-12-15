// RUN: triton-opt --split-input-file --allow-unregistered-dialect --tritongpu-hoist-tmem-alloc --tritongpu-assign-latencies --tritongpu-schedule-loops --tritongpu-automatic-warp-specialization=num-stages=3 --tritongpu-pipeline=num-stages=3 --tritongpu-optimize-partition-warps --tritongpu-combine-tensor-select-and-if --tritongpu-hoist-tmem-alloc=hoist-out-of-if=true --triton-nvidia-gpu-remove-tmem-tokens --tritongpu-combine-tensor-select-and-if --tritongpu-allocate-warp-groups --convert-scf-to-cf --allocate-shared-memory-nv="compute-capability=100 ptx-version=86" --triton-tensor-memory-allocation --tritongpu-global-scratch-memory-allocation --triton-nvidia-gpu-proxy-fence-insertion=compute-capability=100 --convert-triton-gpu-to-llvm="compute-capability=100 ptx-version=86" %s | FileCheck %s

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #acc_layout
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
// Minimal encodings for a single MMA so the scheduler can split load/compute.

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @auto_simple(
  %a_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x128xf16, #shared>>,
  %k_tiles: i32
) {
  // Basic scalars and the zero init for the accumulator carried through the loop.
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void

// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void

// CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];", "r" %2954 : (!llvm.struct<(ptr<3>, i32)>) -> !llvm.void
// CHECK: "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 32768;", "b,r" %663, %654 : (i1, !llvm.ptr<3>) -> !llvm.void



  // Single-tile loop so automatic warp specialization can form producer/consumer partitions.
  %final = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
    // TMA-style descriptor loads give the pass a producer partition to hoist.
    %a = tt.descriptor_load %a_desc[%k, %k] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #oper_layout>
    // Move operands into shared memory so the MMA sees explicit smem inputs.
    %a_shared = ttg.local_alloc %a : (tensor<128x128xf16, #oper_layout>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    // Simple on-chip compute without TMEM to keep conversion patterns minimal.
    %a_reg = ttg.local_load %a_shared : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #oper_layout>
    %a_f32 = arith.extf %a_reg : tensor<128x128xf16, #oper_layout> to tensor<128x128xf32, #oper_layout>
    %c = arith.addf %a_f32, %a_f32 : tensor<128x128xf32, #oper_layout>
    // Dummy user to keep the compute live in the consumer partition.
    "consumer"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}

  // Final sink after the loop prevents DCE and anchors partitioned outputs.
  "final_use"(%final) : (tensor<128x128xf32, #acc_layout>) -> ()

// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 1;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void

// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void
// CHECK: "@$0 mbarrier.init.shared::cta.b64 [$1], 32;", "b,r" %{{.*}}, %{{.*}} : (i1, !llvm.ptr<3>) -> !llvm.void

// CHECK: "mbarrier.arrive.shared::cta.b64 _, [$0];", "r" %2954 : (!llvm.struct<(ptr<3>, i32)>) -> !llvm.void
// CHECK: "@$0 mbarrier.arrive.expect_tx.shared::cta.b64 _, [$1], 32768;", "b,r" %663, %654 : (i1, !llvm.ptr<3>) -> !llvm.void

  // Single-tile loop so automatic warp specialization can form producer/consumer partitions.
  %final_2 = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
    // TMA-style descriptor loads give the pass a producer partition to hoist.
    %a = tt.descriptor_load %a_desc[%k, %k] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #oper_layout>
    // Move operands into shared memory so the MMA sees explicit smem inputs.
    %a_shared = ttg.local_alloc %a : (tensor<128x128xf16, #oper_layout>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
    // Simple on-chip compute without TMEM to keep conversion patterns minimal.
    %a_reg = ttg.local_load %a_shared : !ttg.memdesc<128x128xf16, #shared, #smem> -> tensor<128x128xf16, #oper_layout>
    %a_f32 = arith.extf %a_reg : tensor<128x128xf16, #oper_layout> to tensor<128x128xf32, #oper_layout>
    %c = arith.addf %a_f32, %a_f32 : tensor<128x128xf32, #oper_layout>
    // Dummy user to keep the compute live in the consumer partition.
    "consumer"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}

  // Final sink after the loop prevents DCE and anchors partitioned outputs.
  "final_use"(%final_2) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

}
