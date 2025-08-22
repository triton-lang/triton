// RUN: triton-opt %s --triton-nvidia-interleave-tmem --allow-unregistered-dialect | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100"} {

tt.func public @sink_load(%arg0: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>,
                          %arg1: tensor<128x128xf16, #blocked>,
                          %arg2: !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
                          -> (tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>) {

  // CHECK: ttg.local_alloc
  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: arith.truncf
  %subslice0 = ttng.tmem_subslice %arg0 {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %subtile0 = ttng.tmem_load %subslice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
  %outLHS = ttg.convert_layout %subtile0 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked>
  %subslice1 = ttng.tmem_subslice %arg0 {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
  %subtile1 = ttng.tmem_load %subslice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
  %outRHS = ttg.convert_layout %subtile1 : tensor<128x64xf32, #blocked1> -> tensor<128x64xf32, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: ttng.tmem_store
  // CHECK: arith.truncf
  %4 = ttg.local_alloc %arg1 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
  %5 = arith.truncf %outLHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  %true = arith.constant true
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  ttng.tmem_store %cst, %arg2, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %6 = arith.truncf %outRHS : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>

  // CHECK: ttng.tmem_load
  // CHECK: ttg.convert_layout
  // CHECK: "unknow_may_side_effect"() : () -> ()
  // CHECK: arith.truncf
  %7 = ttng.tmem_load %arg2 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
  %8 = ttg.convert_layout %7 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked>
  "unknow_may_side_effect"() : () -> ()
  %9 = arith.truncf %8 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>

  ttg.local_dealloc %4 : !ttg.memdesc<128x128xf16, #shared, #smem>
  tt.return %5, %6, %9 : tensor<128x64xf16, #blocked>, tensor<128x64xf16, #blocked>, tensor<128x128xf16, #blocked>
}

// CHECK-LABEL: @interleave_load_store_ws
tt.func @interleave_load_store_ws() {
  %0 = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>)
  ttg.warp_specialize(%0)
  default{
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%arg0: !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) num_warps(8) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c32 = arith.constant 32 : i32
    %alpha = arith.constant dense<0.5> : tensor<128x64xf32, #blocked1>
    %true = arith.constant true

    // CHECK: scf.for
    scf.for %i = %c0 to %c32 step %c1 : i32 {
      // CHECK: memdesc_index
      %cur_acc = ttg.memdesc_index %arg0[%i] : !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[S0:%.+]] = ttng.tmem_subslice %{{.+}} {N = 0 : i32}
      // CHECK-NEXT: [[S1:%.+]] = ttng.tmem_subslice %{{.+}} {N = 64 : i32}

      // CHECK-NEXT: [[L0:%.+]] = ttng.tmem_load [[S0]]
      // CHECK-NEXT: [[M0:%.+]] = arith.mulf [[L0]]
      // CHECK-NEXT: ttng.tmem_store [[M0]], [[S0]]
      %slice0 = ttng.tmem_subslice %cur_acc {N = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %val0 = ttng.tmem_load %slice0 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
      %mul0 = arith.mulf %val0, %alpha : tensor<128x64xf32, #blocked1>

      // CHECK-NEXT: [[L1:%.+]] = ttng.tmem_load [[S1]]
      // CHECK-NEXT: [[M1:%.+]] = arith.mulf [[L1]]
      // CHECK-NEXT: ttng.tmem_store [[M1]], [[S1]]
      %slice1 = ttng.tmem_subslice %cur_acc {N = 64 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %val1 = ttng.tmem_load %slice1 : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x64xf32, #blocked1>
      %mul1 = arith.mulf %val1, %alpha : tensor<128x64xf32, #blocked1>

      ttng.tmem_store %mul0, %slice0, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>
      ttng.tmem_store %mul1, %slice1, %true : tensor<128x64xf32, #blocked1> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>

    }
    ttg.warp_return
  } : (!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>) -> ()
  tt.return
}

// CHECK-LABEL: @arrive_barrier
tt.func @arrive_barrier(%arg0: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
  %true = arith.constant true
  %cst = arith.constant dense<0.0> : tensor<128x128xf32, #blocked1>

  // CHECK-COUNT-2: ttng.tmem_alloc
  %alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %noalias_alloc = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store
  // CHECK-NEXT: tmem_load
  %0 = ttng.tmem_load %alloc : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
  ttng.tmem_store %cst, %noalias_alloc, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: arrive_barrier
  ttng.arrive_barrier %arg0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
  "user"(%0) : (tensor<128x128xf32, #blocked1>) -> ()
  tt.return
}

// CHECK-LABEL: @sink_alloc_op
tt.func @sink_alloc_op(%arg0: tensor<128x128xf32, #blocked1>) {
  %c0 = arith.constant 0 : i32
  %true = arith.constant true

  %alloc0 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %subview0 = ttg.memdesc_index %alloc0[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[ALLOC1:%.+]] = ttng.tmem_alloc
  %alloc1 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK: [[SUBVIEW1:%.+]] = ttg.memdesc_index [[ALLOC1]]
  %subview1 = ttg.memdesc_index %alloc1[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW1]]
  ttng.tmem_store %arg0, %subview1, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ALLOC0:%.+]] = ttng.tmem_alloc
  // CHECK: [[SUBVIEW0:%.+]] = ttg.memdesc_index [[ALLOC0]]
  // CHECK-NEXT: tmem_store %arg0, [[SUBVIEW0]]
  ttng.tmem_store %arg0, %subview0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  tt.return
}

}
