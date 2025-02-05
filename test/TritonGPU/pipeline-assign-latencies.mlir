// RUN: triton-opt %s -split-input-file -tritongpu-test-pipeline-assign-latencies=num-stages=3 -canonicalize | FileCheck %s

#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 128, 32]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @default_stages
tt.func @default_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @small_load
// We should *not* assign latency to the load of b_ptr.
tt.func @small_load(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL>) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}}
    // CHECK-NOT: tt.latency
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @load_into_shared
tt.func @load_into_shared(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared1, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @load_into_shared_incompat_layout
tt.func @load_into_shared_incompat_layout(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // CHECK: tt.load
    // CHECK-NOT: {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.local_alloc %b_ : (tensor<32x128xf16, #BL>) -> !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory>

    %c = ttng.warp_group_dot %a, %b, %prev_c {maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<32x128xf16, #shared2, #ttg.shared_memory> -> tensor<128x128xf32, #mma>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>
  }
  tt.return %loop#2: tensor<128x128xf32, #mma>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#2: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load
tt.func @indirect_load(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @mixed_loads
tt.func @mixed_loads(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:4 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop#3: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @per_loop_stages
tt.func @per_loop_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> (tensor<128x128xf32, #C>, tensor<128x128xf32, #C>) {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop_cust_stages:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 3 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 4 : i32}

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }
  tt.return %loop_cust_stages#2, %loop#2: tensor<128x128xf32, #C>, tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_cust_stages
tt.func @indirect_load_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 5 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @indirect_load_few_stages
tt.func @indirect_load_few_stages(%lb : index, %ub : index, %step : index,
                  %a_ind_ptr_init : tensor<128x32x!tt.ptr<i32>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ind_ptr_init : tensor<32x128x!tt.ptr<i32>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_ind_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_ind_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:5 = scf.for %iv = %lb to %ub step %step iter_args(%a_ind_ptr = %a_ind_ptr_init, %b_ind_ptr = %b_ind_ptr_init, %a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_off = tt.load %a_ind_ptr : tensor<128x32x!tt.ptr<i32>, #AL>
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %b_off = tt.load %b_ind_ptr : tensor<32x128x!tt.ptr<i32>, #BL>
    %next_a_ind_ptr = tt.addptr %a_ind_ptr, %a_ind_off : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ind_ptr = tt.addptr %b_ind_ptr, %b_ind_off : tensor<32x128x!tt.ptr<i32>, #BL>, tensor<32x128xi32, #BL>
    %next_a_ptr = tt.addptr %a_ptr, %a_off {tt.divisibility = dense<16> : tensor<128x32xi32>, tt.contiguity = dense<32> : tensor<128x32xi32>, tt.constancy = dense<1> : tensor<128x32xi32>} : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off {tt.divisibility = dense<16> : tensor<32x128xi32>, tt.contiguity = dense<32> : tensor<32x128xi32>, tt.constancy = dense<1> : tensor<32x128xi32>} : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %a_ = tt.load %next_a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 1 : i32}
    %b_ = tt.load %next_b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b = ttg.convert_layout %b_ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>
    scf.yield %next_a_ind_ptr, %next_b_ind_ptr, %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<i32>, #AL>, tensor<32x128x!tt.ptr<i32>, #BL>, tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 2 : i32}
  tt.return %loop#4: tensor<128x128xf32, #C>
}

// CHECK-LABEL: @non_dot_pipeline
tt.func @non_dot_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @no_pipeline
tt.func @no_pipeline(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x32xf16, #A> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x32xf16, #A>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>

  %loop:2 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>) {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    %c = arith.addf %a, %prev_c : tensor<128x32xf16, #A>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    scf.yield %next_a_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xf16, #A>
  }
  tt.return %loop#1: tensor<128x32xf16, #A>
}

// CHECK-LABEL: @intermediate_use
tt.func @intermediate_use_cust_stages(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #C> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %c2 = arith.constant dense<2.00> : tensor<32x128xf16, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %b_ = tt.load %b_ptr : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_2 = arith.mulf %b_ , %c2 : tensor<32x128xf16, #BL>
    %b = ttg.convert_layout %b_2 : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>

    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  } {tt.num_stages = 3 : i32}
  tt.return %loop#2: tensor<128x128xf32, #C>
}

}
