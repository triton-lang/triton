// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-test-pipeline-assign-latencies=num-stages=3 -canonicalize | FileCheck %s

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

// CHECK-LABEL: @load_into_lt_4b
tt.func @load_into_lt_4b(%lb : index, %ub : index, %step : index,
                  %a_ptr_init : tensor<128x32x!tt.ptr<f16>, #AL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32},
                  %b_ptr_init : tensor<32x128x!tt.ptr<f16>, #BL> {tt.divisibility = 16 : i32, tt.contiguity = 32 : i32}) -> tensor<128x128xf32, #mma> {
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #mma>
  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>

  %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #mma>) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %a_ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.local_alloc %a_ : (tensor<128x32xf16, #AL>) -> !ttg.memdesc<128x32xf16, #shared, #ttg.shared_memory>
    // Do not pipeline if cp.async would read less than 4 consecutive bytes
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc
tt.func @tc_gen5_mma_overwrite_acc(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false
tt.func @tc_gen5_mma_acc_use_false(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %false, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false
tt.func @tc_gen5_mma_acc_use_false(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %acc_use = arith.xori %acc_use_init, %true : i1
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false_dist_1
tt.func @tc_gen5_mma_acc_use_false_dist_1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step iter_args(%acc_use = %acc_use_init) -> (i1) {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
    %acc_use_next = arith.xori %acc_use, %true : i1
    scf.yield %acc_use_next : i1
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_acc_use_false_outside_loop
tt.func @tc_gen5_mma_acc_use_false_outside_loop(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>,
                  %acc_use_init : i1) -> () {
  %true = arith.constant true
  %false = arith.constant false
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  %acc_use = arith.xori %acc_use_init, %true : i1
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %acc_use, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc_outside_loop
tt.func @tc_gen5_mma_overwrite_acc_outside_loop(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_overwrite_acc
tt.func @tc_gen5_mma_overwrite_acc_small_load(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    // CHECK: tt.load
    // CHECK-NOT: tt.latency
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_B_outside
tt.func @tc_gen5_mma_B_outside(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_disallow_multibuffer
tt.func @tc_gen5_mma_disallow_multibuffer(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B: tensor<128x128xf16, #blocked1>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  } {tt.disallow_acc_multi_buffer}
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_B_outside2
tt.func @tc_gen5_mma_B_outside2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sh: !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>,
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand1
tt.func @tc_gen5_mma_non_load_operand1(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = "producer"() : () -> tensor<128x128xf16, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_non_load_operand2
tt.func @tc_gen5_mma_non_load_operand2(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B_sh = "producer"() : () -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    // CHECK: ttng.tc_gen5_mma
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @select_after_mma
  tt.func public @select_after_mma(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = "cnd"() : () -> i1
    %1 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %1, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32  : i32 {
      %4 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %7 = ttg.local_alloc %6 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma {{.*}} {tt.latency = 1 : i32}
      ttng.tc_gen5_mma %5, %7, %1, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %8 = arith.xori %0, %true : i1
      ttng.tmem_store %cst_0, %1, %8 : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    } {tt.scheduled_max_stage = 3 : i32}
    %2 = ttng.tmem_load %1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %3 = arith.truncf %2 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %3 : tensor<128x128xf16, #blocked1>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled
tt.func @tc_gen5_mma_scaled(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %B_sc_sh = ttg.local_alloc %B_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}} {tt.latency = 1 : i32}
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_sh, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @tc_gen5_mma_scaled_tmem_scales
tt.func @tc_gen5_mma_scaled_tmem_scales(%lb : index, %ub : index, %step : index,
                  %A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %A_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %B_sc_ptr: tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2> {tt.divisibility = 16 : i32, tt.contiguity = 16 : i32},
                  %acc_init : tensor<128x128xf32, #blocked1>) -> () {
  %true = arith.constant true
  %acc_tm = ttng.tmem_alloc %acc_init : (tensor<128x128xf32, #blocked1>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
  scf.for %iv = %lb to %ub step %step : index {
    %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>
    %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
    %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>

    %A_sc = tt.load %A_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %A_sc_sh = ttg.local_alloc %A_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

    %B_sc = tt.load %B_sc_ptr : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %B_sc_tm = ttng.tmem_alloc %B_sc : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x328x4xi8, #tmem_scales, #ttng.tensor_memory>

    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}
    // CHECK-NOT: tt.latency
    ttng.tmem_store %acc_init, %acc_tm, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %acc_tm, %A_sc_sh, %B_sc_tm, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x328x4xi8, #tmem_scales, #ttng.tensor_memory>, i1, i1) -> ()
    %acc_res = ttng.tmem_load %acc_tm : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    "use"(%acc_res) : (tensor<128x128xf32, #blocked1>) -> ()
  }
  tt.return
}
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @block_scale_mxfp_matmul
  tt.func public @block_scale_mxfp_matmul(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<128x256xi32, #blocked1>
    %cst_1 = arith.constant dense<4> : tensor<256x128xi32, #blocked2>
    %cst_2 = arith.constant dense<4> : tensor<1x2x32x4x4xi32, #blocked3>
    %0 = tt.splat %arg3 : !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
    %1 = tt.splat %arg4 : !tt.ptr<f8E5M2> -> tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>
    %2 = tt.splat %arg5 : !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    %3 = tt.splat %arg6 : !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    %4 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %6 = tt.broadcast %5 : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %7 = tt.addptr %0, %6 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
    %8 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %10 = tt.broadcast %9 : tensor<1x128xi32, #blocked2> -> tensor<256x128xi32, #blocked2>
    %11 = tt.addptr %1, %10 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<256x128xi32, #blocked2>
    %12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>}>>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>>
    %15 = tt.expand_dims %14 {axis = 2 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked3}>}>> -> tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked3}>>
    %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked3}>> -> tensor<1x1x1x1x4xi32, #blocked3>
    %17 = tt.broadcast %16 : tensor<1x1x1x1x4xi32, #blocked3> -> tensor<1x2x32x4x4xi32, #blocked3>
    %18 = tt.addptr %2, %17 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
    %19 = tt.addptr %3, %17 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
    %20 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tmem_store %cst, %20, %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %21:4 = scf.for %arg7 = %arg0 to %arg1 step %arg2 iter_args(%arg8 = %7, %arg9 = %11, %arg10 = %18, %arg11 = %19) -> (tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>) {
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %22 = tt.load %arg8 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>
      %23 = ttg.local_alloc %22 : (tensor<128x256xf8E5M2, #blocked1>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #smem>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %24 = tt.load %arg9 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>
      %25 = ttg.local_alloc %24 : (tensor<256x128xf8E5M2, #blocked2>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %26 = tt.load %arg10 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %27 = tt.load %arg11 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
      %28 = ttg.local_alloc %26 : (tensor<1x2x32x4x4xi8, #blocked3>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %29 = ttg.local_alloc %27 : (tensor<1x2x32x4x4xi8, #blocked3>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      // CHECK: ttng.tc_gen5_mma_scaled
      // No uses in the loop, so no latency is assigned.
      // CHECK-NOT: tt.latency
      ttng.tc_gen5_mma_scaled %23, %25, %20, %28, %29, %true, %true lhs = e5m2 rhs = e5m2 : (!ttg.memdesc<128x256xf8E5M2, #shared, #smem>, !ttg.memdesc<256x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, i1, i1) -> ()
      %30 = tt.addptr %arg8, %cst_0 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<128x256xi32, #blocked1>
      %31 = tt.addptr %arg9, %cst_1 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<256x128xi32, #blocked2>
      %32 = tt.addptr %arg10, %cst_2 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
      %33 = tt.addptr %arg11, %cst_2 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4xi32, #blocked3>
      scf.yield %30, %31, %32, %33 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked3>
    } {tt.num_stages = 3 : i32}
    tt.return %cst : tensor<128x128xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @two_dots
  // Temporarily disable pipelining of loops with multiple dot ops.
  tt.func public @two_dots(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg4: i32) attributes {noinline = false} {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32  : i32 {
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %2 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %3 = ttg.local_alloc %2 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      // CHECK: tt.load {{.*}} {tt.latency = 2 : i32}
      %4 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %5 = ttg.local_alloc %4 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %6 = tt.load %arg2 : tensor<128x128x!tt.ptr<f32>, #blocked1>
      ttng.tmem_store %6, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NOT: ttng.tc_gen5_mma {{.*}} {tt.latency
      ttng.tc_gen5_mma %3, %5, %0, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %7 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      ttng.tmem_store %7, %1, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NOT: ttng.tc_gen5_mma {{.*}} {tt.latency
      ttng.tc_gen5_mma %3, %5, %1, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %8 = ttng.tmem_load %1 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      tt.store %arg3, %8 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABLE: @chained_dot_scaled_acc
  tt.func public @chained_dot_scaled_acc(%arg0: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg1: tensor<128x128x!tt.ptr<f16>, #blocked> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg2: i32) -> tensor<128x128xf16, #blocked1> attributes {noinline = false} {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked1>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttng.tmem_alloc  : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %1 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<128x128xf32, #blocked1>)  : i32 {
      %3 = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %4 = ttg.local_alloc %3 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %5 = tt.load %arg1 : tensor<128x128x!tt.ptr<f16>, #blocked>
      %6 = ttg.local_alloc %5 : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      %7 = arith.mulf %arg4, %cst_0 : tensor<128x128xf32, #blocked1>
      ttng.tmem_store %7, %0, %true : tensor<128x128xf32, #blocked1> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Do not assign latency the the mma, as it is being loaded an modified, so
      // the next instance needs to wait for the result of this one.
      // CHECK-NOT: ttng.tc_gen5_mma {{.*}} {tt.latency
      ttng.tc_gen5_mma %4, %6, %0, %true, %true : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
      %8 = ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
      scf.yield %8 : tensor<128x128xf32, #blocked1>
    }
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    tt.return %2 : tensor<128x128xf16, #blocked1>
  }
}
