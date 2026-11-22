// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups | FileCheck %s
// RUN: triton-opt %s -split-input-file --tritongpu-allocate-warp-groups="instrumented=1" | FileCheck %s --check-prefix=INSTRUMENTED

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {
}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 20 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @kernel() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 18, 4, 12, 16, 19>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  }
  partition2() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition3() num_warps(2)
  // CHECK: partition4() num_warps(1)
  tt.return
}

}

// -----

// CHECK: module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 16 : i32}
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @two_warp_specialize() {
  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 12, 14, 4, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  // CHECK: partition2() num_warps(8)
  // CHECK: partition3() num_warps(1)

  // CHECK: ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 14, 4, 12, 15>}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(8) {
    ttg.warp_return
  } : () -> ()

  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 168 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @setmaxnreg() {
  // CHECK: actualRegisters = array<i32: 208, 80, 80, 80>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 48, 80, 48>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  }
  partition1() num_warps(2) {
    ttg.warp_return
  }
  partition2() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 40 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 40 : i32} {

tt.func @assert_and_print_min_registers() {
  // CHECK: actualRegisters = array<i32: 48, 32>
  // CHECK: requestedRegisters = array<i32: 24>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
  default {
    %zero = arith.constant 0 : i32
    tt.print "zero: " {hex = false, isSigned = array<i32: 1>} : %zero : i32
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %true = arith.constant true
    tt.assert %true, "assert text" : i1
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// INSTRUMENTED: module attributes {instrumented.test, ttg.maxnreg = 40 : i32
module attributes {"instrumented.test", "ttg.num-warps" = 4 : i32, ttg.maxnreg = 40 : i32} {

tt.func @instrumented_min_registers() {
  // INSTRUMENTED: actualRegisters = array<i32: 48, 32>
  // INSTRUMENTED: requestedRegisters = array<i32: 16>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 16>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 128 : i32
module attributes {"ttg.num-warps" = 8 : i32} {

tt.func @steal_from_default() {
  // CHECK: actualRegisters = array<i32: 64, 192>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 192>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 168 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 168 : i32} {

tt.func @cap_default_and_redistribute_surplus() {
  // CHECK: actualRegisters = array<i32: 256, 200, 48>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 88, 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition1() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 256 : i32
module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cap_default_single_worker_full_budget() {
  // CHECK: actualRegisters = array<i32: 256, 256, 256, 256>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24, 24, 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition1() num_warps(1) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition2() num_warps(2) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 224 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 224 : i32} {

tt.func @cap_default_overflow_to_second_warpgroup() {
  // CHECK: actualRegisters = array<i32: 256, 256, 160>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 200, 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition1() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 168 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 168 : i32} {

tt.func @cap_default_prefers_active_over_padding() {
  // CHECK: actualRegisters = array<i32: 168, 168, 168>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 168, 168>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition1() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()

  // CHECK: actualRegisters = array<i32: 256, 224, 24>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 80>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 256 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 256 : i32} {

tt.func @clamp_excessive_requested_registers() {
  // CHECK: actualRegisters = array<i32: 256, 256>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 300>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 256 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 300 : i32} {

tt.func @clamp_excessive_user_maxnreg() {
  // CHECK: actualRegisters = array<i32: 256, 256>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 184 : i32
module attributes {"ttg.num-warps" = 8 : i32, ttg.maxnreg = 184 : i32} {

tt.func @cap_default_rounding_surplus_boundary() {
  // Budget = 184*12 - 32*4 = 2080 warp-regs (260/thread -> rounds to 256),
  // leaving 32 warp-regs (+8 regs for the 4-warp worker).
  // CHECK: actualRegisters = array<i32: 256, 40>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 32>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

// CHECK: module attributes {ttg.maxnreg = 90 : i32
module attributes {"ttg.num-warps" = 4 : i32, ttg.maxnreg = 90 : i32} {

tt.func @cap_default_unequal_warps_remainder_order() {
  // Budget = 90*16 - 24*8 - 40*4 = 1088 warp-regs -> 64 warp-regs surplus.
  // Proportional shares truncate to 0; remainder favors higher per-thread
  // demand (40 > 24) even though the 8-warp group has a larger warp-reg weight.
  // CHECK: actualRegisters = array<i32: 256, 24, 56>
  ttg.warp_specialize() attributes {requestedRegisters = array<i32: 24, 40>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(8) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  }
  partition1() num_warps(4) {
    %0 = arith.constant 0 : i32
    ttg.warp_return
  } : () -> ()
  tt.return
}

}
