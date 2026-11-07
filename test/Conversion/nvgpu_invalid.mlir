// RUN: triton-opt -split-input-file %s -verify-diagnostics

!acc = !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
llvm.func @wgmma_missing_descriptor_bits(%a: i32, %b: i64) {
  %false = llvm.mlir.constant(false) : i1
  // expected-error @below {{expected an i64 descriptor without descriptor bits, an i32 base with descriptor bits, or registers for A}}
  %out = nvg.wgmma %a, %b, %false {
    m = 64 : i32, n = 16 : i32, k = 16 : i32,
    eltTypeA = 5 : i32, eltTypeB = 5 : i32, eltTypeC = 7 : i32,
    layoutA = 0 : i32, layoutB = 1 : i32
  } : (i32, i64, i1) -> !acc
  llvm.return
}

// -----

!acc = !llvm.struct<(f32, f32, f32, f32, f32, f32, f32, f32)>
llvm.func @wgmma_bits_on_full_descriptor(%a: i64, %b: i64) {
  %false = llvm.mlir.constant(false) : i1
  // expected-error @below {{expected an i64 descriptor without descriptor bits, an i32 base with descriptor bits, or registers for A}}
  %out = nvg.wgmma %a, %b, %false {
    m = 64 : i32, n = 16 : i32, k = 16 : i32,
    eltTypeA = 5 : i32, eltTypeB = 5 : i32, eltTypeC = 7 : i32,
    layoutA = 0 : i32, layoutB = 1 : i32, descriptorBitsB = 0 : i64
  } : (i64, i64, i1) -> !acc
  llvm.return
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  llvm.func @cluster_barrier_num_ctas_invalid() {
    // expected-error @below {{requires ttg.num-ctas > 1}}
    ttng.cluster_barrier
    llvm.return
  }
}
