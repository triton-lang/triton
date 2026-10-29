// RUN: triton-opt -split-input-file %s --convert-triton-amdgpu-to-llvm='gfx-arch=gfx942' -verify-diagnostics

// Downcasts to E4M3 only support RTNE.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @f16_to_fp8_rtz(%arg0: f16) -> f8E4M3FN {
    // expected-error @+2 {{Unsupported conversion from 'f16' to 'f8E4M3FN' with rounding mode rtz}}
    // expected-error @+1 {{failed to legalize operation 'tt.fp_to_fp'}}
    %0 = tt.fp_to_fp %arg0, rounding = rtz : f16 -> f8E4M3FN
    tt.return %0 : f8E4M3FN
  }
}

// -----

// RTZ downcasts from f32 to E5M2 only support the OCP format.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @f32_to_bf8fnuz_rtz(%arg0: f32) -> f8E5M2FNUZ {
    // expected-error @+2 {{Unsupported conversion from 'f32' to 'f8E5M2FNUZ' with rounding mode rtz}}
    // expected-error @+1 {{failed to legalize operation 'tt.fp_to_fp'}}
    %0 = tt.fp_to_fp %arg0, rounding = rtz : f32 -> f8E5M2FNUZ
    tt.return %0 : f8E5M2FNUZ
  }
}
