// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=True" | FileCheck %s --check-prefixes=COMMON,LLVM_FTZ
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm="gfx-arch=gfx942 ftz=False" | FileCheck %s --check-prefixes=COMMON,LLVM_NO_FTZ


#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp2(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ: rocdl.exp2
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_exp(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ: llvm.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = math.exp %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_rsqrt(%arg0: tensor<64xf32, #blocked>) {
    // ROCDL ops do not implement LLVM::FastmathFlagsInterface, so the FTZ path
    // cannot record the flags. The non-FTZ path is handled by MathToROCDL.
    // COMMON-LABEL: test_rsqrt
    // LLVM_FTZ: rocdl.rsq {{.*}} f32 -> f32
    // LLVM_FTZ-NOT: fastmathFlags
    // LLVM_NO_FTZ: llvm.call @__ocml_rsqrt_f32
    %0 = math.rsqrt %arg0 fastmath<nnan,afn> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_intrinsics(%arg0: tensor<64xf32, #blocked>) {
    // MathToROCDL leaves f32 log to MathToLLVM, which keeps the flags on the
    // intrinsic. This is what lets the AMDGPU backend act on `afn`.
    // COMMON-LABEL: test_fastmath_intrinsics
    // COMMON: llvm.intr.log({{.*}}) {fastmathFlags = #llvm.fastmath<nnan, afn>} : (f32) -> f32
    %log = math.log %arg0 fastmath<nnan,afn> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_f32(%arg0: tensor<64xf32, #blocked>) {
    // LLVM_FTZ-LABEL: test_sqrt_f32
    // LLVM_FTZ-NOT: llvm.fcmp "ogt"
    // LLVM_FTZ: rocdl.sqrt
    // LLVM_FTZ-NOT: llvm.fmul
    // LLVM_FTZ-NOT: llvm.select
    //
    // LLVM_NO_FTZ-LABEL: test_sqrt_f32
    // LLVM_NO_FTZ: llvm.fcmp "ogt" {{.*}} {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    // LLVM_NO_FTZ: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    // LLVM_NO_FTZ-NEXT: llvm.select {{.*}} {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    // LLVM_NO_FTZ-NEXT: rocdl.sqrt
    // LLVM_NO_FTZ: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    // LLVM_NO_FTZ-NEXT: llvm.select {{.*}} {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    %0 = math.sqrt %arg0 fastmath<nnan,nsz> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f32(%arg0: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_sqrt_rn_f32
    // COMMON: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_sqrt_rn_f64(%arg0: tensor<64xf64, #blocked>) {
    // COMMON-LABEL: test_sqrt_rn_f64
    // COMMON: llvm.intr.sqrt
    %0 = tt.precise_sqrt %arg0 : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_divf_rn_f32(%arg0: tensor<64xf32, #blocked>, %arg1: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_divf_rn_f32
    // COMMON: llvm.fdiv
    %0 = tt.precise_divf %arg0, %arg1 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_arith(
      %arg0: tensor<64xf32, #blocked>,
      %arg1: tensor<64xf32, #blocked>,
      %arg2: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_fastmath_arith
    // COMMON-DAG: llvm.fdiv {{.*}} {fastmathFlags = #llvm.fastmath<nsz, arcp, afn>} : f32
    // COMMON-DAG: llvm.fadd {{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
    // COMMON-DAG: llvm.fsub {{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
    // COMMON-DAG: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract>} : f32
    // COMMON-DAG: llvm.fneg {{.*}} {fastmathFlags = #llvm.fastmath<nnan>} : f32
    // COMMON-DAG: llvm.frem {{.*}} {fastmathFlags = #llvm.fastmath<ninf>} : f32
    // COMMON-DAG: llvm.intr.maxnum({{.*}}) {fastmathFlags = #llvm.fastmath<nnan, nsz>}
    // COMMON-DAG: llvm.intr.maximum({{.*}}) {fastmathFlags = #llvm.fastmath<nnan>}
    // COMMON-DAG: llvm.intr.minnum({{.*}}) {fastmathFlags = #llvm.fastmath<reassoc>}
    // COMMON-DAG: llvm.intr.minimum({{.*}}) {fastmathFlags = #llvm.fastmath<ninf>}
    // COMMON-DAG: llvm.fcmp "olt" {{.*}} {fastmathFlags = #llvm.fastmath<arcp>} : f32
    // COMMON-DAG: llvm.intr.fabs({{.*}}) {fastmathFlags = #llvm.fastmath<nsz>} : (f32) -> f32
    // COMMON-DAG: llvm.intr.fma({{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract>}
    %div = arith.divf %arg0, %arg1 fastmath<nsz,arcp,afn> : tensor<64xf32, #blocked>
    %add = arith.addf %div, %arg2 fastmath<nsz,contract> : tensor<64xf32, #blocked>
    %sub = arith.subf %add, %arg2 fastmath<nsz,contract> : tensor<64xf32, #blocked>
    %mul = arith.mulf %sub, %arg2 fastmath<nsz,contract> : tensor<64xf32, #blocked>
    %neg = arith.negf %mul fastmath<nnan> : tensor<64xf32, #blocked>
    %rem = arith.remf %neg, %arg2 fastmath<ninf> : tensor<64xf32, #blocked>
    %maxnum = arith.maxnumf %rem, %arg2 fastmath<nnan,nsz> : tensor<64xf32, #blocked>
    %maximum = arith.maximumf %maxnum, %arg2 fastmath<nnan> : tensor<64xf32, #blocked>
    %minnum = arith.minnumf %rem, %arg2 fastmath<reassoc> : tensor<64xf32, #blocked>
    %minimum = arith.minimumf %minnum, %arg2 fastmath<ninf> : tensor<64xf32, #blocked>
    %cmp = arith.cmpf olt, %arg0, %arg1 fastmath<arcp> : tensor<64xf32, #blocked>
    %abs = math.absf %arg0 fastmath<nsz> : tensor<64xf32, #blocked>
    %fma = math.fma %maximum, %arg1, %arg2 fastmath<nsz,contract> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_libcalls(
      %arg0: tensor<64xf32, #blocked>,
      %arg1: tensor<64xf16, #blocked>,
      %arg2: tensor<64xf64, #blocked>) {
    // Transcendentals routed to OCML by MathToROCDL do not carry the flags:
    // OpToFuncCallLowering drops them, and they would be inert on the callsite
    // anyway because the inliner does not push callsite flags into the body.
    // COMMON-LABEL: test_fastmath_libcalls
    // COMMON-DAG: llvm.call @__ocml_floor_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_log2_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_cos_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_sin_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_erf_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_log_f16({{.*}}) : (f16) -> f16
    // COMMON-DAG: llvm.call @__ocml_log_f64({{.*}}) : (f64) -> f64
    %floor = math.floor %arg0 fastmath<afn> : tensor<64xf32, #blocked>
    %log2 = math.log2 %arg0 fastmath<nnan,afn> : tensor<64xf32, #blocked>
    %cos = math.cos %arg0 fastmath<ninf,afn> : tensor<64xf32, #blocked>
    %sin = math.sin %arg0 fastmath<arcp,afn> : tensor<64xf32, #blocked>
    %erf = math.erf %arg0 fastmath<nnan> : tensor<64xf32, #blocked>
    %log16 = math.log %arg1 fastmath<nsz> : tensor<64xf16, #blocked>
    %log64 = math.log %arg2 fastmath<ninf> : tensor<64xf64, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_none(%arg0: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_fastmath_none
    // COMMON-NOT: fastmathFlags
    // COMMON: llvm.return
    %floor = math.floor %arg0 : tensor<64xf32, #blocked>
    %log2 = math.log2 %arg0 : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_bf16(
      %arg0: tensor<64xbf16, #blocked>,
      %arg1: tensor<64xbf16, #blocked>) {
    // COMMON-LABEL: test_fastmath_bf16
    // COMMON-DAG: llvm.fdiv {{.*}} {fastmathFlags = #llvm.fastmath<arcp>} : bf16
    // COMMON-DAG: llvm.fadd {{.*}} {fastmathFlags = #llvm.fastmath<contract>} : f32
    // COMMON-DAG: llvm.fsub {{.*}} {fastmathFlags = #llvm.fastmath<contract>} : f32
    // COMMON-DAG: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<contract>} : f32
    // COMMON-DAG: llvm.call @__ocml_exp2_f32({{.*}}) : (f32) -> f32
    // COMMON-DAG: llvm.call @__ocml_rsqrt_f32({{.*}}) : (f32) -> f32
    %div = arith.divf %arg0, %arg1 fastmath<arcp> : tensor<64xbf16, #blocked>
    %add = arith.addf %div, %arg1 fastmath<contract> : tensor<64xbf16, #blocked>
    %sub = arith.subf %add, %arg1 fastmath<contract> : tensor<64xbf16, #blocked>
    %mul = arith.mulf %sub, %arg1 fastmath<contract> : tensor<64xbf16, #blocked>
    %exp2 = math.exp2 %arg0 fastmath<afn> : tensor<64xbf16, #blocked>
    %rsqrt = math.rsqrt %arg0 fastmath<afn> : tensor<64xbf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_exp(%arg0: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_fastmath_exp
    // COMMON-DAG: llvm.fmul {{.*}} {fastmathFlags = #llvm.fastmath<nsz, contract, afn>} : f32
    // COMMON-DAG: llvm.call @llvm.exp2.f32({{.*}}) {fastmathFlags = #llvm.fastmath<nsz, contract, afn>}
    %0 = math.exp %arg0 fastmath<nsz,contract,afn> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_exp2(%arg0: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_fastmath_exp2
    // LLVM_FTZ: rocdl.exp2
    // LLVM_FTZ-NOT: fastmathFlags
    // LLVM_NO_FTZ: llvm.intr.exp2({{.*}}) {fastmathFlags = #llvm.fastmath<afn>}
    %0 = math.exp2 %arg0 fastmath<afn> : tensor<64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_casts(
      %arg0: tensor<64xf16, #blocked>,
      %arg1: tensor<64xf64, #blocked>,
      %arg2: tensor<64xf32, #blocked>) {
    // COMMON-LABEL: test_fastmath_casts
    // COMMON-DAG: llvm.fpext {{.*}} fastmath<nnan> : f16 to f32
    // COMMON-DAG: llvm.fptrunc {{.*}} fastmath<ninf> : f64 to f32
    // COMMON-DAG: llvm.fptrunc {{.*}} fastmath<contract> : f32 to f16
    // The f16 math ops are left to MathToROCDL, which drops the flags.
    // COMMON-DAG: llvm.call @__ocml_exp_f16({{.*}}) : (f16) -> f16
    // COMMON-DAG: llvm.call @__ocml_exp2_f16({{.*}}) : (f16) -> f16
    // COMMON-DAG: llvm.call @__ocml_rsqrt_f16({{.*}}) : (f16) -> f16
    %ext = arith.extf %arg0 fastmath<nnan> : tensor<64xf16, #blocked> to tensor<64xf32, #blocked>
    %trunc = arith.truncf %arg1 fastmath<ninf> : tensor<64xf64, #blocked> to tensor<64xf32, #blocked>
    %truncf16 = arith.truncf %arg2 fastmath<contract> : tensor<64xf32, #blocked> to tensor<64xf16, #blocked>
    %exp = math.exp %arg0 fastmath<afn> : tensor<64xf16, #blocked>
    %exp2 = math.exp2 %arg0 fastmath<afn> : tensor<64xf16, #blocked>
    %rsqrt = math.rsqrt %arg0 fastmath<afn> : tensor<64xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fastmath_bf16_casts(
      %arg0: tensor<64xbf16, #blocked>,
      %arg1: tensor<64xf32, #blocked>) {
    // The bf16 conversions are software sequences of integer bit operations,
    // which cannot represent fast-math flags.
    // COMMON-LABEL: test_fastmath_bf16_casts
    // COMMON-NOT: fastmath
    // COMMON: llvm.return
    %ext = arith.extf %arg0 fastmath<nnan> : tensor<64xbf16, #blocked> to tensor<64xf32, #blocked>
    %trunc = arith.truncf %arg1 fastmath<ninf> : tensor<64xf32, #blocked> to tensor<64xbf16, #blocked>
    tt.return
  }
}
