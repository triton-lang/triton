// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=107 -cse | FileCheck %s

#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_f16_x32
  // CHECK: spdecompress.b16.b2.sp::2:4.x32
  tt.func public @spdecompress_f16_x32(
      %cdata: tensor<128x64xf16, #cdata>,
      %mdata: tensor<128x4xi32, #mdata>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf16, #cdata>, tensor<128x4xi32, #mdata> -> tensor<128x128xf16, #dense>
    tt.return
  }
}

// -----

#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_bf16_x32
  // CHECK: spdecompress.b16.b2.sp::2:4.x32
  tt.func public @spdecompress_bf16_x32(
      %cdata: tensor<128x64xbf16, #cdata>,
      %mdata: tensor<128x4xi32, #mdata>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xbf16, #cdata>, tensor<128x4xi32, #mdata> -> tensor<128x128xbf16, #dense>
    tt.return
  }
}

// -----

#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_f8e5_x32
  // CHECK: spdecompress.b8.b2.sp::2:4.x32
  tt.func public @spdecompress_f8e5_x32(
      %cdata: tensor<128x64xf8E5M2, #cdata>,
      %mdata: tensor<128x4xi32, #mdata>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf8E5M2, #cdata>, tensor<128x4xi32, #mdata> -> tensor<128x128xf8E5M2, #dense>
    tt.return
  }
}

// -----

#cdata = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_f8e4_x32
  // CHECK: spdecompress.b8.b2.sp::2:4.x32
  tt.func public @spdecompress_f8e4_x32(
      %cdata: tensor<128x64xf8E4M3FN, #cdata>,
      %mdata: tensor<128x4xi32, #mdata>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x64xf8E4M3FN, #cdata>, tensor<128x4xi32, #mdata> -> tensor<128x128xf8E4M3FN, #dense>
    tt.return
  }
}

// -----

#cdata_small = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_small = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_small = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_f16_x1
  // CHECK: spdecompress.b16.b2.sp::2:4.x1
  tt.func public @spdecompress_f16_x1(
      %cdata: tensor<128x2xf16, #cdata_small>,
      %mdata: tensor<128x1xi32, #mdata_small>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x2xf16, #cdata_small>, tensor<128x1xi32, #mdata_small> -> tensor<128x4xf16, #dense_small>
    tt.return
  }
}

// -----

#cdata_large = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mdata_large = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#dense_large = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:107", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @spdecompress_f16_x64
  // CHECK: spdecompress.b16.b2.sp::2:4.x64
  tt.func public @spdecompress_f16_x64(
      %cdata: tensor<128x128xf16, #cdata_large>,
      %mdata: tensor<128x8xi32, #mdata_large>) {
    %dense = ttng.sp_decompress %cdata, %mdata :
      tensor<128x128xf16, #cdata_large>, tensor<128x8xi32, #mdata_large> -> tensor<128x256xf16, #dense_large>
    tt.return
  }
}
