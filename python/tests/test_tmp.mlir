
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {triton_gpu.externs = {libdevice = "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc"}, "triton_gpu.num-warps" = 4 : i32} {
  func public @kernel_0d1d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0> : tensor<128xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %1 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>, #blocked>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<f32>, #blocked>
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32, #blocked>
    %4 = arith.sitofp %cst : tensor<128xi32, #blocked> to tensor<128xf32, #blocked>
    %5 = "triton_gpu.cmpf"(%3, %4) {predicate = 3 : i64} : (tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xi1, #blocked>
    %6 = arith.subf %cst_0, %3 : tensor<128xf32, #blocked>
    %7 = "triton_gpu.select"(%5, %3, %6) : (tensor<128xi1, #blocked>, tensor<128xf32, #blocked>, tensor<128xf32, #blocked>) -> tensor<128xf32, #blocked>
    %8 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>, #blocked>
    %9 = tt.addptr %8, %0 : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %9, %7 : tensor<128xf32, #blocked>
    return
  }
}