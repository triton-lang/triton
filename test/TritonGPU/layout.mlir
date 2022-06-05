// RUN: triton-opt %s -split-input-file -verify-diagnostics

#reg = #triton_gpu.sharded_layout<{
  threadTileSize = [1, 1],
  warpTileSize = [32, 1],
  blockTileSize = [64, 1],
  order = [0, 1]
}>

#reg2 = #triton_gpu.sharded_layout<{
  threadTileSize = [2, 1],
  warpTileSize = [64, 1],
  blockTileSize = [128, 1],
  order = [0, 1]
}>

func @add(%arg0: tensor<256xi32, #reg>, %arg1: tensor<256xi32, #reg>) {
  %0 = arith.addi %arg0, %arg1 : tensor<256xi32, #reg>
  return
}

func @add(%arg0: tensor<256xi32, #reg>, %arg1: tensor<256xi32, #reg>) { // expected-note {{prior use here}}
  // expected-error @+1 {{use of value '%arg0' expects different type than prior uses}}
  %0 = arith.addi %arg0, %arg1 : tensor<256xi32, #reg2>
  return
}