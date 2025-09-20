module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

tt.func @optimize_broadcast(%arg0: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: scf.for
  scf.for %i = %c0_i32 to %arg0 step %c1_i32 : i32 {
    // CHECK: [[X:%.*]] = "producer"{{.*}}partition = 0
    %x = "producer"() {ttg.partition = 0 : i32, data} : () -> tensor<128xf32>

    // CHECK-DAG: [[X0_P0:%.*]] = tt.expand_dims [[X]] {{.*}}partition = 0
    // CHECK-DAG: [[X0_P1:%.*]] = tt.expand_dims [[X]] {{.*}}partition = 1
    %x0 = tt.expand_dims %x {axis = 0 : i32, data} : tensor<128xf32> -> tensor<1x128xf32>
    // CHECK-DAG: [[X1_P0:%.*]] = tt.broadcast [[X0_P0]] {{.*}}partition = 0
    // CHECK-DAG: [[X1_P1:%.*]] = tt.broadcast [[X0_P1]] {{.*}}partition = 1
    %x1 = tt.broadcast %x0 {data}: tensor<1x128xf32> -> tensor<128x128xf32>

    // CHECK: "use"([[X1_P0]]) {{.*}}partition = 0
    "use"(%x1) {ttg.partition = 0 : i32, data} : (tensor<128x128xf32>) -> ()
    // CHECK: "use"([[X1_P1]]) {{.*}}partition = 1
    "use"(%x1) {ttg.partition = 1 : i32, data} : (tensor<128x128xf32>) -> ()
  } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}
