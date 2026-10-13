// RUN: triton-opt %s -triton-cpu-math-to-vec-lib="cpu_features=sve" | FileCheck %s --enable-var-scope --implicit-check-not="func.call @Sleef_" --implicit-check-not=math.
// RUN: triton-opt %s -triton-cpu-math-to-vec-lib="cpu_features=neon,sve" | FileCheck %s --enable-var-scope --implicit-check-not="func.call @Sleef_" --implicit-check-not=math.
// RUN: triton-opt %s -triton-cpu-math-to-vec-lib="cpu_features=sve2" | FileCheck %s --enable-var-scope --implicit-check-not="func.call @Sleef_" --implicit-check-not=math.

// SVE calls have a scalable ABI even for fixed blocks. Odd blocks must mask
// the final hardware vector, and each evaluation releases its scratch storage.
// CHECK-DAG: func.func private @Sleef_sinfx_u10sve(vector<[4]xf32>) -> vector<[4]xf32>
// CHECK-DAG: func.func private @Sleef_sindx_u10sve(vector<[2]xf64>) -> vector<[2]xf64>

module {
  // CHECK-LABEL: llvm.func @sin_odd_f32(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<5xf32>
  // CHECK-NOT: math.sin
  // CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[BOUND:.*]] = arith.constant 5 : index
  // CHECK-DAG: %[[LANES:.*]] = arith.constant 4 : index
  // CHECK: %[[SAVE:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK-NEXT: %[[INPUT:.*]] = memref.alloca() : memref<5xf32>
  // CHECK-NEXT: %[[OUTPUT:.*]] = memref.alloca() : memref<5xf32>
  // CHECK: %[[VSCALE:.*]] = vector.vscale
  // CHECK-NEXT: %[[STEP:.*]] = arith.muli %[[VSCALE]], %[[LANES]] : index
  // CHECK: vector.store %[[ARG]], %[[INPUT]][%[[ZERO]]] : memref<5xf32>, vector<5xf32>
  // CHECK: scf.for %[[IV:.*]] = %[[ZERO]] to %[[BOUND]] step %[[STEP]] {
  // CHECK-NEXT: %[[REMAIN:.*]] = arith.subi %[[BOUND]], %[[IV]] : index
  // CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[REMAIN]] : vector<[4]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %[[INPUT]][%[[IV]]], %[[MASK]], %{{.*}} : memref<5xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sinfx_u10sve(%[[LOADED]]) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NEXT: vector.maskedstore %[[OUTPUT]][%[[IV]]], %[[MASK]], %[[CALLED]] : memref<5xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[RESULT:.*]] = vector.load %[[OUTPUT]][%[[ZERO]]] : memref<5xf32>, vector<5xf32>
  // CHECK-NEXT: llvm.intr.stackrestore %[[SAVE]] : !llvm.ptr
  // CHECK-NOT: math.sin
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<5xf32>
  llvm.func @sin_odd_f32(%arg: vector<5xf32>) -> vector<5xf32> {
    %result = math.sin %arg : vector<5xf32>
    llvm.return %result : vector<5xf32>
  }

  // CHECK-LABEL: llvm.func @sin_odd_f64(
  // CHECK-NOT: math.sin
  // CHECK-DAG: %[[BOUND:.*]] = arith.constant 3 : index
  // CHECK-DAG: %[[LANES:.*]] = arith.constant 2 : index
  // CHECK: %[[VSCALE:.*]] = vector.vscale
  // CHECK-NEXT: %[[STEP:.*]] = arith.muli %[[VSCALE]], %[[LANES]] : index
  // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %[[BOUND]] step %[[STEP]] {
  // CHECK-NEXT: %[[REMAIN:.*]] = arith.subi %[[BOUND]], %[[IV]] : index
  // CHECK-NEXT: %[[MASK:.*]] = vector.create_mask %[[REMAIN]] : vector<[2]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %{{.*}}[%[[IV]]], %[[MASK]], %{{.*}} : memref<3xf64>, vector<[2]xi1>, vector<[2]xf64>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sindx_u10sve(%[[LOADED]]) : (vector<[2]xf64>) -> vector<[2]xf64>
  // CHECK-NEXT: vector.maskedstore %{{.*}}[%[[IV]]], %[[MASK]], %[[CALLED]] : memref<3xf64>, vector<[2]xi1>, vector<[2]xf64>
  // CHECK: llvm.intr.stackrestore
  // CHECK-NOT: math.sin
  // CHECK: llvm.return %{{.*}} : vector<3xf64>
  llvm.func @sin_odd_f64(%arg: vector<3xf64>) -> vector<3xf64> {
    %result = math.sin %arg : vector<3xf64>
    llvm.return %result : vector<3xf64>
  }

  // Keep rank-two results live to verify shape restoration explicitly.
  // CHECK-LABEL: tt.func @sin_reshape_f32(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<2x4xf32>
  // CHECK-NOT: math.sin
  // CHECK: vector.vscale
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[ARG]] : vector<2x4xf32> to vector<8xf32>
  // CHECK: vector.store %[[FLAT]], %{{.*}} : memref<8xf32>, vector<8xf32>
  // CHECK: scf.for
  // CHECK: %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<[4]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %{{.*}}, %[[MASK]], %{{.*}} : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sinfx_u10sve(%[[LOADED]]) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NEXT: vector.maskedstore %{{.*}}, %[[MASK]], %[[CALLED]] : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[RESULT:.*]] = vector.load %{{.*}} : memref<8xf32>, vector<8xf32>
  // CHECK-NEXT: llvm.intr.stackrestore
  // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[RESULT]] : vector<8xf32> to vector<2x4xf32>
  // CHECK-NOT: math.sin
  // CHECK-NEXT: tt.return %[[SHAPED]] : vector<2x4xf32>
  tt.func @sin_reshape_f32(%arg: vector<2x4xf32>) -> vector<2x4xf32> {
    %result = math.sin %arg : vector<2x4xf32>
    tt.return %result : vector<2x4xf32>
  }

  // CHECK-LABEL: tt.func @sin_reshape_f64(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<2x3xf64>
  // CHECK-NOT: math.sin
  // CHECK: vector.vscale
  // CHECK: %[[FLAT:.*]] = vector.shape_cast %[[ARG]] : vector<2x3xf64> to vector<6xf64>
  // CHECK: vector.store %[[FLAT]], %{{.*}} : memref<6xf64>, vector<6xf64>
  // CHECK: scf.for
  // CHECK: %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<[2]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %{{.*}}, %[[MASK]], %{{.*}} : memref<6xf64>, vector<[2]xi1>, vector<[2]xf64>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sindx_u10sve(%[[LOADED]]) : (vector<[2]xf64>) -> vector<[2]xf64>
  // CHECK-NEXT: vector.maskedstore %{{.*}}, %[[MASK]], %[[CALLED]] : memref<6xf64>, vector<[2]xi1>, vector<[2]xf64>
  // CHECK: %[[RESULT:.*]] = vector.load %{{.*}} : memref<6xf64>, vector<6xf64>
  // CHECK-NEXT: llvm.intr.stackrestore
  // CHECK-NEXT: %[[SHAPED:.*]] = vector.shape_cast %[[RESULT]] : vector<6xf64> to vector<2x3xf64>
  // CHECK-NOT: math.sin
  // CHECK-NEXT: tt.return %[[SHAPED]] : vector<2x3xf64>
  tt.func @sin_reshape_f64(%arg: vector<2x3xf64>) -> vector<2x3xf64> {
    %result = math.sin %arg : vector<2x3xf64>
    tt.return %result : vector<2x3xf64>
  }

  // CHECK-LABEL: llvm.func @sin_reshape_f16(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<8xf16>
  // CHECK-NOT: math.sin
  // CHECK: %[[INPUT:.*]] = vector.shape_cast %[[ARG]] : vector<8xf16> to vector<2x4xf16>
  // CHECK: %[[PROMOTED:.*]] = arith.extf %[[INPUT]] : vector<2x4xf16> to vector<2x4xf32>
  // CHECK: vector.vscale
  // CHECK: vector.shape_cast %[[PROMOTED]] : vector<2x4xf32> to vector<8xf32>
  // CHECK: scf.for
  // CHECK: %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<[4]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %{{.*}}, %[[MASK]], %{{.*}} : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sinfx_u10sve(%[[LOADED]]) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NEXT: vector.maskedstore %{{.*}}, %[[MASK]], %[[CALLED]] : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<8xf32> to vector<2x4xf32>
  // CHECK-NEXT: %[[NARROWED:.*]] = arith.truncf %[[SHAPED]] : vector<2x4xf32> to vector<2x4xf16>
  // CHECK-NEXT: %[[RESULT:.*]] = vector.shape_cast %[[NARROWED]] : vector<2x4xf16> to vector<8xf16>
  // CHECK-NOT: math.sin
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<8xf16>
  llvm.func @sin_reshape_f16(%arg: vector<8xf16>) -> vector<8xf16> {
    %shaped = vector.shape_cast %arg : vector<8xf16> to vector<2x4xf16>
    %result = math.sin %shaped : vector<2x4xf16>
    %flat = vector.shape_cast %result : vector<2x4xf16> to vector<8xf16>
    llvm.return %flat : vector<8xf16>
  }

  // CHECK-LABEL: llvm.func @sin_reshape_bf16(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<8xbf16>
  // CHECK-NOT: math.sin
  // CHECK: %[[INPUT:.*]] = vector.shape_cast %[[ARG]] : vector<8xbf16> to vector<2x4xbf16>
  // CHECK: %[[PROMOTED:.*]] = arith.extf %[[INPUT]] : vector<2x4xbf16> to vector<2x4xf32>
  // CHECK: vector.vscale
  // CHECK: vector.shape_cast %[[PROMOTED]] : vector<2x4xf32> to vector<8xf32>
  // CHECK: scf.for
  // CHECK: %[[MASK:.*]] = vector.create_mask %{{.*}} : vector<[4]xi1>
  // CHECK-NEXT: %[[LOADED:.*]] = vector.maskedload %{{.*}}, %[[MASK]], %{{.*}} : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK-NEXT: %[[CALLED:.*]] = func.call @Sleef_sinfx_u10sve(%[[LOADED]]) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NEXT: vector.maskedstore %{{.*}}, %[[MASK]], %[[CALLED]] : memref<8xf32>, vector<[4]xi1>, vector<[4]xf32>
  // CHECK: %[[SHAPED:.*]] = vector.shape_cast %{{.*}} : vector<8xf32> to vector<2x4xf32>
  // CHECK-NEXT: %[[NARROWED:.*]] = arith.truncf %[[SHAPED]] : vector<2x4xf32> to vector<2x4xbf16>
  // CHECK-NEXT: %[[RESULT:.*]] = vector.shape_cast %[[NARROWED]] : vector<2x4xbf16> to vector<8xbf16>
  // CHECK-NOT: math.sin
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<8xbf16>
  llvm.func @sin_reshape_bf16(%arg: vector<8xbf16>) -> vector<8xbf16> {
    %shaped = vector.shape_cast %arg : vector<8xbf16> to vector<2x4xbf16>
    %result = math.sin %shaped : vector<2x4xbf16>
    %flat = vector.shape_cast %result : vector<2x4xbf16> to vector<8xbf16>
    llvm.return %flat : vector<8xbf16>
  }

  // An earlier SSA result remains live after releasing its scratch storage.
  // CHECK-LABEL: llvm.func @scoped_math(
  // CHECK-SAME: %[[ARG:[^:]+]]: vector<16xf32>
  // CHECK-NOT: math.sin
  // CHECK: %[[SAVE0:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK-NEXT: %[[INPUT0:.*]] = memref.alloca() : memref<16xf32>
  // CHECK-NEXT: %[[OUTPUT0:.*]] = memref.alloca() : memref<16xf32>
  // CHECK: vector.store %[[ARG]], %[[INPUT0]][%{{.*}}] : memref<16xf32>, vector<16xf32>
  // CHECK: %[[LOADED0:.*]] = vector.maskedload %[[INPUT0]]
  // CHECK-NEXT: %[[CALLED0:.*]] = func.call @Sleef_sinfx_u10sve(%[[LOADED0]])
  // CHECK-NEXT: vector.maskedstore %[[OUTPUT0]][%{{.*}}], %{{.*}}, %[[CALLED0]]
  // CHECK: %[[FIRST:.*]] = vector.load %[[OUTPUT0]][%{{.*}}] : memref<16xf32>, vector<16xf32>
  // CHECK-NEXT: llvm.intr.stackrestore %[[SAVE0]] : !llvm.ptr
  // CHECK: %[[SAVE1:.*]] = llvm.intr.stacksave : !llvm.ptr
  // CHECK-NEXT: %[[INPUT1:.*]] = memref.alloca() : memref<16xf32>
  // CHECK-NEXT: %[[OUTPUT1:.*]] = memref.alloca() : memref<16xf32>
  // CHECK: vector.store %[[FIRST]], %[[INPUT1]][%{{.*}}] : memref<16xf32>, vector<16xf32>
  // CHECK: %[[LOADED1:.*]] = vector.maskedload %[[INPUT1]]
  // CHECK-NEXT: %[[CALLED1:.*]] = func.call @Sleef_cosfx_u10sve(%[[LOADED1]])
  // CHECK-NEXT: vector.maskedstore %[[OUTPUT1]][%{{.*}}], %{{.*}}, %[[CALLED1]]
  // CHECK: %[[SECOND:.*]] = vector.load %[[OUTPUT1]][%{{.*}}] : memref<16xf32>, vector<16xf32>
  // CHECK-NEXT: llvm.intr.stackrestore %[[SAVE1]] : !llvm.ptr
  // CHECK-NOT: math.sin
  // CHECK-NOT: math.cos
  // CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[FIRST]], %[[SECOND]] : vector<16xf32>
  // CHECK-NEXT: llvm.return %[[SUM]] : vector<16xf32>
  llvm.func @scoped_math(%arg: vector<16xf32>) -> vector<16xf32> {
    %first = math.sin %arg : vector<16xf32>
    %second = math.cos %first : vector<16xf32>
    %result = arith.addf %first, %second : vector<16xf32>
    llvm.return %result : vector<16xf32>
  }

  // CHECK-LABEL: llvm.func @sqrt_odd(
  // CHECK-NOT: math.sqrt
  // CHECK: vector.vscale
  // CHECK: func.call @Sleef_sqrtfx_u05sve(%{{.*}}) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NOT: math.sqrt
  // CHECK: llvm.return
  llvm.func @sqrt_odd(%arg: vector<5xf32>) -> vector<5xf32> {
    %result = math.sqrt %arg : vector<5xf32>
    llvm.return %result : vector<5xf32>
  }

  // CHECK-LABEL: llvm.func @floor_odd(
  // CHECK-NOT: math.floor
  // CHECK: vector.vscale
  // CHECK: func.call @Sleef_floorfx_sve(%{{.*}}) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NOT: math.floor
  // CHECK: llvm.return
  llvm.func @floor_odd(%arg: vector<5xf32>) -> vector<5xf32> {
    %result = math.floor %arg : vector<5xf32>
    llvm.return %result : vector<5xf32>
  }

  // CHECK-LABEL: llvm.func @expm1_odd(
  // CHECK-NOT: math.expm1
  // CHECK: vector.vscale
  // CHECK: func.call @Sleef_expm1fx_u10sve(%{{.*}}) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NOT: math.expm1
  // CHECK: llvm.return
  llvm.func @expm1_odd(%arg: vector<5xf32>) -> vector<5xf32> {
    %result = math.expm1 %arg : vector<5xf32>
    llvm.return %result : vector<5xf32>
  }

  // CHECK-LABEL: llvm.func @trunc_odd(
  // CHECK-NOT: math.trunc
  // CHECK: vector.vscale
  // CHECK: func.call @Sleef_truncfx_sve(%{{.*}}) : (vector<[4]xf32>) -> vector<[4]xf32>
  // CHECK-NOT: math.trunc
  // CHECK: llvm.return
  llvm.func @trunc_odd(%arg: vector<5xf32>) -> vector<5xf32> {
    %result = math.trunc %arg : vector<5xf32>
    llvm.return %result : vector<5xf32>
  }

  // Subminimum vectors remain math ops and do not enter an SVE scratch scope.
  // CHECK-LABEL: llvm.func @sin_small_f32_1(
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-NOT: Sleef_
  // CHECK: %[[RESULT:.*]] = math.sin %{{.*}} : vector<1xf32>
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<1xf32>
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: Sleef_
  llvm.func @sin_small_f32_1(%arg: vector<1xf32>) -> vector<1xf32> {
    %result = math.sin %arg : vector<1xf32>
    llvm.return %result : vector<1xf32>
  }

  // CHECK-LABEL: llvm.func @sin_small_f32_2(
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-NOT: Sleef_
  // CHECK: %[[RESULT:.*]] = math.sin %{{.*}} : vector<2xf32>
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<2xf32>
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: Sleef_
  llvm.func @sin_small_f32_2(%arg: vector<2xf32>) -> vector<2xf32> {
    %result = math.sin %arg : vector<2xf32>
    llvm.return %result : vector<2xf32>
  }

  // CHECK-LABEL: llvm.func @sin_small_f64(
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-NOT: Sleef_
  // CHECK: %[[RESULT:.*]] = math.sin %{{.*}} : vector<1xf64>
  // CHECK-NEXT: llvm.return %[[RESULT]] : vector<1xf64>
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: Sleef_
  llvm.func @sin_small_f64(%arg: vector<1xf64>) -> vector<1xf64> {
    %result = math.sin %arg : vector<1xf64>
    llvm.return %result : vector<1xf64>
  }

  // Explicit extern calls keep their fixed ABI even when SVE is available.
  // CHECK-LABEL: llvm.func @extern_fixed_abi(
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: llvm.intr.stacksave
  // CHECK-COUNT-4: func.call @Sleef_ceilf4(%{{.*}}) : (vector<4xf32>) -> vector<4xf32>
  // CHECK-NOT: func.call @Sleef_ceilf4
  // CHECK-NOT: sve
  // CHECK: llvm.return
  // CHECK-NOT: vector.vscale
  // CHECK-NOT: sve
  llvm.func @extern_fixed_abi(%arg: vector<16xf32>) -> vector<16xf32> {
    %result = triton_cpu.extern_elementwise %arg {symbol = "Sleef_ceilf%(numel)", pure = true}
      : (vector<16xf32>) -> vector<16xf32>
    llvm.return %result : vector<16xf32>
  }
}
