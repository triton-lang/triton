// RUN: split-file %s %t
// RUN: triton-opt %t/unsupported.mlir -verify-each -verify-diagnostics \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   | FileCheck %t/unsupported.mlir --check-prefix=REJECT \
// RUN:     --implicit-check-not=arm_neon.intr.bfmmla --implicit-check-not=vector.transpose
// RUN: triton-opt %t/emitter.mlir -verify-each -verify-diagnostics \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla > %t/emitter.out
// RUN: FileCheck %t/emitter.mlir --check-prefix=PACK < %t/emitter.out \
// RUN:     --implicit-check-not=vector.extract --implicit-check-not=vector.shuffle \
// RUN:     --implicit-check-not=arm_neon.intr.bfmmla \
// RUN:     --implicit-check-not=vector.contract --implicit-check-not=vector.shape_cast \
// RUN:     --implicit-check-not=vector.insert_strided_slice \
// RUN:     --implicit-check-not=scf.for --implicit-check-not=llvm.
// RUN: FileCheck %t/emitter.mlir --check-prefix=INSTR < %t/emitter.out \
// RUN:     --implicit-check-not=arm_neon.intr.bfmmla
// RUN: FileCheck %t/emitter.mlir --check-prefix=INSERT < %t/emitter.out \
// RUN:     --implicit-check-not=vector.insert_strided_slice
// RUN: FileCheck %t/emitter.mlir --check-prefix=CAST < %t/emitter.out \
// RUN:     --implicit-check-not=vector.shape_cast
// RUN: FileCheck %t/emitter.mlir --check-prefix=SHUFFLE < %t/emitter.out \
// RUN:     --implicit-check-not=vector.shuffle
// RUN: triton-opt %t/canonicalize.mlir -verify-each -verify-diagnostics \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   | FileCheck %t/canonicalize.mlir --check-prefix=CANON \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=arm_neon.intr.bfmmla \
// RUN:     --implicit-check-not=vector.contract --implicit-check-not=scf.for
// RUN: triton-opt %t/canonicalize.mlir -verify-each \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   | FileCheck %t/canonicalize.mlir --check-prefix=CANON \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=arm_neon.intr.bfmmla \
// RUN:     --implicit-check-not=vector.contract --implicit-check-not=scf.for
// RUN: triton-opt %t/pipeline.mlir -verify-each -triton-cpu-convert-dot-generic \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   | FileCheck %t/pipeline.mlir --check-prefixes=MAPS,PIPELINE \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=arith.extf \
// RUN:     --implicit-check-not=triton_cpu.dot --implicit-check-not=arm_neon.intr.bfmmla
// RUN: triton-opt %t/pipeline.mlir -verify-each -triton-cpu-convert-dot-generic \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   -triton-cpu-add-casts-for-unsupported-ops="promote-bf16-to-fp32=false convert-mixed-precision-matmul=true" \
// RUN:   | FileCheck %t/pipeline.mlir --check-prefixes=MAPS,FALLBACK \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=triton_cpu.dot \
// RUN:     --implicit-check-not=arm_neon.intr.bfmmla --implicit-check-not=arith.extf
// RUN: triton-opt %t/pipeline.mlir -verify-each -triton-cpu-convert-dot-generic \
// RUN:   -triton-cpu-add-casts-for-unsupported-ops="promote-bf16-to-fp32=false convert-mixed-precision-matmul=true" \
// RUN:   | FileCheck %t/pipeline.mlir --check-prefixes=MAPS,DISABLED \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=triton_cpu.dot \
// RUN:     --implicit-check-not=arm_neon.intr.bfmmla
// RUN: triton-opt %t/production-emitter.mlir -verify-each -verify-diagnostics \
// RUN:   -triton-cpu-convert-dot-generic \
// RUN:   -triton-cpu-convert-vector-contract-to-arm-neon-bfmmla \
// RUN:   | triton-opt -verify-each \
// RUN:   | FileCheck %t/production-emitter.mlir \
// RUN:     --implicit-check-not=vector.contract --implicit-check-not=triton_cpu.dot \
// RUN:     --implicit-check-not=vector.transpose --implicit-check-not=arm_neon.intr.bfmmla \
// RUN:     --implicit-check-not=arith.extf --implicit-check-not=scf.for \
// RUN:     --implicit-check-not=llvm.

// Candidate rejection reasons use the pass's debug logging; pattern failures
// use the rewrite driver's debug logging. Neither emits ordinary
// diagnostics, as checked by -verify-diagnostics above.
// The production pass must not introduce runtime loops or partial packing.
// All checks exercise the production pass, including its greedy folding.

//--- unsupported.mlir

// Keep each result live so greedy rewriting cannot erase a rejected contraction.
#mm = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (k, n)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}
#mmt = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// REJECT-LABEL: @scalar_result(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into f32
// REJECT-NEXT: tt.return %[[R]] : f32
// REJECT-NEXT: }
tt.func public @scalar_result(%a: vector<4xbf16>, %b: vector<4xbf16>, %c: f32) -> f32 {
  %0 = vector.contract {
    indexing_maps = [affine_map<(k) -> (k)>, affine_map<(k) -> (k)>, affine_map<(k) -> ()>],
    iterator_types = ["reduction"]
  } %a, %b, %c : vector<4xbf16>, vector<4xbf16> into f32
  tt.return %0 : f32
}

// REJECT-LABEL: @rank3(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2x2xf32>
// REJECT-NEXT: }
tt.func public @rank3(%a: vector<2x2x4xbf16>, %b: vector<2x4x2xbf16>, %c: vector<2x2x2xf32>) -> vector<2x2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(b, m, n, k) -> (b, m, k)>,
                     affine_map<(b, m, n, k) -> (b, k, n)>,
                     affine_map<(b, m, n, k) -> (b, m, n)>],
    iterator_types = ["parallel", "parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<2x2x4xbf16>, vector<2x4x2xbf16> into vector<2x2x2xf32>
  tt.return %0 : vector<2x2x2xf32>
}

// REJECT-LABEL: @scalable(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<[2]x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<[2]x2xf32>
// REJECT-NEXT: }
tt.func public @scalable(%a: vector<[2]x4xbf16>, %b: vector<4x2xbf16>, %c: vector<[2]x2xf32>) -> vector<[2]x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<[2]x4xbf16>, vector<4x2xbf16> into vector<[2]x2xf32>
  tt.return %0 : vector<[2]x2xf32>
}

// REJECT-LABEL: @f16_inputs(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @f16_inputs(%a: vector<2x4xf16>, %b: vector<4x2xf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xf16>, vector<4x2xf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// REJECT-LABEL: @bf16_result(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2xbf16>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xbf16>
// REJECT-NEXT: }
tt.func public @bf16_result(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xbf16>) -> vector<2x2xbf16> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xbf16>
  tt.return %0 : vector<2x2xbf16>
}

// REJECT-LABEL: @mm_attributes(
// REJECT-NEXT: %[[MUL:.*]] = vector.contract {{.*}}kind = #vector.kind<mul>{{.*}}
// REJECT-NEXT: %[[FAST:.*]] = vector.contract {{.*}}fastmath = #arith.fastmath<reassoc>{{.*}}
// REJECT-NEXT: %[[EXTRA:.*]] = vector.contract {{.*}}extra{{.*}}
// REJECT-NEXT: tt.return %[[MUL]], %[[FAST]], %[[EXTRA]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @mm_attributes(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> (vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>) {
  %0 = vector.contract #mm %a, %b, %c {kind = #vector.kind<mul>} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  %1 = vector.contract #mm %a, %b, %c {fastmath = #arith.fastmath<reassoc>} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  %2 = vector.contract #mm %a, %b, %c {extra = unit} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0, %1, %2 : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
}

// REJECT-LABEL: @mmt_attributes(
// REJECT-NEXT: %[[MUL:.*]] = vector.contract {{.*}}kind = #vector.kind<mul>{{.*}}
// REJECT-NEXT: %[[FAST:.*]] = vector.contract {{.*}}fastmath = #arith.fastmath<reassoc>{{.*}}
// REJECT-NEXT: %[[EXTRA:.*]] = vector.contract {{.*}}extra{{.*}}
// REJECT-NEXT: tt.return %[[MUL]], %[[FAST]], %[[EXTRA]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @mmt_attributes(%a: vector<2x4xbf16>, %b: vector<2x4xbf16>, %c: vector<2x2xf32>) -> (vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>) {
  %0 = vector.contract #mmt %a, %b, %c {kind = #vector.kind<mul>} : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
  %1 = vector.contract #mmt %a, %b, %c {fastmath = #arith.fastmath<reassoc>} : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
  %2 = vector.contract #mmt %a, %b, %c {extra = unit} : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
  tt.return %0, %1, %2 : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
}

// REJECT-LABEL: @masked_mm(
// REJECT-NEXT: %[[R:.*]] = vector.mask %{{[^ ]+}} { vector.contract {{.*}} into vector<2x2xf32> } : vector<2x2x4xi1> -> vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @masked_mm(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>, %mask: vector<2x2x4xi1>) -> vector<2x2xf32> {
  %0 = vector.mask %mask {
    vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  } : vector<2x2x4xi1> -> vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// REJECT-LABEL: @masked_mmt(
// REJECT-NEXT: %[[R:.*]] = vector.mask %{{[^ ]+}} { vector.contract {{.*}} into vector<2x2xf32> } : vector<2x2x4xi1> -> vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @masked_mmt(%a: vector<2x4xbf16>, %b: vector<2x4xbf16>, %c: vector<2x2xf32>, %mask: vector<2x2x4xi1>) -> vector<2x2xf32> {
  %0 = vector.mask %mask {
    vector.contract #mmt %a, %b, %c : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
  } : vector<2x2x4xi1> -> vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// REJECT-LABEL: @iterators(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @iterators(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, k, n) -> (m, k)>,
                     affine_map<(m, k, n) -> (k, n)>,
                     affine_map<(m, k, n) -> (m, n)>],
    iterator_types = ["parallel", "reduction", "parallel"]
  } %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// REJECT-LABEL: @maps(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @maps(%a: vector<4x2xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<4x2xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// REJECT-LABEL: @odd_m(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<3x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<3x2xf32>
// REJECT-NEXT: }
tt.func public @odd_m(%a: vector<3x4xbf16>, %b: vector<4x2xbf16>, %c: vector<3x2xf32>) -> vector<3x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<3x4xbf16>, vector<4x2xbf16> into vector<3x2xf32>
  tt.return %0 : vector<3x2xf32>
}

// REJECT-LABEL: @odd_n(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x3xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x3xf32>
// REJECT-NEXT: }
tt.func public @odd_n(%a: vector<2x4xbf16>, %b: vector<4x3xbf16>, %c: vector<2x3xf32>) -> vector<2x3xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x3xbf16> into vector<2x3xf32>
  tt.return %0 : vector<2x3xf32>
}

// REJECT-LABEL: @short_k(
// REJECT-NEXT: %[[R:.*]] = vector.contract {{.*}} into vector<2x2xf32>
// REJECT-NEXT: tt.return %[[R]] : vector<2x2xf32>
// REJECT-NEXT: }
tt.func public @short_k(%a: vector<2x6xbf16>, %b: vector<6x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x6xbf16>, vector<6x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

//--- emitter.mlir

// Check packing through the production pass, including greedy folding.
// Folding removes insertion of a tile that covers the whole result.
// The native tile packs C in row-major order: [C00, C01, C10, C11].
// INSTR-LABEL: @emit_native_tile(
// INSTR-COUNT-1: arm_neon.intr.bfmmla
// INSTR: tt.return
// INSERT-LABEL: @emit_native_tile(
// INSERT-NOT: vector.insert_strided_slice
// INSERT: tt.return
// CAST-LABEL: @emit_native_tile(
// CAST-COUNT-1: vector.shape_cast
// CAST: tt.return
// SHUFFLE-LABEL: @emit_native_tile(
// SHUFFLE-COUNT-3: vector.shuffle
// SHUFFLE: tt.return
// PACK-LABEL: @emit_native_tile(
// PACK-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<2x4xbf16>, %[[C:.*]]: vector<2x2xf32>
// PACK-NEXT: %[[A0:.*]] = vector.extract %[[A]][0] : vector<4xbf16> from vector<2x4xbf16>
// PACK-NEXT: %[[A1:.*]] = vector.extract %[[A]][1] : vector<4xbf16> from vector<2x4xbf16>
// PACK-NEXT: %[[C0:.*]] = vector.extract %[[C]][0] : vector<2xf32> from vector<2x2xf32>
// PACK-NEXT: %[[C1:.*]] = vector.extract %[[C]][1] : vector<2xf32> from vector<2x2xf32>
// PACK-NEXT: %[[B0:.*]] = vector.extract %[[B]][0] : vector<4xbf16> from vector<2x4xbf16>
// PACK-NEXT: %[[B1:.*]] = vector.extract %[[B]][1] : vector<4xbf16> from vector<2x4xbf16>
// PACK-NEXT: %[[ACC:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
// PACK-NEXT: %[[LHS:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// PACK-NEXT: %[[RHS:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// PACK-NEXT: %[[UPDATED:.*]] = arm_neon.intr.bfmmla %[[ACC]], %[[LHS]], %[[RHS]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE:.*]] = vector.shape_cast %[[UPDATED]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: tt.return %[[TILE]] : vector<2x2xf32>
// PACK-NEXT: }
tt.func public @emit_native_tile(%a: vector<2x4xbf16>, %b: vector<2x4xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// K and N differ, and the second row pair checks nonzero outer-loop offsets.
// All accumulator rows must come from the original accumulator argument.
// Each of the six output tiles chains two K updates, then is inserted once.
// INSTR-LABEL: @emit_rectangular_tiles(
// INSTR-COUNT-12: arm_neon.intr.bfmmla
// INSTR: tt.return
// INSERT-LABEL: @emit_rectangular_tiles(
// INSERT-COUNT-6: vector.insert_strided_slice
// INSERT: tt.return
// CAST-LABEL: @emit_rectangular_tiles(
// CAST-COUNT-6: vector.shape_cast
// CAST: tt.return
// SHUFFLE-LABEL: @emit_rectangular_tiles(
// SHUFFLE-COUNT-30: vector.shuffle
// SHUFFLE: tt.return
// PACK-LABEL: @emit_rectangular_tiles(
// PACK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<6x8xbf16>, %[[C:.*]]: vector<4x6xf32>
// PACK-NEXT: %[[A0:.*]] = vector.extract %[[A]][0] : vector<8xbf16> from vector<4x8xbf16>
// PACK-NEXT: %[[A1:.*]] = vector.extract %[[A]][1] : vector<8xbf16> from vector<4x8xbf16>
// PACK-NEXT: %[[C0:.*]] = vector.extract %[[C]][0] : vector<6xf32> from vector<4x6xf32>
// PACK-NEXT: %[[C1:.*]] = vector.extract %[[C]][1] : vector<6xf32> from vector<4x6xf32>
// PACK-NEXT: %[[B0:.*]] = vector.extract %[[B]][0] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B1:.*]] = vector.extract %[[B]][1] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC00:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 6, 7] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS000:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS000:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC000:.*]] = arm_neon.intr.bfmmla %[[ACC00]], %[[LHS000]], %[[RHS000]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS004:.*]] = vector.shuffle %[[A0]], %[[A1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS004:.*]] = vector.shuffle %[[B0]], %[[B1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC004:.*]] = arm_neon.intr.bfmmla %[[ACC000]], %[[LHS004]], %[[RHS004]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE00:.*]] = vector.shape_cast %[[ACC004]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT00:.*]] = vector.insert_strided_slice %[[TILE00]], %[[C]] {offsets = [0, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: %[[B2:.*]] = vector.extract %[[B]][2] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B3:.*]] = vector.extract %[[B]][3] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC02:.*]] = vector.shuffle %[[C0]], %[[C1]] [2, 3, 8, 9] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS020:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS020:.*]] = vector.shuffle %[[B2]], %[[B3]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC020:.*]] = arm_neon.intr.bfmmla %[[ACC02]], %[[LHS020]], %[[RHS020]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS024:.*]] = vector.shuffle %[[A0]], %[[A1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS024:.*]] = vector.shuffle %[[B2]], %[[B3]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC024:.*]] = arm_neon.intr.bfmmla %[[ACC020]], %[[LHS024]], %[[RHS024]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE02:.*]] = vector.shape_cast %[[ACC024]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT02:.*]] = vector.insert_strided_slice %[[TILE02]], %[[RESULT00]] {offsets = [0, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: %[[B4:.*]] = vector.extract %[[B]][4] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B5:.*]] = vector.extract %[[B]][5] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC04:.*]] = vector.shuffle %[[C0]], %[[C1]] [4, 5, 10, 11] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS040:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS040:.*]] = vector.shuffle %[[B4]], %[[B5]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC040:.*]] = arm_neon.intr.bfmmla %[[ACC04]], %[[LHS040]], %[[RHS040]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS044:.*]] = vector.shuffle %[[A0]], %[[A1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS044:.*]] = vector.shuffle %[[B4]], %[[B5]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC044:.*]] = arm_neon.intr.bfmmla %[[ACC040]], %[[LHS044]], %[[RHS044]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE04:.*]] = vector.shape_cast %[[ACC044]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT04:.*]] = vector.insert_strided_slice %[[TILE04]], %[[RESULT02]] {offsets = [0, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: %[[A2:.*]] = vector.extract %[[A]][2] : vector<8xbf16> from vector<4x8xbf16>
// PACK-NEXT: %[[A3:.*]] = vector.extract %[[A]][3] : vector<8xbf16> from vector<4x8xbf16>
// PACK-NEXT: %[[C2:.*]] = vector.extract %[[C]][2] : vector<6xf32> from vector<4x6xf32>
// PACK-NEXT: %[[C3:.*]] = vector.extract %[[C]][3] : vector<6xf32> from vector<4x6xf32>
// PACK-NEXT: %[[B0:.*]] = vector.extract %[[B]][0] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B1:.*]] = vector.extract %[[B]][1] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC20:.*]] = vector.shuffle %[[C2]], %[[C3]] [0, 1, 6, 7] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS200:.*]] = vector.shuffle %[[A2]], %[[A3]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS200:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC200:.*]] = arm_neon.intr.bfmmla %[[ACC20]], %[[LHS200]], %[[RHS200]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS204:.*]] = vector.shuffle %[[A2]], %[[A3]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS204:.*]] = vector.shuffle %[[B0]], %[[B1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC204:.*]] = arm_neon.intr.bfmmla %[[ACC200]], %[[LHS204]], %[[RHS204]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE20:.*]] = vector.shape_cast %[[ACC204]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT20:.*]] = vector.insert_strided_slice %[[TILE20]], %[[RESULT04]] {offsets = [2, 0], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: %[[B2:.*]] = vector.extract %[[B]][2] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B3:.*]] = vector.extract %[[B]][3] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC22:.*]] = vector.shuffle %[[C2]], %[[C3]] [2, 3, 8, 9] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS220:.*]] = vector.shuffle %[[A2]], %[[A3]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS220:.*]] = vector.shuffle %[[B2]], %[[B3]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC220:.*]] = arm_neon.intr.bfmmla %[[ACC22]], %[[LHS220]], %[[RHS220]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS224:.*]] = vector.shuffle %[[A2]], %[[A3]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS224:.*]] = vector.shuffle %[[B2]], %[[B3]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC224:.*]] = arm_neon.intr.bfmmla %[[ACC220]], %[[LHS224]], %[[RHS224]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE22:.*]] = vector.shape_cast %[[ACC224]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT22:.*]] = vector.insert_strided_slice %[[TILE22]], %[[RESULT20]] {offsets = [2, 2], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: %[[B4:.*]] = vector.extract %[[B]][4] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[B5:.*]] = vector.extract %[[B]][5] : vector<8xbf16> from vector<6x8xbf16>
// PACK-NEXT: %[[ACC24:.*]] = vector.shuffle %[[C2]], %[[C3]] [4, 5, 10, 11] : vector<6xf32>, vector<6xf32>
// PACK-NEXT: %[[LHS240:.*]] = vector.shuffle %[[A2]], %[[A3]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS240:.*]] = vector.shuffle %[[B4]], %[[B5]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC240:.*]] = arm_neon.intr.bfmmla %[[ACC24]], %[[LHS240]], %[[RHS240]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[LHS244:.*]] = vector.shuffle %[[A2]], %[[A3]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[RHS244:.*]] = vector.shuffle %[[B4]], %[[B5]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// PACK-NEXT: %[[ACC244:.*]] = arm_neon.intr.bfmmla %[[ACC240]], %[[LHS244]], %[[RHS244]] : vector<8xbf16> to vector<4xf32>
// PACK-NEXT: %[[TILE24:.*]] = vector.shape_cast %[[ACC244]] : vector<4xf32> to vector<2x2xf32>
// PACK-NEXT: %[[RESULT24:.*]] = vector.insert_strided_slice %[[TILE24]], %[[RESULT22]] {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PACK-NEXT: tt.return %[[RESULT24]] : vector<4x6xf32>
// PACK-NEXT: }
tt.func public @emit_rectangular_tiles(%a: vector<4x8xbf16>, %b: vector<6x8xbf16>, %c: vector<4x6xf32>) -> vector<4x6xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<4x8xbf16>, vector<6x8xbf16> into vector<4x6xf32>
  tt.return %0 : vector<4x6xf32>
}

//--- canonicalize.mlir

#mm = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (k, n)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}
#mmt = {
  indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                   affine_map<(m, n, k) -> (n, k)>,
                   affine_map<(m, n, k) -> (m, n)>],
  iterator_types = ["parallel", "parallel", "reduction"]
}

// Check the same output after one and two invocations: normalization is
// selective and idempotent, and rejected contracts retain their semantics.
// CANON-DAG: #[[$LHS_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CANON-DAG: #[[$MM_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CANON-DAG: #[[$ACC_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CANON-DAG: #[[$TRANSPOSED_LHS:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d2, d0)>

// CANON-LABEL: @mm(
// CANON-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<4x2xbf16>, %[[C:.*]]: vector<2x2xf32>
// CANON-NEXT: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<4x2xbf16> to vector<2x4xbf16>
// CANON: %[[B0:.*]] = vector.extract %[[BT]][0] : vector<4xbf16> from vector<2x4xbf16>
// CANON: %[[ACC:.*]] = arm_neon.intr.bfmmla {{.*}} : vector<8xbf16> to vector<4xf32>
// CANON-NEXT: %[[R:.*]] = vector.shape_cast %[[ACC]] : vector<4xf32> to vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @mm(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// Explicit ADD and fastmath=none have the same lowering as absent defaults.
// CANON-LABEL: @mm_defaults(
// CANON-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<4x2xbf16>, %[[C:.*]]: vector<2x2xf32>
// CANON-NEXT: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<4x2xbf16> to vector<2x4xbf16>
// CANON: %[[ACC:.*]] = arm_neon.intr.bfmmla {{.*}} : vector<8xbf16> to vector<4xf32>
// CANON-NEXT: %[[R:.*]] = vector.shape_cast %[[ACC]] : vector<4xf32> to vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @mm_defaults(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c {kind = #vector.kind<add>, fastmath = #arith.fastmath<none>} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// A rectangular RHS exposes swapped N/K dimensions that square tiles can hide.
// CANON-LABEL: @rectangular(
// CANON-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x6xbf16>, %[[C:.*]]: vector<4x6xf32>
// CANON-NEXT: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<8x6xbf16> to vector<6x8xbf16>
// CANON: %[[B0:.*]] = vector.extract %[[BT]][0] : vector<8xbf16> from vector<6x8xbf16>
// CANON-COUNT-12: arm_neon.intr.bfmmla
// CANON: %[[R:.*]] = vector.insert_strided_slice {{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CANON-NEXT: tt.return %[[R]] : vector<4x6xf32>
tt.func public @rectangular(%a: vector<4x8xbf16>, %b: vector<8x6xbf16>, %c: vector<4x6xf32>) -> vector<4x6xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<4x8xbf16>, vector<8x6xbf16> into vector<4x6xf32>
  tt.return %0 : vector<4x6xf32>
}

// Already-MMT uses the original RHS without adding a transpose.
// CANON-LABEL: @mmt(
// CANON-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<6x8xbf16>, %[[C:.*]]: vector<4x6xf32>
// CANON: %[[B0:.*]] = vector.extract %[[B]][0] : vector<8xbf16> from vector<6x8xbf16>
// CANON-COUNT-12: arm_neon.intr.bfmmla
// CANON: %[[R:.*]] = vector.insert_strided_slice {{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CANON-NEXT: tt.return %[[R]] : vector<4x6xf32>
tt.func public @mmt(%a: vector<4x8xbf16>, %b: vector<6x8xbf16>, %c: vector<4x6xf32>) -> vector<4x6xf32> {
  %0 = vector.contract #mmt %a, %b, %c : vector<4x8xbf16>, vector<6x8xbf16> into vector<4x6xf32>
  tt.return %0 : vector<4x6xf32>
}

// Explicit ADD and fastmath=none also remain eligible in MMT form.
// CANON-LABEL: @mmt_defaults(
// CANON-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<6x8xbf16>, %[[C:.*]]: vector<4x6xf32>
// CANON: %[[B0:.*]] = vector.extract %[[B]][0] : vector<8xbf16> from vector<6x8xbf16>
// CANON-COUNT-12: arm_neon.intr.bfmmla
// CANON: %[[R:.*]] = vector.insert_strided_slice {{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// CANON-NEXT: tt.return %[[R]] : vector<4x6xf32>
tt.func public @mmt_defaults(%a: vector<4x8xbf16>, %b: vector<6x8xbf16>, %c: vector<4x6xf32>) -> vector<4x6xf32> {
  %0 = vector.contract #mmt %a, %b, %c {kind = #vector.kind<add>, fastmath = #arith.fastmath<none>} : vector<4x8xbf16>, vector<6x8xbf16> into vector<4x6xf32>
  tt.return %0 : vector<4x6xf32>
}

// CANON-LABEL: @reject_f16(
// CANON-NEXT: %[[R:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} : vector<2x4xf16>, vector<4x2xf16> into vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @reject_f16(%a: vector<2x4xf16>, %b: vector<4x2xf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xf16>, vector<4x2xf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// CANON-LABEL: @reject_bf16_result(
// CANON-NEXT: %[[R:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xbf16>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xbf16>
tt.func public @reject_bf16_result(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xbf16>) -> vector<2x2xbf16> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xbf16>
  tt.return %0 : vector<2x2xbf16>
}

// CANON-LABEL: @reject_short_k(
// CANON-NEXT: %[[R:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} : vector<2x6xbf16>, vector<6x2xbf16> into vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @reject_short_k(%a: vector<2x6xbf16>, %b: vector<6x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract #mm %a, %b, %c : vector<2x6xbf16>, vector<6x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// CANON-LABEL: @reject_maps(
// CANON-NEXT: %[[R:.*]] = vector.contract {indexing_maps = [#[[$TRANSPOSED_LHS]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} : vector<4x2xbf16>, vector<4x2xbf16> into vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @reject_maps(%a: vector<4x2xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (k, m)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<4x2xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// The helper rebuilds attributes: reject non-default or unknown semantics first.
// CANON-LABEL: @reject_attributes(
// CANON-NEXT: %[[MUL:.*]] = vector.contract {{.*}}kind = #vector.kind<mul>{{.*}}
// CANON-NEXT: %[[FAST:.*]] = vector.contract {{.*}}fastmath = #arith.fastmath<reassoc>{{.*}}
// CANON-NEXT: %[[EXTRA:.*]] = vector.contract {{.*}}extra{{.*}}
// CANON-NEXT: tt.return %[[MUL]], %[[FAST]], %[[EXTRA]] : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
tt.func public @reject_attributes(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> (vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>) {
  %0 = vector.contract #mm %a, %b, %c {kind = #vector.kind<mul>} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  %1 = vector.contract #mm %a, %b, %c {fastmath = #arith.fastmath<reassoc>} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  %2 = vector.contract #mm %a, %b, %c {extra = unit} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0, %1, %2 : vector<2x2xf32>, vector<2x2xf32>, vector<2x2xf32>
}

// Inserting a transpose inside vector.mask would invalidate its single-op region.
// CANON-LABEL: @reject_mask(
// CANON-NEXT: %[[R:.*]] = vector.mask %{{[^ ]+}} {
// CANON-SAME: vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
// CANON-SAME: } : vector<2x2x4xi1> -> vector<2x2xf32>
// CANON-NEXT: tt.return %[[R]] : vector<2x2xf32>
tt.func public @reject_mask(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>, %mask: vector<2x2x4xi1>) -> vector<2x2xf32> {
  %0 = vector.mask %mask {
    vector.contract #mm %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  } : vector<2x2x4xi1> -> vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

//--- pipeline.mlir

// ConvertDotGeneric produces ordinary MM. Only the first dot has a native tile.
// The supported dot lowers to BFMMLA without FP32 promotion. The rejected dot
// continues through existing FP32 legalization, retaining its indexing maps.
// MAPS-DAG: #[[$LHS_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// MAPS-DAG: #[[$MM_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// MAPS-DAG: #[[$ACC_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// DISABLED-DAG: #[[$MMT_MAP:[a-zA-Z0-9_]+]] = affine_map<(d0, d1, d2) -> (d1, d2)>

// PIPELINE-LABEL: @mixed_dots(
// PIPELINE-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x6xbf16>, %[[X:.*]]: vector<4x6xbf16>, %[[Y:.*]]: vector<6x6xbf16>, %[[C:.*]]: vector<4x6xf32>
// PIPELINE-NEXT: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<8x6xbf16> to vector<6x8xbf16>
// PIPELINE-COUNT-12: arm_neon.intr.bfmmla
// PIPELINE: %[[ACCEPTED:.*]] = vector.insert_strided_slice {{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// PIPELINE-NEXT: %[[REJECTED:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} %[[X]], %[[Y]], %[[C]] : vector<4x6xbf16>, vector<6x6xbf16> into vector<4x6xf32>
// PIPELINE-NEXT: tt.return %[[ACCEPTED]], %[[REJECTED]] : vector<4x6xf32>, vector<4x6xf32>

// FALLBACK-LABEL: @mixed_dots(
// FALLBACK-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x6xbf16>, %[[X:.*]]: vector<4x6xbf16>, %[[Y:.*]]: vector<6x6xbf16>, %[[C:.*]]: vector<4x6xf32>
// FALLBACK-NEXT: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<8x6xbf16> to vector<6x8xbf16>
// FALLBACK-COUNT-12: arm_neon.intr.bfmmla
// FALLBACK: %[[ACCEPTED:.*]] = vector.insert_strided_slice {{.*}} {offsets = [2, 4], strides = [1, 1]} : vector<2x2xf32> into vector<4x6xf32>
// FALLBACK-NEXT: %[[XF:.*]] = arith.extf %[[X]] : vector<4x6xbf16> to vector<4x6xf32>
// FALLBACK-NEXT: %[[YF:.*]] = arith.extf %[[Y]] : vector<6x6xbf16> to vector<6x6xf32>
// FALLBACK-NEXT: %[[REJECTED:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} %[[XF]], %[[YF]], %[[C]] : vector<4x6xf32>, vector<6x6xf32> into vector<4x6xf32>
// FALLBACK-NEXT: tt.return %[[ACCEPTED]], %[[REJECTED]] : vector<4x6xf32>, vector<4x6xf32>

// When the hardware-gated pass is not scheduled, both dots use FP32 fallback.
// DISABLED-LABEL: @mixed_dots(
// DISABLED-SAME: %[[A:.*]]: vector<4x8xbf16>, %[[B:.*]]: vector<8x6xbf16>, %[[X:.*]]: vector<4x6xbf16>, %[[Y:.*]]: vector<6x6xbf16>, %[[C:.*]]: vector<4x6xf32>
// DISABLED-NEXT: %[[AF:.*]] = arith.extf %[[A]] : vector<4x8xbf16> to vector<4x8xf32>
// DISABLED-NEXT: %[[BF:.*]] = arith.extf %[[B]] : vector<8x6xbf16> to vector<8x6xf32>
// DISABLED-NEXT: %[[ACCEPTED:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} %[[AF]], %[[BF]], %[[C]] : vector<4x8xf32>, vector<8x6xf32> into vector<4x6xf32>
// DISABLED-NEXT: %[[XF:.*]] = arith.extf %[[X]] : vector<4x6xbf16> to vector<4x6xf32>
// DISABLED-NEXT: %[[YF:.*]] = arith.extf %[[Y]] : vector<6x6xbf16> to vector<6x6xf32>
// DISABLED-NEXT: %[[REJECTED:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MM_MAP]], #[[$ACC_MAP]]]{{.*}} %[[XF]], %[[YF]], %[[C]] : vector<4x6xf32>, vector<6x6xf32> into vector<4x6xf32>
// DISABLED-NEXT: tt.return %[[ACCEPTED]], %[[REJECTED]] : vector<4x6xf32>, vector<4x6xf32>
tt.func public @mixed_dots(%a: vector<4x8xbf16>, %b: vector<8x6xbf16>, %x: vector<4x6xbf16>, %y: vector<6x6xbf16>, %c: vector<4x6xf32>) -> (vector<4x6xf32>, vector<4x6xf32>) {
  %0 = triton_cpu.dot %a, %b, %c, inputPrecision = ieee : vector<4x8xbf16> * vector<8x6xbf16> -> vector<4x6xf32>
  %1 = triton_cpu.dot %x, %y, %c, inputPrecision = ieee : vector<4x6xbf16> * vector<6x6xbf16> -> vector<4x6xf32>
  tt.return %0, %1 : vector<4x6xf32>, vector<4x6xf32>
}

// Already-MMT input is lowered directly, with two dependent BFMMLA updates.
// PIPELINE-LABEL: @mmt_k8(
// PIPELINE-SAME: %[[A:.*]]: vector<2x8xbf16>, %[[B:.*]]: vector<2x8xbf16>, %[[C:.*]]: vector<2x2xf32>
// PIPELINE: %[[ACC0:.*]] = arm_neon.intr.bfmmla {{.*}} : vector<8xbf16> to vector<4xf32>
// PIPELINE: %[[ACC1:.*]] = arm_neon.intr.bfmmla %[[ACC0]], {{.*}} : vector<8xbf16> to vector<4xf32>
// PIPELINE-NEXT: %[[R:.*]] = vector.shape_cast %[[ACC1]] : vector<4xf32> to vector<2x2xf32>
// PIPELINE-NEXT: tt.return %[[R]] : vector<2x2xf32>
// PIPELINE-NEXT: }

// FALLBACK-LABEL: @mmt_k8(
// FALLBACK-SAME: %[[A:.*]]: vector<2x8xbf16>, %[[B:.*]]: vector<2x8xbf16>, %[[C:.*]]: vector<2x2xf32>
// FALLBACK: %[[ACC0:.*]] = arm_neon.intr.bfmmla {{.*}} : vector<8xbf16> to vector<4xf32>
// FALLBACK: %[[ACC1:.*]] = arm_neon.intr.bfmmla %[[ACC0]], {{.*}} : vector<8xbf16> to vector<4xf32>
// FALLBACK-NEXT: %[[R:.*]] = vector.shape_cast %[[ACC1]] : vector<4xf32> to vector<2x2xf32>
// FALLBACK-NEXT: tt.return %[[R]] : vector<2x2xf32>
// FALLBACK-NEXT: }

// DISABLED-LABEL: @mmt_k8(
// DISABLED-SAME: %[[A:.*]]: vector<2x8xbf16>, %[[B:.*]]: vector<2x8xbf16>, %[[C:.*]]: vector<2x2xf32>
// DISABLED-NEXT: %[[AF:.*]] = arith.extf %[[A]] : vector<2x8xbf16> to vector<2x8xf32>
// DISABLED-NEXT: %[[BF:.*]] = arith.extf %[[B]] : vector<2x8xbf16> to vector<2x8xf32>
// DISABLED-NEXT: %[[R:.*]] = vector.contract {indexing_maps = [#[[$LHS_MAP]], #[[$MMT_MAP]], #[[$ACC_MAP]]]{{.*}} %[[AF]], %[[BF]], %[[C]] : vector<2x8xf32>, vector<2x8xf32> into vector<2x2xf32>
// DISABLED-NEXT: tt.return %[[R]] : vector<2x2xf32>
// DISABLED-NEXT: }
tt.func public @mmt_k8(%a: vector<2x8xbf16>, %b: vector<2x8xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<2x8xbf16>, vector<2x8xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

//--- production-emitter.mlir

// Exercise normalization, emission, and replacement through the public pass.
// Reparse its output to check ArmNeon registration without scheduling the pass.
// Greedy folding removes the full-size tile insert for these 2x2 results.

// Ordinary MM needs one RHS transpose and one native 2x2x4 BFMMLA update.
// CHECK-LABEL: @mm(
// CHECK-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<4x2xbf16>, %[[C:.*]]: vector<2x2xf32>
// CHECK: %[[BT:.*]] = vector.transpose %[[B]], [1, 0] : vector<4x2xbf16> to vector<2x4xbf16>
// CHECK-DAG: %[[A0:.*]] = vector.extract %[[A]][0] : vector<4xbf16> from vector<2x4xbf16>
// CHECK-DAG: %[[A1:.*]] = vector.extract %[[A]][1] : vector<4xbf16> from vector<2x4xbf16>
// CHECK-DAG: %[[B0:.*]] = vector.extract %[[BT]][0] : vector<4xbf16> from vector<2x4xbf16>
// CHECK-DAG: %[[B1:.*]] = vector.extract %[[BT]][1] : vector<4xbf16> from vector<2x4xbf16>
// CHECK-DAG: %[[C0:.*]] = vector.extract %[[C]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[C1:.*]] = vector.extract %[[C]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[ACC:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
// CHECK-DAG: %[[LHS:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// CHECK-DAG: %[[RHS:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 1, 2, 3, 4, 5, 6, 7] : vector<4xbf16>, vector<4xbf16>
// CHECK: %[[R:.*]] = arm_neon.intr.bfmmla %[[ACC]], %[[LHS]], %[[RHS]] : vector<8xbf16> to vector<4xf32>
// CHECK-NEXT: %[[OUT:.*]] = vector.shape_cast %[[R]] : vector<4xf32> to vector<2x2xf32>
// CHECK-NEXT: tt.return %[[OUT]] : vector<2x2xf32>
// CHECK-NEXT: }

tt.func public @mm(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (k, n)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<2x4xbf16>, vector<4x2xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// -----

// Already-MMT inputs need no transpose. K=8 requires two chained updates.
// CHECK-LABEL: @mmt_k8(
// CHECK-SAME: %[[A:.*]]: vector<2x8xbf16>, %[[B:.*]]: vector<2x8xbf16>, %[[C:.*]]: vector<2x2xf32>
// CHECK-DAG: %[[A0:.*]] = vector.extract %[[A]][0] : vector<8xbf16> from vector<2x8xbf16>
// CHECK-DAG: %[[A1:.*]] = vector.extract %[[A]][1] : vector<8xbf16> from vector<2x8xbf16>
// CHECK-DAG: %[[B0:.*]] = vector.extract %[[B]][0] : vector<8xbf16> from vector<2x8xbf16>
// CHECK-DAG: %[[B1:.*]] = vector.extract %[[B]][1] : vector<8xbf16> from vector<2x8xbf16>
// CHECK-DAG: %[[C0:.*]] = vector.extract %[[C]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[C1:.*]] = vector.extract %[[C]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[ACC:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
// CHECK-DAG: %[[LHS0:.*]] = vector.shuffle %[[A0]], %[[A1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// CHECK-DAG: %[[RHS0:.*]] = vector.shuffle %[[B0]], %[[B1]] [0, 1, 2, 3, 8, 9, 10, 11] : vector<8xbf16>, vector<8xbf16>
// CHECK-DAG: %[[R0:.*]] = arm_neon.intr.bfmmla %[[ACC]], %[[LHS0]], %[[RHS0]] : vector<8xbf16> to vector<4xf32>
// CHECK-DAG: %[[LHS1:.*]] = vector.shuffle %[[A0]], %[[A1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// CHECK-DAG: %[[RHS1:.*]] = vector.shuffle %[[B0]], %[[B1]] [4, 5, 6, 7, 12, 13, 14, 15] : vector<8xbf16>, vector<8xbf16>
// CHECK: %[[R1:.*]] = arm_neon.intr.bfmmla %[[R0]], %[[LHS1]], %[[RHS1]] : vector<8xbf16> to vector<4xf32>
// CHECK-NEXT: %[[OUT:.*]] = vector.shape_cast %[[R1]] : vector<4xf32> to vector<2x2xf32>
// CHECK-NEXT: tt.return %[[OUT]] : vector<2x2xf32>
// CHECK-NEXT: }

tt.func public @mmt_k8(%a: vector<2x8xbf16>, %b: vector<2x8xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = vector.contract {
    indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                     affine_map<(m, n, k) -> (n, k)>,
                     affine_map<(m, n, k) -> (m, n)>],
    iterator_types = ["parallel", "parallel", "reduction"]
  } %a, %b, %c : vector<2x8xbf16>, vector<2x8xbf16> into vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// -----

// The same lowering must be reachable after ConvertDotGeneric.
// CHECK-LABEL: @dot(
// CHECK-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<4x2xbf16>, %[[C:.*]]: vector<2x2xf32>
// CHECK: vector.transpose %[[B]], [1, 0] : vector<4x2xbf16> to vector<2x4xbf16>
// CHECK-DAG: %[[C0:.*]] = vector.extract %[[C]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[C1:.*]] = vector.extract %[[C]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[ACC:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
// CHECK: %[[R:.*]] = arm_neon.intr.bfmmla %[[ACC]], %{{.*}}, %{{.*}} : vector<8xbf16> to vector<4xf32>
// CHECK-NEXT: %[[OUT:.*]] = vector.shape_cast %[[R]] : vector<4xf32> to vector<2x2xf32>
// CHECK-NEXT: tt.return %[[OUT]] : vector<2x2xf32>
// CHECK-NEXT: }

tt.func public @dot(%a: vector<2x4xbf16>, %b: vector<4x2xbf16>, %c: vector<2x2xf32>) -> vector<2x2xf32> {
  %0 = triton_cpu.dot %a, %b, %c, inputPrecision = ieee : vector<2x4xbf16> * vector<4x2xbf16> -> vector<2x2xf32>
  tt.return %0 : vector<2x2xf32>
}

// A loop-carried accumulator must be packed from the current iteration's
// argument, not reset to zero or to the loop's nonzero initial value.
// CHECK-LABEL: @loop_carried_accumulator(
// CHECK-SAME: %[[A:.*]]: vector<2x4xbf16>, %[[B:.*]]: vector<2x4xbf16>, %[[COUNT:.*]]: i32
// CHECK: %[[INITIAL:.*]] = arith.constant dense<1.000000e+00> : vector<2x2xf32>
// CHECK: %[[RESULT:.*]] = scf.for %{{.*}} = %{{.*}} to %[[COUNT]] step %{{.*}} iter_args(%[[ACC:.*]] = %[[INITIAL]]) -> (vector<2x2xf32>)
// CHECK-DAG: %[[C0:.*]] = vector.extract %[[ACC]][0] : vector<2xf32> from vector<2x2xf32>
// CHECK-DAG: %[[C1:.*]] = vector.extract %[[ACC]][1] : vector<2xf32> from vector<2x2xf32>
// CHECK: %[[PACKED:.*]] = vector.shuffle %[[C0]], %[[C1]] [0, 1, 2, 3] : vector<2xf32>, vector<2xf32>
// CHECK: %[[UPDATED:.*]] = arm_neon.intr.bfmmla %[[PACKED]], %{{.*}}, %{{.*}} : vector<8xbf16> to vector<4xf32>
// CHECK-NEXT: %[[TILE:.*]] = vector.shape_cast %[[UPDATED]] : vector<4xf32> to vector<2x2xf32>
// CHECK-NEXT: scf.yield %[[TILE]] : vector<2x2xf32>
// CHECK-NEXT: }
// CHECK-NEXT: tt.return %[[RESULT]] : vector<2x2xf32>
tt.func public @loop_carried_accumulator(%a: vector<2x4xbf16>, %b: vector<2x4xbf16>, %count: i32) -> vector<2x2xf32> {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %initial = arith.constant dense<1.0> : vector<2x2xf32>
  %result = scf.for %i = %zero to %count step %one iter_args(%acc = %initial) -> (vector<2x2xf32>) : i32 {
    %next = vector.contract {
      indexing_maps = [affine_map<(m, n, k) -> (m, k)>,
                       affine_map<(m, n, k) -> (n, k)>,
                       affine_map<(m, n, k) -> (m, n)>],
      iterator_types = ["parallel", "parallel", "reduction"]
    } %a, %b, %acc : vector<2x4xbf16>, vector<2x4xbf16> into vector<2x2xf32>
    scf.yield %next : vector<2x2xf32>
  }
  tt.return %result : vector<2x2xf32>
}
