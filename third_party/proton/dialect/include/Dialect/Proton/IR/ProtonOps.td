#ifndef PROTON_OPS
#define PROTON_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "ProtonDialect.td"
include "ProtonAttrDefs.td"

class PT_Op<string mnemonic, list<Trait> traits = []> :
    Op<Proton_Dialect, mnemonic, !listconcat(traits, [])> {
}

// Proton profiling metric.
def MetricAttr : I32EnumAttr<
    "Metric", "",
    [
        I32EnumAttrCase<"CYCLE", 0, "cycle">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

// Proton profiling granularity.
def GranularityAttr : I32EnumAttr<
    "Granularity", "",
    [
        I32EnumAttrCase<"WARPGROUP", 0, "warpgroup">,
        I32EnumAttrCase<"WARP", 1, "warp">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

// Proton profiling backend strategy.
def StrategyAttr : I32EnumAttr<
    "Strategy", "",
    [
        I32EnumAttrCase<"CIRCULAR", 0, "circular">,
    ]> {
    let cppNamespace = "::mlir::triton::proton";
}

def PT_RecordOp : PT_Op<"record", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Record a GPU hardware event";

  let description = [{
    The operator records GPU events from performance counters.
    Currently only cycle counter is supported.

    Example:

    ```mlir
    proton.record() {isStart = true, regionId = 4 : i32}
    ...
    proton.record() {isStart = false, regionId = 4 : i32}
    ...
    proton.record() {isStart = true, regionId = 1 : i32, granularity = 1 : i32}
    ...
    proton.record() {isStart = false, regionId = 1 : i32, granularity = 1 : i32}
    ```
  }];
  let arguments = (
    ins BoolAttr: $isStart,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$regionId,
    DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARPGROUP">:$granularity,
    DefaultValuedAttr<StrategyAttr, "Strategy::CIRCULAR">:$strategy
  );

  let assemblyFormat = " `(` operands `)` attr-dict";
}

def PT_CircularRecordOp : PT_Op<"circular_record", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Record a GPU hardware event into a circular buffer";

  let description = [{
    The intra kernel profiler records an event into a circular buffer backed by the shared memory `data`.
    The circular buffer indexing `indexPtr` is automatically maintained.
  }];
  let arguments = (
    ins TTG_MemDescType:$data,
    TT_PtrLike :$indexPtr,
    BoolAttr: $isStart,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$regionId,
    DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARPGROUP">:$granularity
  );
  let hasVerifier = 1;

  let assemblyFormat = [{$data `,` $indexPtr attr-dict `:` qualified(type($data)) `,` type($indexPtr)}];
}

def PT_FinalizeOp : PT_Op<"finalize", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Finalize the intra kernel profiler";

  let description = [{
    Finalize the intra kernel profiler, writing back the metadata and measurements to the global memory.
  }];
  let arguments = (
    ins TTG_MemDescType:$data,
    TT_PtrLike :$indexPtr,
    TT_PtrLike :$ptr,
    I32Attr :$size
  );

  let assemblyFormat = [{$data `,` $indexPtr `,` $ptr attr-dict `:` qualified(type($data)) `,` type($indexPtr) `,` type($ptr)}];
}

def PT_InitOp : PT_Op<"init", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
    let summary = "Initialize the intra kernel profiler";

    let description = [{
        Stack allocation and initialization for the intra kernel profiler.
        `indexPtr` stores the number of entries proton recorded (zero initialized).
        We expect `indexPtr` to be register promoted during the LLVM lowering phase.
    }];
    let arguments = (ins);
    let results = (outs TT_PtrLike :$indexPtr);
    let assemblyFormat = "attr-dict `:` type($indexPtr)";
}

#endif // PROTON_OPS
