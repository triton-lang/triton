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

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

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

def PT_RecordOp : PT_Op<"record", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>
  ]> {
  let summary = "Record a GPU metric event";

  let description = [{
    The operator records GPU event of a particular metric.
    Essentially a marker with a region id.

    Example:

    ```mlir
    proton.record() {isStart = true, regionId = 4 : i32}
    ...
    proton.record() {isStart = false, regionId = 4 : i32}
    ```
  }];
  let arguments = (
    ins BoolAttr: $isStart,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$regionId
  );

  let assemblyFormat = " `(` operands `)` attr-dict";
}

def PT_CircularRecordOp : PT_Op<"circular_record", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  MemoryEffects<[MemWrite<DefaultResource>]>
  ]> {
  let summary = "Record a GPU metric event into a circular buffer";

  let description = [{
    Records a metric event into a circular buffer backed by the internal memory `data`.
    The circular buffer indexing `indexPtr` is automatically maintained. Older events
    get dropped if the `data` is full.
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

def PT_FinalizeOp : PT_Op<"finalize", [
  MemoryEffects<[MemRead<SharedMemory>]>,
  MemoryEffects<[MemRead<GlobalMemory>]>,
  MemoryEffects<[MemWrite<GlobalMemory>]>
  ]> {
  let summary = "Finalize the intra kernel profiler";

  let description = [{
    Finalize the intra kernel profiler, writing back the metadata and profile to the global memory.
  }];
  let arguments = (
    ins TTG_MemDescType:$data,
    TT_PtrLike :$indexPtr,
    TT_PtrLike :$ptr,
    I32Attr :$size
  );

  let assemblyFormat = [{$data `,` $indexPtr `,` $ptr attr-dict `:` qualified(type($data)) `,` type($indexPtr) `,` type($ptr)}];
}

def PT_InitOp : PT_Op<"init", [
  MemoryEffects<[MemAlloc<GlobalMemory>]>
  ]> {
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
