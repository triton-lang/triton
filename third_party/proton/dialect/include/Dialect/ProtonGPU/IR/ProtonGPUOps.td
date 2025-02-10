#ifndef PROTON_GPU_OPS
#define PROTON_GPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "proton/dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td"

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

class PTG_Op<string mnemonic, list<Trait> traits = []> :
  Op<ProtonGPU_Dialect, mnemonic, !listconcat(traits, [])> {
}

def PTG_CircularRecordOp : PTG_Op<"circular_record", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Record a GPU metric event into a circular buffer";

  let description = [{
    Records a metric event into a circular buffer backed by the internal memory `data`.
    The circular buffer indexing `indexPtr` is automatically maintained. Older events
    get dropped if the `data` is full.
  }];
  let arguments = (
    ins UnitAttr: $isStart,
    I32: $scopeId,
    TTG_MemDescType:$data,
    TT_PtrLike :$indexPtr,
    DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARPGROUP">:$granularity
  );
  let hasVerifier = 1;

  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $scopeId `,` $data `,` $indexPtr attr-dict `:`
    qualified(type($data)) `,` type($indexPtr)
  }];
}

def PTG_FinalizeOp : PTG_Op<"finalize", [
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

def PTG_InitOp : PTG_Op<"init", [
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

#endif  // PROTON_GPU_OPS
