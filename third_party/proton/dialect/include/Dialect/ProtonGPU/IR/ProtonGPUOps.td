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

def PTG_CircularStoreOp : PTG_Op<"circular_store", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Store the value into a circular buffer";

  let description = [{
    Store a metric `counter` into a circular buffer backed by the internal memory `data`.
    The circular buffer indexing `curIndex` is automatically updated and returned as `newIndex`.
    Older metric counter get dropped if the `data` is full.
  }];
  let arguments = (
    ins TTG_MemDescType: $data,
    I32: $curIndex,
    I32: $counter,
    UnitAttr: $isStart,
    I32Attr: $scopeId,
    DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARPGROUP">:$granularity
  );

  let results = (outs I32 : $newIndex);

  let hasVerifier = 1;

  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $data `,` $curIndex `,` $counter attr-dict `:`
    qualified(type($data)) `,` type($curIndex) `,` type($counter) `->` type($newIndex)
  }];
}

def PTG_ReadCounterOp : PTG_Op<"read_counter", [
  MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Read a GPU metric counter into a scalar register";

  let description = [{
    Read a GPU metric counter into a scalar register.
  }];
  let arguments = (
    ins DefaultValuedAttr<MetricAttr, "Metric::CYCLE">:$metric
  );
  let results = (outs I32 : $counter);

  let assemblyFormat = [{
    attr-dict `:` type($counter)
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
    I32 :$index,
    TT_PtrLike :$ptr,
    I32Attr :$size
  );

  let assemblyFormat = [{$data `,` $index `,` $ptr attr-dict `:` qualified(type($data)) `,` type($index) `,` type($ptr)}];
}

def PTG_InitBufferIndexOp : PTG_Op<"init_buffer_index", []> {
  let summary = "Initialize the internal buffer index";

  let description = [{
      Zero-initialization for the internal buffer index for the intra kernel profiler.
  }];
  let arguments = (ins);
  let results = (outs I32 :$index);
  let assemblyFormat = "attr-dict `:` type($index)";
}

#endif  // PROTON_GPU_OPS
