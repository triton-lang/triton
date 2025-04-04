#ifndef PROTONGPU_OPS
#define PROTONGPU_OPS

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
include "proton/dialect/include/Dialect/ProtonGPU/IR/ProtonGPUTypes.td"

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
    The circular buffer indexing is stored in the address of `indexPtr`. Index value is
    automatically updated. Older metric counter get dropped if the `data` is full.
  }];
  let arguments = (
    ins TTG_MemDescType: $data,
    TT_PtrType :$indexPtr,
    I32: $counter,
    PTG_SegmentBase:$seg,
    UnitAttr: $isStart,
    I32Attr: $scopeId
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $data `,` $indexPtr `,` $counter `,` $seg attr-dict `:`
    qualified(type($data)) `,` type($indexPtr) `,` type($counter) `,` type($seg)
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
    TT_PtrType :$indexPtr,
    TT_Ptr :$ptr
  );

  let assemblyFormat = [{$data `,` $indexPtr `,` $ptr attr-dict `:` qualified(type($data)) `,` type($indexPtr) `,` type($ptr)}];
}

def PTG_InitBufferIndexOp : PTG_Op<"init_buffer_index", []> {
  let summary = "Initialize the internal buffer index";

  let description = [{
      Zero-initialization for the internal buffer index for the intra kernel profiler.
      The index is stored in the address of `indexPtr`.
  }];
  let arguments = (ins);
  let results = (outs TT_PtrType :$indexPtr);
  let assemblyFormat = "attr-dict `:` type($indexPtr)";
}

def PTG_GlobalScratchAllocOp : PTG_Op<"global_scratch_alloc", [MemoryEffects<[MemAlloc<GlobalMemory>]>]> {
  let summary = "Allocate a proton global memory profiling buffer";
  let description = [{
    This operation allocates a proton buffer in global memory, which holds the device-side profiling data of the current program.
  }];
  let arguments = (
    ins
    I32Attr:$nbytes,
    I32Attr:$alignment
  );
  let results = (outs TT_Ptr:$result);

  let builders = [
    OpBuilder<(ins "Type":$result, "int32_t":$nbytes, "int32_t":$alignment),
              [{ build($_builder, $_state, result,
                       $_builder.getI32IntegerAttr(nbytes), $_builder.getI32IntegerAttr(alignment)); }]>
  ];

  let assemblyFormat = [{attr-dict `:` qualified(type($result))}];
}

def PTG_SegmentBaseOp : PTG_Op<"segment_base", [Pure]> {
  let summary = "Get the base offset of the segment of the profiler internal buffer";

  let description = [{
      Get the location of the memory segment in the profiling buffer.
      The internal profiler buffer is partitioned into segments.
      If `selectIds` is none, then it is the full list of ids.
  }];
  let arguments = (ins
    TTG_MemDescType: $data,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARP">:$granularity,
    OptionalAttr<DenseI32ArrayAttr>:$selectIds
  );
  let results = (outs PTG_SegmentBase:$segBaseOffset);
  let hasVerifier = 1;
  let assemblyFormat = "$data `,` attr-dict `:` qualified(type($data)) `->` type($segBaseOffset)";
}

def PTG_StackAllocOp : PTG_Op<"stack_alloc", [MemoryEffects<[MemAlloc<GlobalMemory>]>]> {
  let summary = "Allocate a proton stack memory buffer";
  let description = [{
    This operation allocates a proton buffer in device stack memory.
  }];
  let arguments = (
    ins
  );
  let results = (outs TTG_MemDescType : $data);
  let assemblyFormat = "attr-dict `:` qualified(type($data))";
}

#endif  // PROTONGPU_OPS
