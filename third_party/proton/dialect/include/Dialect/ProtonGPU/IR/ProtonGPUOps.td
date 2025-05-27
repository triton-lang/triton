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

//===----------------------------------------------------------------------===//
// Resources
//===----------------------------------------------------------------------===//

def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

//===----------------------------------------------------------------------===//
// Base Class
//===----------------------------------------------------------------------===//

class PTG_Op<string mnemonic, list<Trait> traits = []> :
    Op<ProtonGPU_Dialect, mnemonic, !listconcat(traits, [])> {
}

//===----------------------------------------------------------------------===//
// ProtonGPU Operations
//===----------------------------------------------------------------------===//

def PTG_CircularStoreOp : PTG_Op<"circular_store", [
    MemoryEffects<[MemRead<DefaultResource>, MemWrite<DefaultResource>]>
]> {
  let summary = "Store the value into a circular buffer";

  let description = [{
    Store a metric `counter` into a circular buffer backed by the internal memory `data`.
    The circular buffer indexing is stored at the address of `indexPtr`. The index value is
    automatically updated. Older metric counters are dropped if the `data` buffer is full.
  }];

  let arguments = (ins
    TTG_MemDescType:$data,
    TT_Ptr:$indexPtr,
    I32:$counter,
    PTG_SegmentBase:$seg,
    UnitAttr:$isStart,
    I32Attr:$scopeId
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

  let arguments = (ins
    DefaultValuedAttr<MetricTypeAttr, "MetricType::CYCLE">:$metric
  );

  let results = (outs I32:$counter);

  let assemblyFormat = [{
    attr-dict `:` type($counter)
  }];
}

def PTG_FinalizeOp : PTG_Op<"finalize", [
    MemoryEffects<[MemRead<SharedMemory>]>, // FIXME: it shouldn't always have shared memory effects
    MemoryEffects<[MemRead<GlobalMemory>]>,
    MemoryEffects<[MemWrite<GlobalMemory>]>
]> {
  let summary = "Finalize the intra kernel profiler";

  let description = [{
    Write back the metadata and profile to global memory.
    `data` is the internal profiling buffer that contains the profiling data.
    `indexPtr` is the address of the internal buffer index, indicating the current position in the internal profiling buffer.
    `scratchPtr` is the address of the profile scratch buffer.
  }];

  let arguments = (ins
    TTG_MemDescType:$data,
    TT_Ptr:$indexPtr,
    TT_Ptr:$scratchPtr
  );

  let assemblyFormat = [{
    $data `,` $indexPtr `,` $scratchPtr attr-dict `:` qualified(type($data)) `,` type($indexPtr) `,` type($scratchPtr)
  }];
}

def PTG_InitBufferIndexOp : PTG_Op<"init_buffer_index", []> {
  let summary = "Initialize the internal buffer index";

  let description = [{
    Zero-initialize the internal buffer index.
    The index is stored at the address of `indexPtr`.
  }];

  let arguments = (ins);

  let results = (outs TT_Ptr:$indexPtr);

  let assemblyFormat = "attr-dict `:` type($indexPtr)";
}

def PTG_GlobalScratchAllocOp : PTG_Op<"global_scratch_alloc", [
    MemoryEffects<[MemAlloc<GlobalMemory>]>
]> {
  let summary = "Allocate a global memory profile buffer";

  let description = [{
    This operation allocates a profile buffer in global memory.
    Each Triton program being profiled will hold a pointer to the buffer until profiling is finalized.
  }];

  let arguments = (ins
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
  let summary = "Get the base offset of the segment of the internal buffer";

  let description = [{
    The internal buffer is partitioned into segments for each profiling "unit".
    This operation gets the location of the memory segment in the internal buffer.
    `granularity` indicates the granularity of the profiling function unit.
    If `selectIds` is none, then all available function units will collect profiling data.
  }];

  let arguments = (ins
    TTG_MemDescType:$data,
    DefaultValuedAttr<GranularityAttr, "Granularity::WARP">:$granularity,
    OptionalAttr<DenseI32ArrayAttr>:$selectIds
  );

  let results = (outs PTG_SegmentBase:$segBaseOffset);

  let hasVerifier = 1;

  let assemblyFormat = "$data `,` attr-dict `:` qualified(type($data)) `->` type($segBaseOffset)";
}

def PTG_StackAllocOp : PTG_Op<"stack_alloc", [
    MemoryEffects<[MemAlloc<GlobalMemory>]>
]> {
  let summary = "Allocate a proton buffer in stack memory";

  let description = [{
    This operation allocates a proton buffer in device stack memory.
  }];

  let arguments = (ins);

  let hasVerifier = 1;

  let results = (outs TTG_MemDescType:$data);

  let assemblyFormat = "attr-dict `:` qualified(type($data))";
}

#endif  // PROTONGPU_OPS
