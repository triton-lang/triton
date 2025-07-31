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
include "proton/Dialect/include/Dialect/Proton/IR/ProtonAttrDefs.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUDialect.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUAttrDefs.td"
include "proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUTypes.td"

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
    Store a metric `counter` into a circular buffer backed by the internal memory `segment`.
    automatically updated. Older metric counters are dropped if the `segment` buffer is full.
  }];

  let arguments = (ins
    PTG_SegmentType:$segment,
    AnyTypeOf<[I32, I64]>:$counter,
    UnitAttr:$isStart,
    I32Attr:$scopeId
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    (`start` $isStart^):(`end`)? $segment `,` $counter attr-dict `:`
    qualified(type($segment)) `,` type($counter)
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

  let results = (outs AnyTypeOf<[I32, I64]>:$counter);

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
    `segment` is the segment of the internal profiling buffer that contains the profiling data.
    `scratchPtr` is the address of the profiling scratch buffer.
  }];

  let arguments = (ins
    PTG_SegmentType:$segment,
    TT_Ptr:$scratchPtr
  );

  let assemblyFormat = [{
    $segment `,` $scratchPtr attr-dict `:` qualified(type($segment)) `,` qualified(type($scratchPtr))
  }];
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

def PTG_SegmentAllocOp : PTG_Op<"segment_alloc", [Pure]> {
  let summary = "Get the base offset of the segment of the internal buffer";

  let description = [{
    The internal buffer is partitioned into segments for each profiling "unit".
    This operation gets the location of the memory segment in the internal buffer.
  }];

  let arguments = (ins
    TTG_MemDescType:$buffer
  );

  let results = (outs PTG_SegmentType:$segment);

  let hasVerifier = 1;

  let assemblyFormat = "$buffer attr-dict `:` qualified(type($buffer)) `->` type($segment)";
}


def PTG_InitCtxOp : PTG_Op<"init_ctx", [
    MemoryEffects<[MemWrite<GlobalMemory>]>
]> {
  let summary = "Initialize the intra kernel profiler warp-level contexts";

  let description = [{
    Initialize the intra kernel profiler warp-level contexts for all warps in
    `scratchPtr` (base address of the profiling scratch buffer). It can't be
    called inside `ttg.warp_specialize`.
  }];

  let arguments = (ins
    TT_Ptr:$scratchPtr
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    $scratchPtr attr-dict `:` qualified(type($scratchPtr))
  }];
}

def PTG_RestoreCtxOp : PTG_Op<"restore_ctx", [
    MemoryEffects<[MemRead<GlobalMemory>]>,
    MemoryEffects<[MemWrite<GlobalMemory>]>
]> {
  let summary = "Restore the current warp-level context";

  let description = [{
    Restore the current warp context in `$segment` from
    `scratchPtr` (base address of the profiling scratch buffer).
  }];

  let arguments = (ins
    PTG_SegmentType:$segment,
    TT_Ptr:$scratchPtr
  );

  let assemblyFormat = [{
    $segment `,` $scratchPtr attr-dict `:` qualified(type($segment)) `,` qualified(type($scratchPtr))
  }];
}

def PTG_SaveCtxOp : PTG_Op<"save_ctx", [
    MemoryEffects<[MemWrite<GlobalMemory>]>
]> {
  let summary = "Save the current warp-level context";

  let description = [{
    Save the current warp context from `$segment` to
    `scratchPtr` (base address of the profiling scratch buffer).
  }];

  let arguments = (ins
    PTG_SegmentType:$segment,
    TT_Ptr:$scratchPtr
  );

  let assemblyFormat = [{
    $segment `,` $scratchPtr attr-dict `:` qualified(type($segment)) `,` qualified(type($scratchPtr))
  }];
}

#endif  // PROTONGPU_OPS
