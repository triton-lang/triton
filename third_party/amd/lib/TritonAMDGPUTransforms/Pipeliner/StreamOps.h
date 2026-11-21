#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_STREAMOPS_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_STREAMOPS_H_

#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include <variant>

// Stream chains: the ops that replace a pipelined global load. A chain fills
// one slot of a multi-buffered LDS allocation and, if the loaded value has
// register users, reads the tile back with a local_load.

namespace mlir {

// tt.load -> ttg.local_store into the slot.
struct StreamCopyChainOps {
  triton::LoadOp loadOp;
  triton::gpu::MemDescIndexOp subviewOp;
  triton::gpu::LocalStoreOp localStoreOp;
  triton::gpu::LocalLoadOp maybeLocalLoadOp;
};

// Direct-to-LDS async copy with its commit group and wait.
struct AsyncCopyChainOps {
  triton::gpu::AsyncCopyGlobalToLocalOp copyOp;
  triton::gpu::AsyncCommitGroupOp commitOp;
  triton::gpu::AsyncWaitOp waitOp;
  triton::gpu::LocalLoadOp maybeLocalLoadOp;
};

// Common shape for an async TDM chain (copy or gather). The first op (copy or
// gather) is type-erased because every consumer only needs Operation *.
struct TDMChainOps {
  Operation *tdmOp;
  triton::amdgpu::AsyncTDMWait waitOp;
  triton::gpu::LocalLoadOp maybeLocalLoadOp;
};

using StreamOpVariant =
    std::variant<StreamCopyChainOps, AsyncCopyChainOps, TDMChainOps>;
using LoadToStreamOpMap = llvm::MapVector<Operation *, StreamOpVariant>;

// Each builder replaces the uses of its load with a chain filling slot
// `extractIdx` of `alloc`.
TDMChainOps createTDMAsyncCopy(triton::DescriptorLoadOp loadOp, Value alloc,
                               Value extractIdx);
TDMChainOps createTDMAsyncGather(triton::DescriptorGatherOp gatherOp,
                                 Value alloc, Value extractIdx);
AsyncCopyChainOps createAsyncCopy(triton::LoadOp loadOp, Value alloc,
                                  Value extractIdx, int contiguity);
StreamCopyChainOps createStreamCopy(triton::LoadOp loadOp, Value alloc,
                                    Value extractIdx);

} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_STREAMOPS_H_
