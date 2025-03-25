#ifndef GET_ILUVATAR_BLOKED_LAYOUT_BUILDER_PLUGIN_H
#define GET_ILUVATAR_BLOKED_LAYOUT_BUILDER_PLUGIN_H

#include "mlir/Support/LLVM.h"
#include "python/src/plugin.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using AttrBuilderFunc = mlir::triton::gpu::CTALayoutAttr (*)(
    unsigned, unsigned, mlir::Type, llvm::ArrayRef<int64_t>,
    llvm::ArrayRef<unsigned>, llvm::ArrayRef<unsigned>,
    llvm::ArrayRef<unsigned>, llvm::ArrayRef<unsigned>, unsigned,
    llvm::SmallVector<unsigned> &, mlir::MLIRContext *);
DEFINE_LOAD_FUNC(AttrBuilder)

#endif
