#ifndef TRITON_INCLUDE_TRITON_CODEGEN_TRANSFORM_PREFETCH_H
#define TRITON_INCLUDE_TRITON_CODEGEN_TRANSFORM_PREFETCH_H

// forward dclaration
namespace triton::ir{
class module;
}

namespace triton::codegen {
class target;
}

namespace triton::codegen::transform {
class prefetch {
  target* tgt_;
public:
  prefetch(target *tgt) : tgt_(tgt) {}
  void run(ir::module &module);
};
}

#endif