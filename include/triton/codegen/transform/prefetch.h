#ifndef TRITON_INCLUDE_TRITON_CODEGEN_TRANSFORM_PREFETCH_H
#define TRITON_INCLUDE_TRITON_CODEGEN_TRANSFORM_PREFETCH_H

#include <set>

// forward dclaration
namespace triton::ir{
class module;
class value;
}

namespace triton::codegen {
class target;
}

namespace triton::codegen::transform {
class prefetch {
  target* tgt_;
  std::set<ir::value*> prefetched_vals_;
public:
  prefetch(target *tgt) : tgt_(tgt) {}
  void run(ir::module &module);
  bool is_prefetched(ir::value* v) { return prefetched_vals_.find(v) != prefetched_vals_.end(); }
};
}

#endif