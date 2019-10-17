#pragma once

#ifndef _TRITON_SELECTION_SELECTION_H_
#define _TRITON_SELECTION_SELECTION_H_

#include <map>

namespace llvm{
  class Module;
  class Value;
}


namespace triton{

namespace ir{
class value;
class module;
}

namespace codegen{
// typedef
typedef llvm::Module Module;
typedef llvm::Value Value;
// forward
namespace analysis{
class liveness;
class align;
class allocation;
class axes;
class layout;
}
class target;
class tile;

}
}

namespace triton{
namespace codegen{

// Selection pass
class selection{
  typedef std::map<ir::value *, Value *> vmap_t;
  typedef std::map<ir::value *, tile *> tmap_t;

public:
  selection(analysis::liveness* liveness, analysis::allocation *alloc,
            analysis::align *alignment, analysis::axes *axes,
            analysis::layout *layouts, target *tgt, unsigned num_warps)
    : liveness_(liveness), alloc_(alloc),
      alignment_(alignment), a_axes_(axes), layouts_(layouts),
      tgt_(tgt), num_warps_(num_warps){ }

  void run(ir::module &src, Module &dst);

private:
  analysis::liveness *liveness_;
  analysis::allocation *alloc_;
  analysis::axes *a_axes_;
  analysis::layout *layouts_;
  analysis::align *alignment_;
  target *tgt_;
  unsigned num_warps_;
};

}
}

#endif
