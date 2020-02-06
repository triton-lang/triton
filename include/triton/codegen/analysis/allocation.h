#ifndef TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H
#define TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H

#include <map>
#include <set>
#include <iostream>
#include "triton/codegen/analysis/liveness.h"

namespace triton{

namespace ir{
  class value;
  class function;
  class module;
}

namespace codegen{
namespace analysis{

class tiles;

class liveness;
class cts;

class allocation {
public:
  allocation(liveness *live)
    : liveness_(live) { }
  // accessors
  bool has_offset(const data_layout *x)    const { return offsets_.find(x) != offsets_.end(); }
  unsigned offset(const data_layout *x)    const { return offsets_.at(x); }
  unsigned allocated_size()        const { return allocated_size_; }
  // run
  void run(ir::module& mod);

private:
  std::map<const data_layout*, unsigned> offsets_;
  size_t allocated_size_;
  // dependences
  liveness *liveness_;
};

}
}
}

#endif
