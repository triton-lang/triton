#ifndef TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H
#define TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H

#include <map>
#include <set>
#include <iostream>

namespace triton{

namespace ir{
  class value;
  class function;
}

namespace codegen{
namespace analysis{

class tiles;

class liveness;
class cts;

class allocation {
public:
  allocation(liveness *live, cts *buffer_info, tiles *params)
    : liveness_(live), buffer_info_(buffer_info), tiles_(params){ }
  // utilities
  unsigned num_bytes(ir::value *x);
  unsigned is_ld_padded(ir::value* x);
  // accessors
  unsigned offset(ir::value *x)    const { return offsets_.at(x); }
  unsigned allocated_size()        const { return allocated_size_; }
  // run
  void run();

private:
  std::map<ir::value*, unsigned> offsets_;
  std::map<ir::value*, unsigned> num_bytes_;
  size_t allocated_size_;
  // dependences
  liveness *liveness_;
  cts *buffer_info_;
  tiles *tiles_;
};

}
}
}

#endif
