#ifndef TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H
#define TDL_INCLUDE_IR_CODEGEN_STORAGE_ALLOC_H

#include <map>
#include <set>

namespace tdl{

namespace ir{
  class value;
  class function;
}

namespace codegen{

class layout;
class target_tuner;
class liveness;
class loop_info;

class allocation {
public:
  allocation(liveness *live)
    : liveness_(live){ }

  // accessors
  unsigned get_offset(ir::value *x)    const { return offsets_.at(x); }
  unsigned get_allocated_size()        const { return allocated_size_; }

  // run
  void run();

private:
  std::map<ir::value*, unsigned> offsets_;
  std::map<ir::value*, unsigned> num_bytes_;
  size_t allocated_size_;
  // dependences
  liveness *liveness_;
};

}
}

#endif
