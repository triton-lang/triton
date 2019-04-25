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

class layout;
class target_tuner;
class shmem_liveness;
class shmem_info;

class shmem_allocation {
public:
  shmem_allocation(shmem_liveness *live, shmem_info *buffer_info)
    : liveness_(live), buffer_info_(buffer_info){ }

  // utilities
  unsigned get_num_bytes(ir::value *x);

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
  shmem_liveness *liveness_;
  shmem_info *buffer_info_;
};

}
}

#endif
