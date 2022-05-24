#ifndef TDL_INCLUDE_IR_CODEGEN_LIVENESS_H
#define TDL_INCLUDE_IR_CODEGEN_LIVENESS_H

#include "triton/codegen/analysis/layout.h"
#include "triton/tools/graph.h"

#include "llvm/ADT/MapVector.h"

#include <set>
#include <vector>

namespace triton{

namespace ir{
  class value;
  class phi_node;
  class function;
  class module;
  class instruction;
}

namespace codegen{
namespace analysis{

typedef unsigned slot_index;

class tiles;
class layouts;
class data_layout;

struct segment {
  slot_index start;
  slot_index end;

  bool contains(slot_index idx) const {
    return start <= idx && idx < end;
  }

  bool intersect(const segment &Other){
    return contains(Other.start) || Other.contains(start);
  }
};


class liveness {
private:
  typedef llvm::MapVector<shared_layout*, segment>    intervals_map_t;

public:
  // constructor
  liveness(layouts *l): layouts_(l){ }
  // accessors
  const intervals_map_t& get()  const { return intervals_; }
  segment get(shared_layout* v)  const { return intervals_.lookup(v); }
  // run
  void run(ir::module &mod);

private:
  // analysis
  layouts *layouts_;
  intervals_map_t intervals_;
};

}
}
}


#endif
