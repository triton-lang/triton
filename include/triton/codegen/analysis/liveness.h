#ifndef TDL_INCLUDE_IR_CODEGEN_LIVENESS_H
#define TDL_INCLUDE_IR_CODEGEN_LIVENESS_H

#include <map>
#include <set>
#include <vector>
#include "triton/codegen/analysis/layout.h"
#include "triton/tools/graph.h"

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
class layout;
class layout_t;

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
  typedef std::map<layout_t*, segment>    intervals_map_t;

public:
  // constructor
  liveness(layout *l): layouts_(l){ }
  // accessors
  const intervals_map_t& get()  const { return intervals_; }
  segment get(layout_t* v)  const { return intervals_.at(v); }
  // run
  void run(ir::module &mod);

private:
  // analysis
  layout *layouts_;
  intervals_map_t intervals_;
};

}
}
}


#endif
