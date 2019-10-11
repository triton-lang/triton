#ifndef TDL_INCLUDE_IR_CODEGEN_LIVENESS_H
#define TDL_INCLUDE_IR_CODEGEN_LIVENESS_H

#include <map>
#include <set>
#include <vector>
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

struct double_buffer_info_t {
  ir::value* latch;
  ir::phi_node* phi;
};


class liveness {
private:
  typedef std::map<ir::value*, slot_index> indices_map_t;
  typedef std::map<layout_t*, segment>    intervals_map_t;
  typedef std::map<ir::value*, bool>       has_storage_map_t;
  typedef ir::value* node_t;
  typedef std::map <node_t, std::set<node_t>> graph_t;

public:
  // Intervals iterators
  using iterator = intervals_map_t::iterator;
  using const_iterator = intervals_map_t::const_iterator;




private:
  void extract_double_bufferable(ir::instruction *i);
  void extract_buffers(ir::instruction *i);
  void get_parents(ir::instruction *i, std::vector<ir::value *>& res);
  void make_graph(ir::instruction *i);
  bool do_pad(ir::value *x);


public:
  liveness(layout *l): layouts_(l){ }
  // padding
  unsigned get_pad(ir::value *v) const { return pad_.at(v); }
  // buffer size
  unsigned num_bytes(ir::value *x);
  // accessors
  const intervals_map_t& intervals()  const { return intervals_; }
  segment get_interval(layout_t* v)  const { return intervals_.at(v); }
  // double-buffering
  bool has_double(ir::value *x)       const { return double_.find(x) != double_.end(); }
  double_buffer_info_t get_double(ir::value *x) const { return double_.at(x); }
  // run
  void run(ir::module &mod);

private:
  // analysis
  layout *layouts_;
  // stuff
  has_storage_map_t has_dedicated_storage_;
  indices_map_t indices;
  intervals_map_t intervals_;
  std::map<ir::value*, double_buffer_info_t> double_;
  std::map<ir::value*, size_t> pad_;
};

}
}
}


#endif
