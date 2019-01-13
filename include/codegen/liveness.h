#ifndef TDL_INCLUDE_IR_CODEGEN_LIVENESS_H
#define TDL_INCLUDE_IR_CODEGEN_LIVENESS_H

#include <map>

namespace tdl{

namespace ir{
  class value;
  class function;
  class module;
}

namespace codegen{

typedef unsigned slot_index;

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
  typedef std::map<ir::value*, slot_index> indices_map_t;
  typedef std::map<ir::value*, segment>    intervals_map_t;
  typedef std::map<ir::value*, bool>       has_storage_map_t;

public:
  /// Intervals iterators...
  using iterator = intervals_map_t::iterator;
  using const_iterator = intervals_map_t::const_iterator;

public:

  // accessors
  const intervals_map_t& intervals() const { return intervals_; }
  segment get_interval(ir::value* v) const { return intervals_.at(v); }

  // run
  void run(ir::module &mod);

private:
  has_storage_map_t has_dedicated_storage_;
  indices_map_t indices_;
  intervals_map_t intervals_;
};

}
}

#endif
