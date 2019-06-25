#ifndef TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H

#include <set>
#include <map>

namespace triton {

namespace ir {
  class value;
  class module;
}

namespace codegen{

class alignment_info {
private:
  // helpers
  bool is_first_axis_unit(ir::value *v);

  // populate maps
  bool populate_is_constant(ir::value *v);
  unsigned populate_max_contiguous(ir::value *v);
  unsigned populate_starting_multiple(ir::value *v);

public:
  void run(ir::module &mod);
  unsigned get_starting_multiple(ir::value* v) const;
  unsigned get_max_contiguous(ir::value* v) const;

private:
  std::map<ir::value*, bool> is_constant_;
  std::map<ir::value*, unsigned> max_contiguous_;
  std::map<ir::value*, unsigned> starting_multiple_;
};


}
}

#endif
