#ifndef TDL_INCLUDE_CODEGEN_AXIS_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_AXIS_INFO_PASS_H

#include <set>
#include <map>

namespace triton {

namespace ir {
  class value;
  class module;
}

namespace codegen{

class axis_info {
private:
  // helpers
  bool is_first_axis_unit(ir::value *x);

  // populate maps
  bool populate_is_constant(ir::value *i);
  unsigned populate_max_contiguous(ir::value *i);
  unsigned populate_multiple_of(ir::value *i);

public:
  void run(ir::module &mod);

private:
  std::map<ir::value*, bool> is_constant_;
  std::map<ir::value*, unsigned> max_contiguous_;
  std::map<ir::value*, unsigned> multiple_of_;
};


}
}

#endif
