#ifndef TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_ALIGNMENT_INFO_PASS_H

#include <map>
#include <vector>

namespace triton {

namespace ir {
  class value;
  class module;
}

namespace codegen{
namespace analysis{

class align {
  struct cst_info {
    unsigned num_cst;
    unsigned value;
  };

private:
  // helpers
  bool is_first_axis_unit(ir::value *v);

  // populate maps
  std::vector<cst_info> populate_is_constant(ir::value *v);
  unsigned populate_max_contiguous(ir::value *v);
  unsigned populate_starting_multiple(ir::value *v);

public:
  void run(ir::module &mod);
  unsigned get_starting_multiple(ir::value* v) const;
  unsigned get_max_contiguous(ir::value* v) const;
  void copy(ir::value *dst, ir::value *src);

private:
  std::map<ir::value*, std::vector<cst_info>> is_constant_;
  std::map<ir::value*, std::vector<unsigned>> max_contiguous_;
  std::map<ir::value*, std::vector<unsigned>> starting_multiple_;
};


}
}
}

#endif
