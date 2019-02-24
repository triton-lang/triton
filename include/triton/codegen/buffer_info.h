#ifndef TDL_INCLUDE_CODEGEN_BUFFER_INFO_PASS_H
#define TDL_INCLUDE_CODEGEN_BUFFER_INFO_PASS_H

#include <set>
#include <map>

namespace triton {

namespace ir {
  class module;
  class value;
  class phi_node;
}

namespace codegen{

class buffer_info_pass {
public:
  void run(ir::module &mod);
  // queries
  bool is_double(ir::value *x);
  bool is_shared(ir::value *x);
  bool is_loop_latch(ir::phi_node *phi, ir::value *terminator);
  ir::value *get_reference(ir::value *x);


private:
  std::set<ir::value*> shared_;
  std::set<ir::value*> double_;
  std::map<ir::value*, ir::value*> refs_;
};


}
}

#endif
