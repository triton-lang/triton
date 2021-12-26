#pragma once

#include <map>
#include <vector>

namespace triton {

namespace ir {
  class module;
  class function;
  class call_inst;
  class builder;
}

namespace codegen{
namespace transform{

class inliner {
public:
  inliner() {}
  void do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder,
                 std::map<ir::function*, std::vector<ir::call_inst*>>& callsites);
  void run(ir::module &mod);
};


}
}
}
