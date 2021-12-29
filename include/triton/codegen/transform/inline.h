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

struct fncmp {
  bool operator()(ir::function* x, ir::function* y) const;
};

class inliner {
public:
  inliner() {}
  void do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder,
                 std::map<ir::function *, std::vector<ir::call_inst *>, triton::codegen::transform::fncmp> &callsites);
  void run(ir::module &mod);
};


}
}
}
