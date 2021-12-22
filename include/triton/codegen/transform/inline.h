#pragma once


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
  void do_inline(ir::function* fn, ir::call_inst* callsite, ir::builder& builder);
  void run(ir::module &mod);
};


}
}
}
