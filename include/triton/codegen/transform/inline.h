#pragma once


namespace triton {

namespace ir {
  class module;
}

namespace codegen{
namespace transform{

class inliner {
public:
  inliner() {}
  void run(ir::module &mod);
};


}
}
}
