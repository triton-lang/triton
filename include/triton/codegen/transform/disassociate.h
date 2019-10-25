#ifndef _TRITON_SELECTION_TRANSFORM_DISASSOCIATE_H_
#define _TRITON_SELECTION_TRANSFORM_DISASSOCIATE_H_


namespace triton {
namespace ir {
  class module;
}

namespace codegen{
namespace transform{

class disassociate {
public:
  void run(ir::module &mod);
};

}
}
}

#endif
