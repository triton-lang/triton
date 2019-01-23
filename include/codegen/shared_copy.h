#ifndef TDL_INCLUDE_CODEGEN_SHARED_COPY_H
#define TDL_INCLUDE_CODEGEN_SHARED_COPY_H

namespace tdl {

namespace ir {
  class module;
}

namespace codegen{

class place_shared_copy {
public:
  void run(ir::module &mod);
};


}
}

#endif
