#ifndef TDL_INCLUDE_CODEGEN_SHARED_COPY_H
#define TDL_INCLUDE_CODEGEN_SHARED_COPY_H

namespace tdl {

namespace ir {
  class module;
  class value;
  class builder;
}

namespace codegen{

class place_shared_copy {
private:
  void add(ir::value *x, ir::builder &builder);

public:
  void run(ir::module &mod);
};


}
}

#endif
