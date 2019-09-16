#ifndef TDL_INCLUDE_CODEGEN_VECTORIZE_H
#define TDL_INCLUDE_CODEGEN_VECTORIZE_H

namespace triton {

namespace ir {
  class module;
}

namespace codegen{

namespace analysis{
  class tiles;
}

namespace transform{

class vectorize {
public:
  vectorize(analysis::tiles *params): params_(params){}
  void run(ir::module &mod);

private:
  analysis::tiles *params_;
};

}
}
}

#endif
