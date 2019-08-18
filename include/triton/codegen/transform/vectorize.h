#ifndef TDL_INCLUDE_CODEGEN_VECTORIZE_H
#define TDL_INCLUDE_CODEGEN_VECTORIZE_H

namespace triton {

namespace ir {
  class module;
}

namespace codegen{

namespace analysis{
  class grids;
}

namespace transform{

class vectorize {
public:
  vectorize(analysis::grids *params): params_(params){}
  void run(ir::module &mod);

private:
  analysis::grids *params_;
};

}
}
}

#endif
