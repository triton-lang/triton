#ifndef TDL_INCLUDE_CODEGEN_VECTORIZE_H
#define TDL_INCLUDE_CODEGEN_VECTORIZE_H

namespace tdl {

namespace ir {
  class module;
}

namespace codegen{

class tune;

class vectorize {
public:
  vectorize(tune *params): params_(params){}
  void run(ir::module &mod);

private:
  tune *params_;
};


}
}

#endif
