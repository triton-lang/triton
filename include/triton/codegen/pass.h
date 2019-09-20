#ifndef _TRITON_CODEGEN_PASS_H_
#define _TRITON_CODEGEN_PASS_H_

#include <list>

namespace triton{

namespace ir{
  class module;
}

namespace codegen{

class pass {
public:
  virtual void run(ir::module& m);
};


class pass_manager {
public:
  void add(pass* p);
  void run(ir::module& m);

private:
  std::list<pass*> passes;
};

}
}
