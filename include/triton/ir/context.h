#ifndef TDL_INCLUDE_IR_CONTEXT_H
#define TDL_INCLUDE_IR_CONTEXT_H

#include <memory>
#include "triton/ir/type.h"

namespace tdl{
namespace ir{

class type;
class context_impl;

/* Context */
class context {
public:
  context();

public:
  std::shared_ptr<context_impl> p_impl;
};

}
}

#endif
