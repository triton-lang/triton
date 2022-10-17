#pragma once

#ifndef _TRITON_IR_CONTEXT_H_
#define _TRITON_IR_CONTEXT_H_

#include <memory>
#include "triton/ir/type.h"

namespace triton{
namespace ir{

class type;
class context_impl;

/* Context */
class context {
public:
  context();
  context(const context&) = delete;
  context& operator=(const context&) = delete;

public:
  std::shared_ptr<context_impl> p_impl;
};

}
}

#endif
