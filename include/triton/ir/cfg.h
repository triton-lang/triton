#pragma once

#ifndef _TRITON_IR_CFG_H_
#define _TRITON_IR_CFG_H_

#include <vector>

namespace triton{
namespace ir{

class function;
class basic_block;

class cfg {
public:
  static std::vector<basic_block *> reverse_post_order(function* fn);
};

}
}

#endif
