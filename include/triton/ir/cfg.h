#ifndef TDL_INCLUDE_IR_CFG_H
#define TDL_INCLUDE_IR_CFG_H

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
