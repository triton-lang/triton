#ifndef TDL_INCLUDE_IR_BASIC_BLOCK_H
#define TDL_INCLUDE_IR_BASIC_BLOCK_H

#include <string>
#include "value.h"

namespace tdl{
namespace ir{

class context;
class function;

/* Basic Block */
class basic_block: public value{
public:
  function* get_parent();
  // Factory functions
  static basic_block* create(context &ctx, const std::string &name, function *parent);

private:
  context &ctx_;
  std::string name_;
  function *parent_;
};

}
}

#endif
