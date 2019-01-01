#ifndef TDL_INCLUDE_IR_FUNCTION_H
#define TDL_INCLUDE_IR_FUNCTION_H

#include <string>
#include "value.h"

namespace tdl{
namespace ir{

class function_type;
class module;

/* Argument */
class argument: public value{

};

/* Function */
class function: public value{
  using arg_iterator = argument *;
  using const_arg_iterator = const argument *;

public:
  arg_iterator arg_begin();
  arg_iterator arg_end();
  const_arg_iterator arg_begin() const;
  const_arg_iterator arg_end() const;
  // Factory methods
  static function *create(function_type *type, const std::string &name, module *mod);

private:
  function_type *type_;
  std::string name_;
  module *mod_;
};

}
}

#endif
