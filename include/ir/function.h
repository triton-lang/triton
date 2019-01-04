#ifndef TDL_INCLUDE_IR_FUNCTION_H
#define TDL_INCLUDE_IR_FUNCTION_H

#include <string>
#include "value.h"
#include "constant.h"

namespace tdl{
namespace ir{

class function;
class function_type;
class module;

/* Argument */
class argument: public value{
  argument(type *ty, const std::string &name, function *parent, unsigned arg_no);

public:
  static argument* create(type *ty, const std::string &name,
                          function *parent = nullptr, unsigned arg_no = 0);

private:
  function *parent_;
  unsigned arg_no_;
};

/* Function */
class function: public global_object{
  typedef std::vector<argument*> args_t;
  typedef args_t::iterator       arg_iterator;
  typedef args_t::const_iterator const_arg_iterator;
private:
  function(function_type *ty, linkage_types_t linkage,
           const std::string &name = "", module *parent = nullptr);

public:
  arg_iterator arg_begin() { return args_.begin(); }
  arg_iterator arg_end() { return args_.end(); }
  const_arg_iterator arg_begin() const { return args_.begin(); }
  const_arg_iterator arg_end() const { return args_.end(); }
  // Factory methods
  static function *create(function_type *ty, linkage_types_t linkage,
                          const std::string &name, module *mod);

private:
  module *parent_;
  args_t args_;
  bool init_;
  function_type *fn_ty_;
};

}
}

#endif
