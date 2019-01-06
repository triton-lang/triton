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
class basic_block;

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

/* Attribute */
class attribute {

};

/* Function */
class function: public global_object{
  typedef std::vector<argument*> args_t;
  typedef args_t::iterator       arg_iterator;
  typedef args_t::const_iterator const_arg_iterator;

  typedef std::vector<basic_block*> blocks_t;
  typedef blocks_t::iterator        block_iterator;
  typedef blocks_t::const_iterator  const_block_iterator;

private:
  function(function_type *ty, linkage_types_t linkage,
           const std::string &name = "", module *parent = nullptr);

public:
  // accessors
  const args_t &args() { return args_; }
  function_type* get_fn_type() { return fn_ty_; }
  // factory methods
  static function *create(function_type *ty, linkage_types_t linkage,
                          const std::string &name, module *mod);
  // blocks
  const blocks_t &blocks() { return blocks_; }
  void insert_block(basic_block* block, basic_block *next = nullptr);

private:
  module *parent_;
  bool init_;
  function_type *fn_ty_;
  args_t args_;
  blocks_t blocks_;
};

}
}

#endif
