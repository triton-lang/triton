#ifndef TDL_INCLUDE_IR_BASIC_BLOCK_H
#define TDL_INCLUDE_IR_BASIC_BLOCK_H

#include <string>
#include "value.h"

namespace tdl{
namespace ir{

class context;
class function;
class instruction;

/* Basic Block */
class basic_block: public value{
public:
  // Accessors
  function* get_parent();
  instruction* get_first_non_phi_or_dbg();
  // Iterators
  instruction* begin();
  instruction* end();
  // CFG
  const std::vector<basic_block*>& get_predecessors() const;
  void add_predecessor(basic_block* pred);
  // Factory functions
  static basic_block* create(context &ctx, const std::string &name, function *parent);

private:
  context &ctx_;
  std::string name_;
  function *parent_;
  std::vector<basic_block*> preds_;
};

}
}

#endif
