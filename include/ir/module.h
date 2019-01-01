#ifndef TDL_INCLUDE_IR_MODULE_H
#define TDL_INCLUDE_IR_MODULE_H

#include <map>
#include <set>
#include <string>
#include "builder.h"

namespace tdl{
namespace ir{

class basic_block;
class phi_node;
class value;
class context;

/* Module */
class module {
  typedef std::pair<std::string, basic_block*> val_key_t;
  phi_node *make_phi(type *ty, unsigned num_values, basic_block *block);
  void add_phi_operands(const std::string& name, phi_node *&phi);
  value *get_value_recursive(const std::string& name, basic_block *block);

public:
  module(const std::string &name, context *ctx);
  context& get_context();
  builder& get_builder();
  // Setters
  void set_value(const std::string& name, basic_block* block, value *x);
  void set_value(const std::string& name, value* x);
  // Getters
  value *get_value(const std::string& name, basic_block* block);
  value *get_value(const std::string& name);
  // Seal block -- no more predecessors will be added
  void seal_block(basic_block *block);

private:
  builder builder_;
  std::map<val_key_t, value*> values_;
  std::set<basic_block*> sealed_blocks_;
  std::map<basic_block*, std::map<std::string, phi_node*>> incomplete_phis_;
};

}
}

#endif
