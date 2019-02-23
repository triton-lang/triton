#ifndef TDL_INCLUDE_IR_MODULE_H
#define TDL_INCLUDE_IR_MODULE_H

#include <map>
#include <set>
#include <stack>
#include <string>
#include <functional>
#include "builder.h"

namespace tdl{

namespace ast{

class iteration_statement;
class compound_statement;

}

namespace ir{

class basic_block;
class phi_node;
class value;
class context;
class function;
class attribute;
class function_type;
class constant;
class global_value;

/* Module */
class module {
  typedef std::pair<std::string, basic_block*> val_key_t;
  friend class function;

public:
  typedef std::map<std::string, global_value*> symbols_map_t;
  typedef std::vector<function*> functions_list_t;
  struct current_iteration_info_t{
    ast::iteration_statement *statement;
    basic_block *block;
  };

private:
  phi_node *make_phi(type *ty, unsigned num_values, basic_block *block);
  value *try_remove_trivial_phis(ir::phi_node *&phi, value **pre_user);
  value *add_phi_operands(const std::string& name, phi_node *&phi);
  value *get_value_recursive(const std::string& name, basic_block *block);
  void push_function(function *fn) { functions_.push_back(fn); }

public:
  module(const std::string &name, context &ctx);
  context& get_context();
  builder& get_builder();
  // Setters
  void set_value(const std::string& name, basic_block* block, value *x);
  void set_value(const std::string& name, value* x);
  void set_type(const std::string& name, basic_block* block, type* x);
  void set_type(const std::string& name, type* x);
  void set_continue_fn(std::function<ir::value*()> fn);
  // Getters
  value *get_value(const std::string& name, basic_block* block);
  value *get_value(const std::string& name);
  type *get_type(const std::string& name, basic_block* block);
  type *get_type(const std::string& name);
  std::function<ir::value*()> get_continue_fn();
  // Seal block -- no more predecessors will be added
  void seal_block(basic_block *block);
  // Functions
  const functions_list_t &get_function_list() const { return functions_; }
  functions_list_t &get_function_list()             { return functions_; }
  function *get_or_insert_function(const std::string &name, function_type *ty);
  // Scope
  void push_scope(const ast::compound_statement* scope)   { scopes_.push(scope); }
  void pop_scope()                                        { scopes_.pop(); }
  const ast::compound_statement* get_scope()              { return scopes_.top(); }


private:
  std::string name_;
  context &context_;
  builder builder_;
  std::map<val_key_t, value*> values_;
  std::map<val_key_t, type*> types_;
  std::set<basic_block*> sealed_blocks_;
  std::map<basic_block*, std::map<std::string, phi_node*>> incomplete_phis_;
  functions_list_t functions_;
  symbols_map_t symbols_;
  std::function<ir::value*()> continue_fn_;
  std::map<value*, value**> current_phi_;
  std::stack<const ast::compound_statement*> scopes_;
};

}
}

#endif
