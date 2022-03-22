#pragma once

#ifndef _TRITON_IR_MODULE_H_
#define _TRITON_IR_MODULE_H_

#include <map>
#include <set>
#include <stack>
#include <string>
#include <functional>
#include "triton/ir/builder.h"
#include "triton/ir/metadata.h"
#include "triton/ir/context.h"

namespace triton{

namespace lang{

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
class alloc_const;

/* Module */

class module {
  typedef std::pair<std::string, basic_block*> val_key_t;
  friend class function;
  typedef std::pair<ir::metadata::kind_t, unsigned> md_pair_t;

public:
  typedef std::map<std::string, global_value*> symbols_map_t;
  typedef std::vector<function*> functions_list_t;
  struct current_iteration_info_t{
    lang::iteration_statement *statement;
    basic_block *block;
  };

private:
  phi_node *make_phi(type *ty, unsigned num_values, basic_block *block);
  value *try_remove_trivial_phis(ir::phi_node *&phi);
  value *add_phi_operands(const std::string& name, phi_node *&phi);
  value *get_value_recursive(const std::string& name, basic_block *block);
  void push_function(function *fn) { functions_.push_back(fn); }

public:
  module(const std::string &name, builder &builder): name_(name), builder_(builder) {}
  builder &get_builder() { return builder_; };
  const std::string& get_name() { return name_; };

  // Functions
  const functions_list_t &get_function_list() const { return functions_; }
  functions_list_t &get_function_list()             { return functions_; }
  function *get_or_insert_function(const std::string &name, function_type *ty);
  // Const allocation
  void add_alloc(ir::alloc_const* x)                          { allocs_.push_back(x); }
  const std::vector<ir::alloc_const*>& allocs()               { return allocs_; }
  // Register global
  void register_global(const std::string& name, ir::value *x) { globals_[name] = x; }
  const std::map<std::string, ir::value*>& globals() const    { return globals_; }
  // Metadata
  void add_metadata(const std::string &name, md_pair_t x)     { metadatas_[name] = x; }
  const std::map<std::string, md_pair_t> &get_metadatas() const { return metadatas_; }
  void print(std::ostream &os);

private:
  std::string name_;
  builder &builder_;
  functions_list_t functions_;
  symbols_map_t symbols_;
  std::vector<ir::alloc_const*> allocs_;
  std::map<std::string, ir::value*> globals_;
  std::map<std::string, md_pair_t> metadatas_;
};

}
}

#endif
