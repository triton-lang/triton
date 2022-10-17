#pragma once

#ifndef _TRITON_IR_BASIC_BLOCK_H_
#define _TRITON_IR_BASIC_BLOCK_H_

#include <string>
#include <list>
#include "value.h"
#include "visitor.h"

namespace triton{
namespace ir{

class context;
class function;
class instruction;

/* Basic Block */
class basic_block: public value{
public:
  // instruction iterator types
  typedef std::list<instruction*>                inst_list_t;
  typedef inst_list_t::iterator                  iterator;
  typedef inst_list_t::const_iterator            const_iterator;
  typedef inst_list_t::reverse_iterator          reverse_iterator;
  typedef inst_list_t::const_reverse_iterator    const_reverse_iterator;

private:
  // constructors
  basic_block(context &ctx, const std::string &name, function *parent, basic_block *next);

public:
  // accessors
  function* get_parent() { return parent_; }
  context& get_context() { return ctx_; }

  // get iterator to first instruction that is not a phi
  void replace_phi_uses_with(basic_block* before, basic_block* after);
  iterator get_first_non_phi();

  // get instruction list
  inst_list_t           &get_inst_list()       { return inst_list_; }
  const inst_list_t     &get_inst_list() const { return inst_list_; }
  void  erase(instruction *i)                  {  inst_list_.remove(i); }

  // instruction iterator functions
  inline iterator                begin()       { return inst_list_.begin(); }
  inline const_iterator          begin() const { return inst_list_.begin(); }
  inline iterator                end  ()       { return inst_list_.end();   }
  inline const_iterator          end  () const { return inst_list_.end();   }

  inline reverse_iterator        rbegin()       { return inst_list_.rbegin(); }
  inline const_reverse_iterator  rbegin() const { return inst_list_.rbegin(); }
  inline reverse_iterator        rend  ()       { return inst_list_.rend();   }
  inline const_reverse_iterator  rend  () const { return inst_list_.rend();   }

  inline size_t                   size() const { return inst_list_.size();  }
  inline bool                    empty() const { return inst_list_.empty(); }
  inline const instruction      &front() const { return *inst_list_.front(); }
  inline       instruction      &front()       { return *inst_list_.front(); }
  inline const instruction       &back() const { return *inst_list_.back();  }
  inline       instruction       &back()       { return *inst_list_.back();  }

  void append_instruction(ir::instruction* i);
  // split
  basic_block* split_before(ir::instruction* loc, const std::string& name);

  // predecessors
  std::vector<basic_block*> get_predecessors() const;
  std::vector<basic_block*> get_successors() const;

  // factory functions
  static basic_block* create(context &ctx, const std::string &name, function *parent, basic_block *next = nullptr);

  void print(std::ostream &os);

  // visitor
  void accept(visitor *v) { v->visit_basic_block(this); }

private:
  context &ctx_;
  std::string name_;
  function *parent_;
  std::vector<basic_block*> preds_;
  std::vector<basic_block*> succs_;
  inst_list_t inst_list_;
};

}
}

#endif
