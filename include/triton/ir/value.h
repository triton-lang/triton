#pragma once

#ifndef _TRITON_IR_VALUE_H_
#define _TRITON_IR_VALUE_H_

#include <string>
#include <vector>
#include <set>

namespace triton{
namespace ir{

class type;
class use;
class user;
class visitor;

//===----------------------------------------------------------------------===//
//                               value class
//===----------------------------------------------------------------------===//

class value {
public:
  typedef std::vector<user*> users_t;

public:
  // constructor
  value(type *ty, const std::string &name = "");
  virtual ~value(){ }
  // uses
  void add_use(user* arg);
  users_t::iterator erase_use(user* arg);
  const std::vector<user*> &get_users() { return users_; }
  void replace_all_uses_with(value *target);
  // name
  void set_name(const std::string &name);
  const std::string &get_name() const { return name_; }
  bool has_name() const { return !name_.empty(); }
  type* get_type() const { return ty_; }
  // visitor
  virtual void accept(visitor *v) = 0;

private:
  std::string name_;

protected:
  type *ty_;
  users_t users_;
};

//===----------------------------------------------------------------------===//
//                               user class
//===----------------------------------------------------------------------===//

class user: public value{
public:
  typedef std::vector<value*>      ops_t;
  typedef ops_t::iterator       op_iterator;
  typedef ops_t::const_iterator const_op_iterator;

protected:
  void resize_ops(unsigned num_ops) { ops_.resize(num_ops + num_hidden_); num_ops_ = num_ops; }
  void resize_hidden(unsigned num_hidden) { ops_.resize(num_ops_ + num_hidden); num_hidden_ = num_hidden; }

public:
  // Constructor
  user(type *ty, unsigned num_ops, const std::string &name = "")
      : value(ty, name), ops_(num_ops), num_ops_(num_ops), num_hidden_(0){
  }
  virtual ~user() { }

  // Operands
  const ops_t& ops() { return ops_; }
  const ops_t& ops() const { return ops_; }
  op_iterator op_begin() { return ops_.begin(); }
  op_iterator op_end()   { return ops_.end(); }
  void     set_operand(unsigned i, value *x);
  value   *get_operand(unsigned i) const;
  unsigned get_num_operands() const ;
  unsigned get_num_hidden() const;

  // Utils
  value::users_t::iterator replace_uses_of_with(value *before, value *after);


private:
  ops_t ops_;
  unsigned num_ops_;
  unsigned num_hidden_;
};

}
}

#endif
