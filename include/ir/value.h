#ifndef TDL_INCLUDE_IR_VALUE_H
#define TDL_INCLUDE_IR_VALUE_H

#include <string>
#include <vector>
#include <memory>

namespace tdl{
namespace ir{

class type;
class use;

//===----------------------------------------------------------------------===//
//                               value class
//===----------------------------------------------------------------------===//

class value {
public:
  // constructor
  value(type *ty, const std::string &name = "");
  virtual ~value(){ }
  // uses
  void add_use(use *arg);
  // name
  void set_name(const std::string &name);
  const std::string &get_name() const { return name_; }
  type* get_type() const { return ty_; }

private:
  type *ty_;
  std::string name_;
};

//===----------------------------------------------------------------------===//
//                               use class
//===----------------------------------------------------------------------===//

class use {
public:
  // Implicit conversions to/from value
  friend class value;
  operator value *() const { return val_; }
  value *get() const { return val_; }
  value *operator->() { return val_; }
  const value *operator->() const { return val_; }
  inline void set(value *val);
  inline value *operator=(value *rhs);
  inline const use &operator=(const use &rhs);

private:
  value *val_;
};

//===----------------------------------------------------------------------===//
//                               user class
//===----------------------------------------------------------------------===//

class user: public value{
protected:
  void resize_ops(unsigned n) { ops_.resize(n); }

public:
  // Constructor
  user(type *ty, unsigned num_ops, const std::string &name = "")
      : value(ty, name), ops_(num_ops){ }

  // Operands
  void set_operand(unsigned i, value *x);
  value *get_operand(unsigned i);
  unsigned get_num_operands();

private:
  std::vector<use> ops_;
};

}
}

#endif
