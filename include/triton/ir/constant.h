#ifndef TDL_INCLUDE_IR_CONSTANT_H
#define TDL_INCLUDE_IR_CONSTANT_H

#include "enums.h"
#include "value.h"
#include <cassert>

namespace triton{
namespace ir{

class type;
class context;

/* Constant */
class constant: public user{
protected:
  using user::user;

public:
  static constant* get_all_ones_value(type *ty);
  static constant* get_null_value(type *ty);
};

/* Undef value */
class undef_value: public constant{
private:
  undef_value(type *ty);

public:
  static undef_value* get(type* ty);
};


/* Constant int */
class constant_int: public constant{
protected:
  constant_int(type *ty, uint64_t value);

public:
  virtual uint64_t get_value() const { return value_; }
  virtual std::string repr() const { return std::to_string(get_value()); }
  static constant_int *get(type *ty, uint64_t value);

protected:
  uint64_t value_;
};

/* Metaparameter (int) */
class metaparameter: public constant_int {
private:
  metaparameter(type *ty, const std::vector<unsigned>& space);

public:
  static metaparameter *create(context &ctx, type *ty, unsigned lo, unsigned hi);
  static metaparameter *create(context &ctx, type *ty, const std::vector<unsigned>& space);
  void set_value(uint64_t value) { has_value_ = true; value_ = value; }
  bool has_value() { return has_value_; }
  const std::vector<unsigned>& get_space() { return space_; }
  void set_space(const std::vector<unsigned> &space) { space_ = space; }
  uint64_t get_value() const { assert(has_value_); return value_; }
  std::string repr() const { return has_value_? std::to_string(value_) : "?" ;}
private:
  std::vector<unsigned> space_;
  bool has_value_;
};

class constant_expression: public constant_int {
  typedef binary_op_t op_t;

private:
  constant_expression(op_t op, constant_int* lhs, constant_int* rhs);

public:
  uint64_t get_value() const;
  // Wraps
  void set_has_no_unsigned_wrap(bool b = true) { has_no_unsigned_wrap_ = b; }
  void set_has_no_signed_wrap(bool b = true)   { has_no_signed_wrap_ = b; }
  // Factory
  static constant_expression *create(op_t op, constant_int* lhs, constant_int* rhs);

private:
  op_t op_;
  constant_int* lhs_;
  constant_int* rhs_;
  bool has_no_unsigned_wrap_;
  bool has_no_signed_wrap_;
};

/* constant range */
class constant_range: public constant{
  constant_range(type *ty, constant_int* first, constant_int* last);

public:
  static constant *get(constant_int *first, constant_int *last);
  const constant_int* get_first() const;
  const constant_int* get_last() const;

private:
  constant_int* first_;
  constant_int* last_;
};

/* constant fp */
class constant_fp: public constant{
  constant_fp(context &ctx, double value);

public:
  double get_value() { return value_; }
  static constant* get_negative_zero(type *ty);
  static constant* get_zero_value_for_negation(type *ty);
  static constant *get(context &ctx, double v);

private:
  double value_;
};

/* global value */
class global_value: public constant {
public:
  enum linkage_types_t {
    external
  };

public:
  global_value(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space);

private:
  linkage_types_t linkage_;
};

/* global object */
class global_object: public global_value {
public:
  global_object(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space = 0);
};

/* global variable */
class alloc_const: public global_object {
public:
  alloc_const(type *ty, constant_int *size,
              const std::string &name = "");
};

}
}

#endif
