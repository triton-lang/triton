#pragma once

#ifndef _TRITON_IR_CONSTANT_H_
#define _TRITON_IR_CONSTANT_H_

#include "enums.h"
#include "value.h"
#include <cassert>
#include "visitor.h"

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
  virtual std::string repr() const = 0;
};

/* Undef value */
class undef_value: public constant{
private:
  undef_value(type *ty);

public:
  static undef_value* get(type* ty);
  std::string repr() const { return "undef"; }
  void accept(visitor* vst) { vst->visit_undef_value(this); }
};


/* Constant int */
class constant_int: public constant{
protected:
  constant_int(type *ty, uint64_t value);

public:
  virtual uint64_t get_value() const { return value_; }
  static constant_int *get(type *ty, uint64_t value);
  std::string repr() const { return std::to_string(value_); }
  void accept(visitor* vst) { vst->visit_constant_int(this); }

protected:
  uint64_t value_;
};

/* Constant fp */
class constant_fp: public constant{
  constant_fp(type *ty, double value);

public:
  double get_value() { return value_; }
  static constant* get_negative_zero(type *ty);
  static constant* get_zero_value_for_negation(type *ty);
  static constant* get(context &ctx, double v);
  static constant* get(type *ty, double v);
  std::string repr() const { return std::to_string(value_); }
  void accept(visitor* vst) { vst->visit_constant_fp(this); }

private:
  double value_;
};


/* Global Value */
class global_value: public constant {
public:
  enum linkage_types_t {
    external
  };

public:
  global_value(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space);
  std::string repr() const { return get_name(); }

private:
  linkage_types_t linkage_;
};

/* global object */
class global_object: public global_value {
public:
  global_object(type *ty, unsigned num_ops,
               linkage_types_t linkage, const std::string &name,
               unsigned addr_space = 0);
  std::string repr() const { return get_name(); }
};

/* global variable */
class alloc_const: public global_object {
public:
  alloc_const(type *ty, constant_int *size,
              const std::string &name = "");
  std::string repr() const { return get_name(); }
  void accept(visitor* vst) { vst->visit_alloc_const(this); }


};

}
}

#endif
