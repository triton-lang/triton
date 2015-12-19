#ifndef _ISAAC_SYMBOLIC_EXPRESSION_H
#define _ISAAC_SYMBOLIC_EXPRESSION_H

#include <vector>
#include <list>
#include "isaac/driver/backend.h"
#include "isaac/driver/context.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/event.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/driver/buffer.h"


#include "isaac/types.h"
#include "isaac/value_scalar.h"
#include <memory>
#include <iostream>

namespace isaac
{

class array_base;

/** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
enum operation_type_family
{
  INVALID_TYPE_FAMILY = 0,

  // BLAS1-type
  UNARY_TYPE_FAMILY,
  BINARY_TYPE_FAMILY,
  VECTOR_DOT_TYPE_FAMILY,

  // BLAS2-type
  ROWS_DOT_TYPE_FAMILY,
  COLUMNS_DOT_TYPE_FAMILY,

  // BLAS3-type
  MATRIX_PRODUCT_TYPE_FAMILY
};

/** @brief Enumeration for identifying the possible operations */
enum operation_type
{
  INVALID_TYPE = 0,

  // unary operator
  MINUS_TYPE,
  NEGATE_TYPE,

  // unary expression
  CAST_BOOL_TYPE,
  CAST_CHAR_TYPE,
  CAST_UCHAR_TYPE,
  CAST_SHORT_TYPE,
  CAST_USHORT_TYPE,
  CAST_INT_TYPE,
  CAST_UINT_TYPE,
  CAST_LONG_TYPE,
  CAST_ULONG_TYPE,
  CAST_HALF_TYPE,
  CAST_FLOAT_TYPE,
  CAST_DOUBLE_TYPE,

  ABS_TYPE,
  ACOS_TYPE,
  ASIN_TYPE,
  ATAN_TYPE,
  CEIL_TYPE,
  COS_TYPE,
  COSH_TYPE,
  EXP_TYPE,
  FABS_TYPE,
  FLOOR_TYPE,
  LOG_TYPE,
  LOG10_TYPE,
  SIN_TYPE,
  SINH_TYPE,
  SQRT_TYPE,
  TAN_TYPE,
  TANH_TYPE,
  TRANS_TYPE,

  // binary expression
  ASSIGN_TYPE,
  INPLACE_ADD_TYPE,
  INPLACE_SUB_TYPE,
  ADD_TYPE,
  SUB_TYPE,
  MULT_TYPE,
  DIV_TYPE,
  ELEMENT_ARGFMAX_TYPE,
  ELEMENT_ARGFMIN_TYPE,
  ELEMENT_ARGMAX_TYPE,
  ELEMENT_ARGMIN_TYPE,
  ELEMENT_PROD_TYPE,
  ELEMENT_DIV_TYPE,
  ELEMENT_EQ_TYPE,
  ELEMENT_NEQ_TYPE,
  ELEMENT_GREATER_TYPE,
  ELEMENT_GEQ_TYPE,
  ELEMENT_LESS_TYPE,
  ELEMENT_LEQ_TYPE,
  ELEMENT_POW_TYPE,
  ELEMENT_FMAX_TYPE,
  ELEMENT_FMIN_TYPE,
  ELEMENT_MAX_TYPE,
  ELEMENT_MIN_TYPE,

  //Products
  OUTER_PROD_TYPE,
  MATRIX_PRODUCT_NN_TYPE,
  MATRIX_PRODUCT_TN_TYPE,
  MATRIX_PRODUCT_NT_TYPE,
  MATRIX_PRODUCT_TT_TYPE,

  //Access modifiers
  MATRIX_DIAG_TYPE,
  MATRIX_ROW_TYPE,
  MATRIX_COLUMN_TYPE,
  REPEAT_TYPE,
  RESHAPE_TYPE,
  SHIFT_TYPE,
  VDIAG_TYPE,
  ACCESS_INDEX_TYPE,


  PAIR_TYPE,

  OPERATOR_FUSE,
  SFOR_TYPE,
};

struct op_element
{
  op_element();
  op_element(operation_type_family const & _type_family, operation_type const & _type);
  operation_type_family   type_family;
  operation_type          type;
};

struct for_idx_t
{
  math_expression operator=(value_scalar const & ) const;
  math_expression operator=(math_expression const & ) const;

  math_expression operator+=(value_scalar const & ) const;
  math_expression operator-=(value_scalar const & ) const;
  math_expression operator*=(value_scalar const & ) const;
  math_expression operator/=(value_scalar const & ) const;

  int level;
};

enum node_type
{
  INVALID_SUBTYPE = 0,
  COMPOSITE_OPERATOR_TYPE,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE,
  FOR_LOOP_INDEX_TYPE
};

struct tree_node
{
  tree_node();
  node_type subtype;
  numeric_type dtype;
  union
  {
    std::size_t   node_index;
    values_holder vscalar;
    isaac::array_base* array;
    for_idx_t for_idx;
  };
};

struct invalid_node{};

void fill(tree_node &x, for_idx_t index);
void fill(tree_node &x, invalid_node);
void fill(tree_node & x, std::size_t node_index);
void fill(tree_node & x, array_base const & a);
void fill(tree_node & x, value_scalar const & v);

class math_expression
{
public:
  struct node
  {
    tree_node    lhs;
    op_element   op;
    tree_node    rhs;
  };

  typedef std::vector<node>     container_type;

public:
  math_expression(value_scalar const &lhs, for_idx_t const &rhs, const op_element &op, const numeric_type &dtype);
  math_expression(for_idx_t const &lhs, for_idx_t const &rhs, const op_element &op);
  math_expression(for_idx_t const &lhs, value_scalar const &rhs, const op_element &op, const numeric_type &dtype);

  template<class LT, class RT>
  math_expression(LT const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape);
  template<class RT>
  math_expression(math_expression const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape);
  template<class LT>
  math_expression(LT const & lhs, math_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape);
  math_expression(math_expression const & lhs, math_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, shape_t const & shape);

  shape_t shape() const;
  math_expression& reshape(int_t size1, int_t size2=1);
  int_t dim() const;
  container_type & tree();
  container_type const & tree() const;
  std::size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;

  math_expression operator-();
  math_expression operator!();
private:
  container_type tree_;
  std::size_t root_;
  driver::Context const * context_;
  numeric_type dtype_;
  shape_t shape_;
};



struct execution_options_type
{
  execution_options_type(unsigned int _queue_id = 0, std::list<driver::Event>* _events = NULL, std::vector<driver::Event>* _dependencies = NULL) :
     events(_events), dependencies(_dependencies), queue_id_(_queue_id)
  {}

  execution_options_type(driver::CommandQueue const & queue, std::list<driver::Event> *_events = NULL, std::vector<driver::Event> *_dependencies = NULL) :
      events(_events), dependencies(_dependencies), queue_id_(-1), queue_(new driver::CommandQueue(queue))
  {}

  void enqueue(driver::Context const & context, driver::Kernel const & kernel, driver::NDRange global, driver::NDRange local) const
  {
    driver::Event event = queue(context).enqueue(kernel, global, local, dependencies);
    if(events)
      events->push_back(event);
  }

  driver::CommandQueue & queue(driver::Context const & context) const
  {
    if(queue_)
        return *queue_;
    return driver::backend::queues::get(context, queue_id_);
  }

  std::list<driver::Event>* events;
  std::vector<driver::Event>* dependencies;

private:
  int queue_id_;
  std::shared_ptr<driver::CommandQueue> queue_;
};

struct dispatcher_options_type
{
  dispatcher_options_type(bool _tune = false, int _label = -1) : tune(_tune), label(_label){}
  bool tune;
  int label;
};

struct compilation_options_type
{
  compilation_options_type(std::string const & _program_name = "", bool _recompile = false) : program_name(_program_name), recompile(_recompile){}
  std::string program_name;
  bool recompile;
};

class execution_handler
{
public:
  execution_handler(math_expression const & x, execution_options_type const& execution_options = execution_options_type(),
             dispatcher_options_type const & dispatcher_options = dispatcher_options_type(),
             compilation_options_type const & compilation_options = compilation_options_type())
                : x_(x), execution_options_(execution_options), dispatcher_options_(dispatcher_options), compilation_options_(compilation_options){}
  execution_handler(math_expression const & x, execution_handler const & other) : x_(x), execution_options_(other.execution_options_), dispatcher_options_(other.dispatcher_options_), compilation_options_(other.compilation_options_){}
  math_expression const & x() const { return x_; }
  execution_options_type const & execution_options() const { return execution_options_; }
  dispatcher_options_type const & dispatcher_options() const { return dispatcher_options_; }
  compilation_options_type const & compilation_options() const { return compilation_options_; }
private:
  math_expression x_;
  execution_options_type execution_options_;
  dispatcher_options_type dispatcher_options_;
  compilation_options_type compilation_options_;
};

math_expression::node const & lhs_most(math_expression::container_type const & array_base, math_expression::node const & init);
math_expression::node const & lhs_most(math_expression::container_type const & array_base, size_t root);


}

#endif
