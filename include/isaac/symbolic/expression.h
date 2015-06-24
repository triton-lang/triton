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
#include "isaac/tools/shared_ptr.hpp"
#include <iostream>

namespace isaac
{

class array;
class repeat_infos;

/** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
enum operation_node_type_family
{
  OPERATOR_INVALID_TYPE_FAMILY = 0,

  // BLAS1-type
  OPERATOR_UNARY_TYPE_FAMILY,
  OPERATOR_BINARY_TYPE_FAMILY,
  OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY,

  // BLAS2-type
  OPERATOR_ROWS_REDUCTION_TYPE_FAMILY,
  OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY,

  // BLAS3-type
  OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY
};

/** @brief Enumeration for identifying the possible operations */
enum operation_node_type
{
  OPERATOR_INVALID_TYPE = 0,

  // unary operator
  OPERATOR_MINUS_TYPE,
  OPERATOR_NEGATE_TYPE,

  // unary expression
  OPERATOR_CAST_BOOL_TYPE,
  OPERATOR_CAST_CHAR_TYPE,
  OPERATOR_CAST_UCHAR_TYPE,
  OPERATOR_CAST_SHORT_TYPE,
  OPERATOR_CAST_USHORT_TYPE,
  OPERATOR_CAST_INT_TYPE,
  OPERATOR_CAST_UINT_TYPE,
  OPERATOR_CAST_LONG_TYPE,
  OPERATOR_CAST_ULONG_TYPE,
  OPERATOR_CAST_HALF_TYPE,
  OPERATOR_CAST_FLOAT_TYPE,
  OPERATOR_CAST_DOUBLE_TYPE,

  OPERATOR_ABS_TYPE,
  OPERATOR_ACOS_TYPE,
  OPERATOR_ASIN_TYPE,
  OPERATOR_ATAN_TYPE,
  OPERATOR_CEIL_TYPE,
  OPERATOR_COS_TYPE,
  OPERATOR_COSH_TYPE,
  OPERATOR_EXP_TYPE,
  OPERATOR_FABS_TYPE,
  OPERATOR_FLOOR_TYPE,
  OPERATOR_LOG_TYPE,
  OPERATOR_LOG10_TYPE,
  OPERATOR_SIN_TYPE,
  OPERATOR_SINH_TYPE,
  OPERATOR_SQRT_TYPE,
  OPERATOR_TAN_TYPE,
  OPERATOR_TANH_TYPE,
  OPERATOR_TRANS_TYPE,

  // binary expression
  OPERATOR_ACCESS_TYPE,
  OPERATOR_ASSIGN_TYPE,
  OPERATOR_INPLACE_ADD_TYPE,
  OPERATOR_INPLACE_SUB_TYPE,
  OPERATOR_ADD_TYPE,
  OPERATOR_SUB_TYPE,
  OPERATOR_MULT_TYPE,
  OPERATOR_DIV_TYPE,
  OPERATOR_ELEMENT_ARGFMAX_TYPE,
  OPERATOR_ELEMENT_ARGFMIN_TYPE,
  OPERATOR_ELEMENT_ARGMAX_TYPE,
  OPERATOR_ELEMENT_ARGMIN_TYPE,
  OPERATOR_ELEMENT_PROD_TYPE,
  OPERATOR_ELEMENT_DIV_TYPE,
  OPERATOR_ELEMENT_EQ_TYPE,
  OPERATOR_ELEMENT_NEQ_TYPE,
  OPERATOR_ELEMENT_GREATER_TYPE,
  OPERATOR_ELEMENT_GEQ_TYPE,
  OPERATOR_ELEMENT_LESS_TYPE,
  OPERATOR_ELEMENT_LEQ_TYPE,
  OPERATOR_ELEMENT_POW_TYPE,
  OPERATOR_ELEMENT_FMAX_TYPE,
  OPERATOR_ELEMENT_FMIN_TYPE,
  OPERATOR_ELEMENT_MAX_TYPE,
  OPERATOR_ELEMENT_MIN_TYPE,

  OPERATOR_OUTER_PROD_TYPE,
  OPERATOR_MATRIX_DIAG_TYPE,
  OPERATOR_MATRIX_ROW_TYPE,
  OPERATOR_MATRIX_COLUMN_TYPE,
  OPERATOR_REPEAT_TYPE,
  OPERATOR_RESHAPE_TYPE,
  OPERATOR_SHIFT_TYPE,
  OPERATOR_VDIAG_TYPE,

  OPERATOR_MATRIX_PRODUCT_NN_TYPE,
  OPERATOR_MATRIX_PRODUCT_TN_TYPE,
  OPERATOR_MATRIX_PRODUCT_NT_TYPE,
  OPERATOR_MATRIX_PRODUCT_TT_TYPE,

  OPERATOR_PAIR_TYPE
};

/** @brief Groups the type of a node in the array_expression tree. Used for faster dispatching */
enum array_expression_node_type_family
{
  INVALID_TYPE_FAMILY = 0,
  COMPOSITE_OPERATOR_FAMILY,
  VALUE_TYPE_FAMILY,
  ARRAY_TYPE_FAMILY,
  INFOS_TYPE_FAMILY
};

/** @brief Encodes the type of a node in the array_expression tree. */
enum array_expression_node_subtype
{
  INVALID_SUBTYPE = 0,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE,
  REPEAT_INFOS_TYPE
};

struct op_element
{
  op_element();
  op_element(operation_node_type_family const & _type_family, operation_node_type const & _type);
  operation_node_type_family   type_family;
  operation_node_type          type;
};

struct lhs_rhs_element
{
  lhs_rhs_element();
  array_expression_node_type_family   type_family;
  array_expression_node_subtype       subtype;
  numeric_type  dtype;
  union
  {
    unsigned int        node_index;
    values_holder vscalar;
    repeat_infos  tuple;
    isaac::array* array;
  };
};

struct invalid_node{};

void fill(lhs_rhs_element &x, invalid_node);
void fill(lhs_rhs_element & x, unsigned int node_index);
void fill(lhs_rhs_element & x, array const & a);
void fill(lhs_rhs_element & x, value_scalar const & v);
void fill(lhs_rhs_element & x, repeat_infos const & r);

class array_expression : public array_base
{
public:
  struct node
  {
    lhs_rhs_element    lhs;
    op_element         op;
    lhs_rhs_element    rhs;
  };

  typedef std::vector<node>     container_type;

public:
  template<class LT, class RT>
  array_expression(LT const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape);
  template<class RT>
  array_expression(array_expression const & lhs, RT const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape);
  template<class LT>
  array_expression(LT const & lhs, array_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape);
  array_expression(array_expression const & lhs, array_expression const & rhs, op_element const & op, driver::Context const & context, numeric_type const & dtype, size4 const & shape);

  size4 shape() const;
  array_expression& reshape(int_t size1, int_t size2=1);
  int_t nshape() const;
  container_type & tree();
  container_type const & tree() const;
  std::size_t root() const;
  driver::Context const & context() const;
  numeric_type const & dtype() const;

  array_expression operator-();
  array_expression operator!();
private:
  container_type tree_;
  std::size_t root_;
  driver::Context context_;
  numeric_type dtype_;
  size4 shape_;
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
    return (driver::CommandQueue &)driver::queues[context][queue_id_];
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

template<class TYPE>
class controller
{
public:
  controller(TYPE const & x, execution_options_type const& execution_options = execution_options_type(),
             dispatcher_options_type const & dispatcher_options = dispatcher_options_type(), compilation_options_type const & compilation_options = compilation_options_type())
                : x_(x), execution_options_(execution_options), dispatcher_options_(dispatcher_options), compilation_options_(compilation_options){}
  controller(TYPE const & x, controller const & other) : x_(x), execution_options_(other.execution_options_), dispatcher_options_(other.dispatcher_options_), compilation_options_(other.compilation_options_){}
  TYPE const & x() const { return x_; }
  execution_options_type const & execution_options() const { return execution_options_; }
  dispatcher_options_type const & dispatcher_options() const { return dispatcher_options_; }
  compilation_options_type const & compilation_options() const { return compilation_options_; }
private:
  TYPE const & x_;
  execution_options_type execution_options_;
  dispatcher_options_type dispatcher_options_;
  compilation_options_type compilation_options_;
};

template<class TYPE>
controller<TYPE> control(TYPE const & x, execution_options_type const& execution_options = execution_options_type(),
                         dispatcher_options_type const & dispatcher_options = dispatcher_options_type(), compilation_options_type const & compilation_options = compilation_options_type())
{ return controller<TYPE>(x, execution_options, dispatcher_options, compilation_options); }

class expressions_tuple
{
private:
  tools::shared_ptr<array_expression> create(array_expression const & s);
public:
  typedef std::list<tools::shared_ptr<array_expression> > data_type;
  enum order_type { SEQUENTIAL, INDEPENDENT };

  expressions_tuple(array_expression const & s0);
  expressions_tuple(order_type order, array_expression const & s0, array_expression const & s1);
  expressions_tuple(data_type const & data, order_type order);

  data_type const & data() const;
  driver::Context const & context() const;
  order_type order() const;
private:
  data_type data_;
  order_type order_;
};

array_expression::node const & lhs_most(array_expression::container_type const & array, array_expression::node const & init);
array_expression::node const & lhs_most(array_expression::container_type const & array, size_t root);

}

#endif
