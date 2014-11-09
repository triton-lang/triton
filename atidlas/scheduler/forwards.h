#ifndef ATIDLAS_SCHEDULER_STATEMENT_HPP
#define ATIDLAS_SCHEDULER_STATEMENT_HPP

#include "atidlas/forwards.h"
#include "atidlas/tools/predicate.hpp"
#include "atidlas/tools/enable_if.hpp"

#include <list>
#include <vector>

namespace atidlas
{
namespace scheduler
{

/** @brief Optimization enum for grouping operations into unary or binary operations. Just for optimization of lookups. */
enum operation_node_type_family
{
  OPERATION_INVALID_TYPE_FAMILY = 0,

  // unary or binary expression
  OPERATION_UNARY_TYPE_FAMILY,
  OPERATION_BINARY_TYPE_FAMILY,

  //reductions
  OPERATION_VECTOR_REDUCTION_TYPE_FAMILY,
  OPERATION_ROWS_REDUCTION_TYPE_FAMILY,
  OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY
};

/** @brief Enumeration for identifying the possible operations */
enum operation_node_type
{
  OPERATION_INVALID_TYPE = 0,

  // unary operator
  OPERATION_UNARY_MINUS_TYPE,

  // unary expression
  OPERATION_UNARY_CAST_CHAR_TYPE,
  OPERATION_UNARY_CAST_UCHAR_TYPE,
  OPERATION_UNARY_CAST_SHORT_TYPE,
  OPERATION_UNARY_CAST_USHORT_TYPE,
  OPERATION_UNARY_CAST_INT_TYPE,
  OPERATION_UNARY_CAST_UINT_TYPE,
  OPERATION_UNARY_CAST_LONG_TYPE,
  OPERATION_UNARY_CAST_ULONG_TYPE,
  OPERATION_UNARY_CAST_HALF_TYPE,
  OPERATION_UNARY_CAST_FLOAT_TYPE,
  OPERATION_UNARY_CAST_DOUBLE_TYPE,

  OPERATION_UNARY_ABS_TYPE,
  OPERATION_UNARY_ACOS_TYPE,
  OPERATION_UNARY_ASIN_TYPE,
  OPERATION_UNARY_ATAN_TYPE,
  OPERATION_UNARY_CEIL_TYPE,
  OPERATION_UNARY_COS_TYPE,
  OPERATION_UNARY_COSH_TYPE,
  OPERATION_UNARY_EXP_TYPE,
  OPERATION_UNARY_FABS_TYPE,
  OPERATION_UNARY_FLOOR_TYPE,
  OPERATION_UNARY_LOG_TYPE,
  OPERATION_UNARY_LOG10_TYPE,
  OPERATION_UNARY_SIN_TYPE,
  OPERATION_UNARY_SINH_TYPE,
  OPERATION_UNARY_SQRT_TYPE,
  OPERATION_UNARY_TAN_TYPE,
  OPERATION_UNARY_TANH_TYPE,

  OPERATION_UNARY_TRANS_TYPE,
  OPERATION_UNARY_NORM_1_TYPE,
  OPERATION_UNARY_NORM_2_TYPE,
  OPERATION_UNARY_NORM_INF_TYPE,
  OPERATION_UNARY_MAX_TYPE,
  OPERATION_UNARY_MIN_TYPE,

  // binary expression
  OPERATION_BINARY_ACCESS_TYPE,
  OPERATION_BINARY_ASSIGN_TYPE,
  OPERATION_BINARY_INPLACE_ADD_TYPE,
  OPERATION_BINARY_INPLACE_SUB_TYPE,
  OPERATION_BINARY_ADD_TYPE,
  OPERATION_BINARY_SUB_TYPE,
  OPERATION_BINARY_MULT_TYPE,
  OPERATION_BINARY_DIV_TYPE,
  OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE,
  OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE,
  OPERATION_BINARY_ELEMENT_ARGMAX_TYPE,
  OPERATION_BINARY_ELEMENT_ARGMIN_TYPE,
  OPERATION_BINARY_ELEMENT_PROD_TYPE,
  OPERATION_BINARY_ELEMENT_DIV_TYPE,
  OPERATION_BINARY_ELEMENT_EQ_TYPE,
  OPERATION_BINARY_ELEMENT_NEQ_TYPE,
  OPERATION_BINARY_ELEMENT_GREATER_TYPE,
  OPERATION_BINARY_ELEMENT_GEQ_TYPE,
  OPERATION_BINARY_ELEMENT_LESS_TYPE,
  OPERATION_BINARY_ELEMENT_LEQ_TYPE,
  OPERATION_BINARY_ELEMENT_POW_TYPE,
  OPERATION_BINARY_ELEMENT_FMAX_TYPE,
  OPERATION_BINARY_ELEMENT_FMIN_TYPE,
  OPERATION_BINARY_ELEMENT_MAX_TYPE,
  OPERATION_BINARY_ELEMENT_MIN_TYPE,

  OPERATION_BINARY_MATRIX_DIAG_TYPE,
  OPERATION_BINARY_VECTOR_DIAG_TYPE,
  OPERATION_BINARY_MATRIX_ROW_TYPE,
  OPERATION_BINARY_MATRIX_COLUMN_TYPE,
  OPERATION_BINARY_MAT_VEC_PROD_TYPE,
  OPERATION_BINARY_MAT_MAT_PROD_TYPE,
  OPERATION_BINARY_INNER_PROD_TYPE

};



namespace result_of
{
  template<typename T>
  struct op_type_info
  {
    typedef typename T::ERROR_UNKNOWN_OP_TYPE   error_type;
  };

  // elementwise casts
  template<> struct op_type_info<op_element_cast<char> >      { enum { id = OPERATION_UNARY_CAST_CHAR_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<unsigned char> >      { enum { id = OPERATION_UNARY_CAST_UCHAR_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<short> >      { enum { id = OPERATION_UNARY_CAST_SHORT_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<unsigned short> >      { enum { id = OPERATION_UNARY_CAST_USHORT_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<int> >      { enum { id = OPERATION_UNARY_CAST_INT_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<unsigned int> >      { enum { id = OPERATION_UNARY_CAST_UINT_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<long> >      { enum { id = OPERATION_UNARY_CAST_LONG_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<unsigned long> >      { enum { id = OPERATION_UNARY_CAST_ULONG_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<float> >      { enum { id = OPERATION_UNARY_CAST_FLOAT_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_cast<double> >      { enum { id = OPERATION_UNARY_CAST_DOUBLE_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };

  // elementwise functions
  template<> struct op_type_info<op_element_unary<op_abs>   >      { enum { id = OPERATION_UNARY_ABS_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_acos>  >      { enum { id = OPERATION_UNARY_ACOS_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_asin>  >      { enum { id = OPERATION_UNARY_ASIN_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_atan>  >      { enum { id = OPERATION_UNARY_ATAN_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_ceil>  >      { enum { id = OPERATION_UNARY_CEIL_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_cos>   >      { enum { id = OPERATION_UNARY_COS_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_cosh>  >      { enum { id = OPERATION_UNARY_COSH_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_exp>   >      { enum { id = OPERATION_UNARY_EXP_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_fabs>  >      { enum { id = OPERATION_UNARY_FABS_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_floor> >      { enum { id = OPERATION_UNARY_FLOOR_TYPE,        family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_log>   >      { enum { id = OPERATION_UNARY_LOG_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_log10> >      { enum { id = OPERATION_UNARY_LOG10_TYPE,        family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_sin>   >      { enum { id = OPERATION_UNARY_SIN_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_sinh>  >      { enum { id = OPERATION_UNARY_SINH_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_sqrt>  >      { enum { id = OPERATION_UNARY_SQRT_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_tan>   >      { enum { id = OPERATION_UNARY_TAN_TYPE,          family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_unary<op_tanh>  >      { enum { id = OPERATION_UNARY_TANH_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };

  template<> struct op_type_info<op_element_binary<op_argmax> >       { enum { id = OPERATION_BINARY_ELEMENT_ARGMAX_TYPE ,     family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_argmin> >       { enum { id = OPERATION_BINARY_ELEMENT_ARGMIN_TYPE ,     family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_pow> >       { enum { id = OPERATION_BINARY_ELEMENT_POW_TYPE ,     family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_eq> >        { enum { id = OPERATION_BINARY_ELEMENT_EQ_TYPE,       family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_neq> >       { enum { id = OPERATION_BINARY_ELEMENT_NEQ_TYPE,      family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_greater> >   { enum { id = OPERATION_BINARY_ELEMENT_GREATER_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_less> >      { enum { id = OPERATION_BINARY_ELEMENT_LESS_TYPE,     family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_geq> >       { enum { id = OPERATION_BINARY_ELEMENT_GEQ_TYPE,      family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_leq> >       { enum { id = OPERATION_BINARY_ELEMENT_LEQ_TYPE,      family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_fmax> >       { enum { id = OPERATION_BINARY_ELEMENT_FMAX_TYPE,    family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_fmin> >       { enum { id = OPERATION_BINARY_ELEMENT_FMIN_TYPE,    family = OPERATION_BINARY_TYPE_FAMILY}; };


  //structurewise function
  template<> struct op_type_info<op_norm_1                  >      { enum { id = OPERATION_UNARY_NORM_1_TYPE,        family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_norm_2                  >      { enum { id = OPERATION_UNARY_NORM_2_TYPE,        family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_norm_inf                >      { enum { id = OPERATION_UNARY_NORM_INF_TYPE,      family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_max                     >      { enum { id = OPERATION_UNARY_MAX_TYPE,           family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_min                     >      { enum { id = OPERATION_UNARY_MIN_TYPE,           family = OPERATION_UNARY_TYPE_FAMILY}; };

  template<> struct op_type_info<op_trans                   >      { enum { id = OPERATION_UNARY_TRANS_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_row                   >      { enum { id = OPERATION_BINARY_MATRIX_ROW_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_column                   >      { enum { id = OPERATION_BINARY_MATRIX_COLUMN_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY}; };

  template<> struct op_type_info<op_matrix_diag>                    { enum { id = OPERATION_BINARY_MATRIX_DIAG_TYPE,   family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_vector_diag>                    { enum { id = OPERATION_BINARY_VECTOR_DIAG_TYPE,   family = OPERATION_BINARY_TYPE_FAMILY}; };

  template<> struct op_type_info<op_prod>                          { enum { id = OPERATION_BINARY_MAT_VEC_PROD_TYPE, family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_mat_mat_prod>                  { enum { id = OPERATION_BINARY_MAT_MAT_PROD_TYPE, family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_inner_prod>                    { enum { id = OPERATION_BINARY_INNER_PROD_TYPE,   family = OPERATION_BINARY_TYPE_FAMILY}; };

  template<typename OP> struct op_type_info<op_reduce_vector<OP>   >      { enum { id = op_type_info<OP>::id,        family = OPERATION_VECTOR_REDUCTION_TYPE_FAMILY}; };
  template<typename OP> struct op_type_info<op_reduce_rows<OP>   >       { enum { id = op_type_info<OP>::id,         family = OPERATION_ROWS_REDUCTION_TYPE_FAMILY}; };
  template<typename OP> struct op_type_info<op_reduce_columns<OP>   >      { enum { id = op_type_info<OP>::id,       family = OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY}; };

  //elementwise operator
  template<> struct op_type_info<op_assign>                        { enum { id = OPERATION_BINARY_ASSIGN_TYPE,       family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_inplace_add>                   { enum { id = OPERATION_BINARY_INPLACE_ADD_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_inplace_sub>                   { enum { id = OPERATION_BINARY_INPLACE_SUB_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_add>                           { enum { id = OPERATION_BINARY_ADD_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_sub>                           { enum { id = OPERATION_BINARY_SUB_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_prod> >      { enum { id = OPERATION_BINARY_ELEMENT_PROD_TYPE, family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_element_binary<op_div>  >      { enum { id = OPERATION_BINARY_ELEMENT_DIV_TYPE,  family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_mult>                          { enum { id = OPERATION_BINARY_MULT_TYPE,         family = OPERATION_BINARY_TYPE_FAMILY}; };
  template<> struct op_type_info<op_div>                           { enum { id = OPERATION_BINARY_DIV_TYPE,          family = OPERATION_BINARY_TYPE_FAMILY}; };

  template<> struct op_type_info<op_flip_sign>                     { enum { id = OPERATION_UNARY_MINUS_TYPE,         family = OPERATION_UNARY_TYPE_FAMILY}; };


  /** \endcond */
} // namespace result_of





/** @brief Groups the type of a node in the statement tree. Used for faster dispatching */
enum statement_node_type_family
{
  INVALID_TYPE_FAMILY = 0,
  // LHS or RHS are again an expression:
  COMPOSITE_OPERATION_FAMILY,
  // device scalars:
  SCALAR_TYPE_FAMILY,
  // vector:
  VECTOR_TYPE_FAMILY,
  // matrices:
  MATRIX_TYPE_FAMILY
};

/** @brief Encodes the type of a node in the statement tree. */
enum statement_node_subtype
{
  INVALID_SUBTYPE = 0,

  HOST_SCALAR_TYPE,
  DEVICE_SCALAR_TYPE,

  DENSE_VECTOR_TYPE,
  IMPLICIT_VECTOR_TYPE,

  DENSE_MATRIX_TYPE,
  IMPLICIT_MATRIX_TYPE,
};

/** @brief A class representing the 'data' for the LHS or RHS operand of the respective node.
  *
  * If it represents a compound expression, the union holds the array index within the respective statement array.
  * If it represents a object with data (vector, matrix, etc.) it holds the respective pointer (scalar, vector, matrix) or value (host scalar)
  *
  * The member 'type_family' is an optimization for quickly retrieving the 'type', which denotes the currently 'active' member in the union
  */
struct lhs_rhs_element
{
  statement_node_type_family   type_family;
  statement_node_subtype       subtype;
  numeric_type  numeric_t;

  union
  {
    unsigned int        node_index;
    atidlas::vector_base * vector;
    atidlas::matrix_base * matrix;
  };
};


/** @brief Struct for holding the type family as well as the type of an operation (could be addition, subtraction, norm, etc.) */
struct op_element
{
  operation_node_type_family   type_family;
  operation_node_type          type;
};

/** @brief Main datastructure for an node in the statement tree */
struct statement_node
{
  lhs_rhs_element    lhs;
  op_element         op;
  lhs_rhs_element    rhs;
};

namespace result_of
{

  template<class T> struct num_nodes { enum { value = 0 }; };
  template<class LHS, class OP, class RHS> struct num_nodes<       vector_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };
  template<class LHS, class OP, class RHS> struct num_nodes< const vector_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };
  template<class LHS, class OP, class RHS> struct num_nodes<       matrix_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };
  template<class LHS, class OP, class RHS> struct num_nodes< const matrix_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };
  template<class LHS, class OP, class RHS> struct num_nodes<       scalar_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };
  template<class LHS, class OP, class RHS> struct num_nodes< const scalar_expression<LHS, RHS, OP> > { enum { value = 1 + num_nodes<LHS>::value + num_nodes<RHS>::value + num_nodes<OP>::value }; };

}

/** \brief The main class for representing a statement such as x = inner_prod(y,z); at runtime.
  *
  * This is the equivalent to an expression template tree, but entirely built at runtime in order to perform really cool stuff such as kernel fusion.
  */
class statement
{
public:
  typedef statement_node              value_type;
  typedef std::vector<value_type>     container_type;

  statement(container_type const & custom_array) : array_(custom_array) {}

  /** @brief Generate the runtime statement from an expression template.
      *
      * Constructing a runtime statement from expression templates makes perfect sense, because this way only a single allocation is needed when creating the statement. */
  template<typename LHS, typename OP, typename RHS>
  statement(LHS & lhs, OP const &, RHS const & rhs) : array_(1 + result_of::num_nodes<RHS>::value)
  {
    array_[0].op.type_family = operation_node_type_family(result_of::op_type_info<OP>::family);
    array_[0].op.type        = operation_node_type(result_of::op_type_info<OP>::id);
    add_lhs(0, 1, lhs);
    add_rhs(0, 1, rhs);
  }

  container_type const & array() const { return array_; }
  unsigned int root() const { return 0; }
private:

  //////////// Tree nodes (non-terminals) ////////////////////

  unsigned int add_element(unsigned int next_free, lhs_rhs_element & elem, vector_base const & x)
  {
    elem.type_family  = VECTOR_TYPE_FAMILY;
    elem.subtype      = DENSE_VECTOR_TYPE;
    elem.vector = const_cast<vector_base*>(&x);
    return next_free;
  }

  template<typename LHS, typename RHS, typename OP>
  unsigned int add_element(unsigned int       next_free,
                         lhs_rhs_element & elem,
                         scalar_expression<LHS, RHS, OP> const & t)
  {
    elem.type_family  = COMPOSITE_OPERATION_FAMILY;
    elem.subtype      = INVALID_SUBTYPE;
    elem.node_index   = next_free;
    return add_node(next_free, next_free + 1, t);
  }

  template<typename LHS, typename RHS, typename OP>
  unsigned int add_element(unsigned int       next_free,
                         lhs_rhs_element & elem,
                         vector_expression<LHS, RHS, OP> const & t)
  {
    elem.type_family  = COMPOSITE_OPERATION_FAMILY;
    elem.subtype      = INVALID_SUBTYPE;
    elem.node_index   = next_free;
    return add_node(next_free, next_free + 1, t);
  }

  template<typename LHS, typename RHS, typename OP>
  unsigned int add_element(unsigned int next_free,
                         lhs_rhs_element & elem,
                         matrix_expression<LHS, RHS, OP> const & t)
  {
    elem.type_family   = COMPOSITE_OPERATION_FAMILY;
    elem.subtype      = INVALID_SUBTYPE;
    elem.numeric_t = INVALID_NUMERIC_TYPE;
    elem.node_index    = next_free;
    return add_node(next_free, next_free + 1, t);
  }

  template<typename T>
  unsigned int add_lhs(unsigned int current_index, unsigned int next_free, T const & t)
  { return add_element(next_free, array_[current_index].lhs, t);  }

  template<typename T>
  unsigned int add_rhs(unsigned int current_index, unsigned int next_free, T const & t)
  { return add_element(next_free, array_[current_index].rhs, t);  }

  template<template<typename, typename, typename> class ExpressionT, typename LHS, typename RHS, typename OP>
  unsigned int add_node(unsigned int current_index, unsigned int next_free, ExpressionT<LHS, RHS, OP> const & proxy)
  {
    // set OP:
    array_[current_index].op.type_family = operation_node_type_family(result_of::op_type_info<OP>::family);
    array_[current_index].op.type        = operation_node_type(result_of::op_type_info<OP>::id);

    // set LHS and RHS:
    if (array_[current_index].op.type_family == OPERATION_UNARY_TYPE_FAMILY)
    {
      // unary expression: set rhs to invalid:
      array_[current_index].rhs.type_family  = INVALID_TYPE_FAMILY;
      array_[current_index].rhs.subtype      = INVALID_SUBTYPE;
      array_[current_index].rhs.numeric_t = INVALID_NUMERIC_TYPE;
      return add_lhs(current_index, next_free, proxy.lhs());
    }

    return add_rhs(current_index, add_lhs(current_index, next_free, proxy.lhs()), proxy.rhs());
  }

  container_type   array_;
};

class statements_container
{
public:
  typedef std::list<scheduler::statement> data_type;
  enum order_type { SEQUENTIAL, INDEPENDENT };

  statements_container(data_type const & data, order_type order) : data_(data), order_(order)
  { }

  statements_container(scheduler::statement const & s0) : order_(INDEPENDENT)
  {
    data_.push_back(s0);
  }

  statements_container(scheduler::statement const & s0, scheduler::statement const & s1, order_type order) : order_(order)
  {
    data_.push_back(s0);
    data_.push_back(s1);
  }

  std::list<scheduler::statement> const & data() const { return data_; }
  order_type order() const { return order_; }
private:
  std::list<scheduler::statement> data_;
  order_type order_;
};

} // namespace scheduler
} // namespace viennacl

#endif

