#ifndef _ATIDLAS_SYMBOLIC_EXPRESSION_H
#define _ATIDLAS_SYMBOLIC_EXPRESSION_H

#include <vector>
#include <list>
#include "atidlas/types.h"
#include "atidlas/value_scalar.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/tools/shared_ptr.hpp"

namespace atidlas
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

  // unary expression
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
  OPERATOR_VDIAG_TYPE,

  OPERATOR_MATRIX_PRODUCT_NN_TYPE,
  OPERATOR_MATRIX_PRODUCT_TN_TYPE,
  OPERATOR_MATRIX_PRODUCT_NT_TYPE,
  OPERATOR_MATRIX_PRODUCT_TT_TYPE,

  OPERATOR_PAIR_TYPE
};

/** @brief Groups the type of a node in the symbolic_expression tree. Used for faster dispatching */
enum symbolic_expression_node_type_family
{
  INVALID_TYPE_FAMILY = 0,
  COMPOSITE_OPERATOR_FAMILY,
  VALUE_TYPE_FAMILY,
  ARRAY_TYPE_FAMILY,
  INFOS_TYPE_FAMILY
};

/** @brief Encodes the type of a node in the symbolic_expression tree. */
enum symbolic_expression_node_subtype
{
  INVALID_SUBTYPE = 0,
  VALUE_SCALAR_TYPE,
  DENSE_ARRAY_TYPE,
  REPEAT_INFOS_TYPE
};

struct lhs_rhs_element
{
  lhs_rhs_element();
  lhs_rhs_element(unsigned int _node_index);
  lhs_rhs_element(atidlas::array const & x);
  lhs_rhs_element(value_scalar const & x);
  lhs_rhs_element(repeat_infos const & x);

  symbolic_expression_node_type_family   type_family;
  symbolic_expression_node_subtype       subtype;
  numeric_type  dtype;

  union
  {
    unsigned int        node_index;
    atidlas::array * array;
    values_holder vscalar;
    atidlas::repeat_infos * tuple;
  };
};


struct op_element
{
  op_element(operation_node_type_family const & _type_family, operation_node_type const & _type);
  operation_node_type_family   type_family;
  operation_node_type          type;
};


struct symbolic_expression_node
{
  symbolic_expression_node(lhs_rhs_element const & _lhs, op_element const& _op, lhs_rhs_element const & _rhs);
  lhs_rhs_element    lhs;
  op_element         op;
  lhs_rhs_element    rhs;
};

class symbolic_expression
{
public:
  typedef symbolic_expression_node              value_type;
  typedef std::vector<value_type>     container_type;

  symbolic_expression(lhs_rhs_element const & lhs, lhs_rhs_element const & rhs, op_element const & op, cl::Context const & context, numeric_type const & dtype);
  symbolic_expression(symbolic_expression const & lhs, lhs_rhs_element const & rhs, op_element const & op);
  symbolic_expression(lhs_rhs_element const & lhs, symbolic_expression const & rhs, op_element const & op);
  symbolic_expression(symbolic_expression const & lhs, symbolic_expression const & rhs, op_element const & op);

  container_type & tree();
  container_type const & tree() const;
  std::size_t root() const;
  cl::Context const & context() const;
  numeric_type const & dtype() const;
protected:
  container_type tree_;
  std::size_t root_;
  cl::Context context_;
  numeric_type dtype_;
};

struct array_expression: public symbolic_expression
{
  array_expression(lhs_rhs_element const & lhs, lhs_rhs_element const & rhs, op_element const & op, cl::Context const & ctx, numeric_type const & dtype, size4 shape);
  array_expression(symbolic_expression const & lhs, lhs_rhs_element const & rhs, op_element const & op, size4 shape);
  array_expression(lhs_rhs_element const & lhs, symbolic_expression const & rhs, op_element const & op, size4 shape);
  array_expression(symbolic_expression const & lhs, symbolic_expression const & rhs, op_element const & op, size4 shape);
  size4 shape() const;
  array_expression& reshape(int_t size1, int_t size2=1);
  int_t nshape() const;

  array_expression& operator-();
  array_expression& operator+=(value_scalar const &);
  array_expression& operator+=(array const &);
  array_expression& operator+=(array_expression const &);
  array_expression& operator-=(value_scalar const &);
  array_expression& operator-=(array const &);
  array_expression& operator-=(array_expression const &);
  array_expression& operator*=(value_scalar const &);
  array_expression& operator*=(array const &);
  array_expression& operator*=(array_expression const &);
  array_expression& operator/=(value_scalar const &);
  array_expression& operator/=(array const &);
  array_expression& operator/=(array_expression const &);
private:
  size4 shape_;
};

class symbolic_expressions_container
{
private:
  tools::shared_ptr<symbolic_expression> create(symbolic_expression const & s);
public:
  typedef std::list<tools::shared_ptr<symbolic_expression> > data_type;
  enum order_type { SEQUENTIAL, INDEPENDENT };

  symbolic_expressions_container(symbolic_expression const & s0);
  symbolic_expressions_container(order_type order, symbolic_expression const & s0, symbolic_expression const & s1);
  symbolic_expressions_container(data_type const & data, order_type order);

  data_type const & data() const;
  cl::Context const & context() const;
  order_type order() const;
private:
  data_type data_;
  order_type order_;
};

symbolic_expression_node const & lhs_most(symbolic_expression::container_type const & array, symbolic_expression_node const & init);
symbolic_expression_node const & lhs_most(symbolic_expression::container_type const & array, size_t root);

}

#endif
