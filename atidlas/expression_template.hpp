#ifndef ATIDLAS_EXPRESSION_TEMPLATE_HPP_
#define ATIDLAS_EXPRESSION_TEMPLATE_HPP_

#include "atidlas/forwards.h"
#include "atidlas/traits/size.hpp"

namespace atidlas
{

namespace detail
{

template<typename T>
struct reference_if_nonscalar
{
  typedef T &    type;
};

#define ATIDLAS_REFERENCE_IF_NONSCALAR(TNAME) \
template<> struct reference_if_nonscalar<TNAME>                { typedef                TNAME  type; }; \
template<> struct reference_if_nonscalar<const TNAME>          { typedef          const TNAME  type; };

  ATIDLAS_REFERENCE_IF_NONSCALAR(char)
  ATIDLAS_REFERENCE_IF_NONSCALAR(short)
  ATIDLAS_REFERENCE_IF_NONSCALAR(int)
  ATIDLAS_REFERENCE_IF_NONSCALAR(long)
  ATIDLAS_REFERENCE_IF_NONSCALAR(unsigned char)
  ATIDLAS_REFERENCE_IF_NONSCALAR(unsigned short)
  ATIDLAS_REFERENCE_IF_NONSCALAR(unsigned int)
  ATIDLAS_REFERENCE_IF_NONSCALAR(unsigned long)

  ATIDLAS_REFERENCE_IF_NONSCALAR(float)
  ATIDLAS_REFERENCE_IF_NONSCALAR(double)
#undef ATIDLAS_REFERENCE_IF_NONSCALAR

}

/** @brief An expression template class that represents a binary operation
* @tparam LHS   left hand side operand
* @tparam RHS   right hand side operand
* @tparam OP    the operator
*/
template<typename LHS, typename RHS, typename OP>
class expression_template
{
  typedef typename detail::reference_if_nonscalar<LHS>::type     lhs_reference_type;
  typedef typename detail::reference_if_nonscalar<RHS>::type     rhs_reference_type;
public:
  expression_template(LHS & l, RHS & r) : lhs_(l), rhs_(r) {}
  /** @brief Get left hand side operand */
  lhs_reference_type lhs() const { return lhs_; }
  /** @brief Get right hand side operand  */
  rhs_reference_type rhs() const { return rhs_; }
  /** @brief Returns the size of the result vector */
  atidlas_int_t size() const { return traits::size(*this); }
private:
  /** @brief The left hand side operand */
  lhs_reference_type lhs_;
  /** @brief The right hand side operand */
  rhs_reference_type rhs_;
};

template<typename LHS, typename RHS, typename OP>
struct vector_expression: public expression_template<LHS, RHS, OP>{
    vector_expression(LHS & l, RHS & r) : expression_template<LHS, RHS, OP>(l, r){ }
};

template<typename LHS, typename RHS, typename OP>
class matrix_expression: public expression_template<LHS, RHS, OP>{
    matrix_expression(LHS & l, RHS & r) : expression_template<LHS, RHS, OP>(l, r){ }
};

template<typename LHS, typename RHS, typename OP>
class scalar_expression: public expression_template<LHS, RHS, OP>{
    scalar_expression(LHS & l, RHS & r) : expression_template<LHS, RHS, OP>(l, r){ }
};

}

#endif
