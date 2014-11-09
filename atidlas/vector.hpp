#ifndef ATIDLAS_VECTOR_H
#define ATIDLAS_VECTOR_H

#include <cassert>

#include "atidlas/forwards.h"
#include "atidlas/scheduler/forwards.h"
#include "atidlas/expression_template.hpp"

namespace atidlas
{

class vector : public vector_base
{
public:
    vector(atidlas_int_t size, numeric_type dtype, cl::Context context) : vector_base(size, dtype, context){}

    template<typename T>
    vector & operator=(T const & other)
    {
      vector_base::operator=(other);
      return *this;
    }
//    using vector_base::operator+=;
//    using vector_base::operator-=;
};

#define ATIDLAS_ADD_BINARY_OPERATOR(TYPE, OP) \
template<typename XL, typename XR, typename XOP, \
         typename YL, typename YR, typename YOP> \
TYPE ## _expression< const TYPE ## _expression< XL, XR, XOP>, const TYPE ## _expression< YL, YR, YOP>, OP> \
operator + (TYPE ## _expression<XL, XR, XOP> const & x, TYPE ## _expression<YL, YR, YOP> const & y) \
{ \
  assert(x.size() == y.size() && bool("Incompatible TYPE sizes!")); \
  return   TYPE ## _expression< const TYPE ## _expression<XL, XR, XOP>, const TYPE ## _expression<YL, YR, YOP>, OP>(x, y); \
} \
 \
template<typename XL, typename XR, typename XOP> \
TYPE ## _expression< const TYPE ## _expression< XL, XR, XOP>, const TYPE ## _base, OP> \
operator + (TYPE ## _expression<XL, XR, XOP> const & x, TYPE ## _base const & y) \
{ \
  assert(x.size() == y.size() && bool("Incompatible TYPE sizes!")); \
  return   TYPE ## _expression< const TYPE ## _expression<XL, XR, XOP>, const TYPE ## _base, OP>(x, y); \
} \
 \
template<typename T, typename YL, typename YR, typename YOP> \
TYPE ## _expression< const TYPE ## _expression< YL, YR, YOP>, const TYPE ## _base, OP> \
operator + (TYPE ## _base const & x, TYPE ## _expression<YL, YR, YOP> const & y) \
{ \
  assert(x.size() == y.size() && bool("Incompatible TYPE sizes!")); \
  return TYPE ## _expression<const TYPE ## _base, const TYPE ## _expression<YL, YR, YOP>, OP>(x, y); \
} \
 \
TYPE ## _expression< const TYPE ## _base, const TYPE ## _base, OP> \
operator + (TYPE ## _base const & x, TYPE ## _base const & y) \
{ \
  assert(x.size() == y.size() && bool("Incompatible TYPE sizes!")); \
  return TYPE ## _expression<const TYPE ## _base, const TYPE ## _base, OP>(x, y); \
}

ATIDLAS_ADD_BINARY_OPERATOR(vector, op_add)

#undef ATIDLAS_ADD_BINARY_OPERATOR

template<class LHS, class RHS, class OP>
vector_base & vector_base::operator=(vector_expression<LHS, RHS, OP> const & operation)
{
  scheduler::statement s(*this, op_assign(), operation);
  return *this;
}

}

#endif
