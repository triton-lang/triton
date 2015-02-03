#include <cassert>

#include "atidlas/array.h"
#include <CL/cl.hpp>
#include "atidlas/exception/unknown_datatype.h"
#include "atidlas/model/model.h"
#include "atidlas/symbolic/execute.h"

#include <stdexcept>

namespace atidlas
{

array_infos array::init_infos(numeric_type dtype, cl_mem data, int_t shape1, int_t shape2, int_t start1, int_t start2, int_t stride1, int_t stride2, int_t ld)
{
  array_infos res;
  res.dtype = dtype;
  res.data = data;
  res.shape1 = shape1;
  res.shape2 = shape2;
  res.start1 = start1;
  res.start2 = start2;
  res.stride1 = stride1;
  res.stride2 = stride2;
  res.ld = ld;
  return res;
}
/*--- Constructors ---*/

//1D Constructors

array::array(int_t size1, numeric_type dtype, cl::Context context) :
  context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype)*size1),
  infos_(init_infos(dtype, data_(), size1, 1, 0, 0, 1, 1, size1))
{}

template<class DT>
array::array(std::vector<DT> const & x, cl::Context context):
  context_(context), data_(context, CL_MEM_READ_WRITE, size_of(to_numeric_type<DT>::value)*x.size()),
  infos_(init_infos(to_numeric_type<DT>::value, data_(), x.size(), 1, 0, 0, 1, 1, x.size()))
{ *this = x; }

array::array(array & v, slice const & s1) : context_(v.data_.getInfo<CL_MEM_CONTEXT>()), data_(v.data_),
 infos_(init_infos(v.infos_.dtype, data_(), s1.size, 1, v.infos_.start1 + v.infos_.stride1*s1.start, 0, v.infos_.stride1*s1.stride, 1, v.infos_.ld))
{}

#define INSTANTIATE(T) template array::array<T>(std::vector<T> const &, cl::Context)
INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE

// 2D
array::array(int_t size1, int_t size2, numeric_type dtype, cl::Context context) :
context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype)*size1*size2),
infos_(init_infos(dtype, data_(), size1, size2, 0, 0, 1, 1, size1))
{}

array::array(array & M, slice const & s1, slice const & s2) :
context_(M.data_.getInfo<CL_MEM_CONTEXT>()), data_(M.data_),
infos_(init_infos(M.dtype(), data_(), s1.size, s2.size, M.start()._1 + M.stride()._1*s1.start, M.start()._2 + M.stride()._2*s2.start,
       M.stride()._1*s1.stride, M.stride()._2*s2.stride, M.ld()))
{ }

template<typename DT>
array::array(int_t size1, int_t size2, std::vector<DT> const & data, cl::Context context):
context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(to_numeric_type<DT>::value)*size1*size2),
infos_(init_infos(to_numeric_type<DT>::value, data_(), size1, size2, 0, 0, 1, 1, size1))
{
  atidlas::copy(data, *this);
}

#define INSTANTIATE(T) template array::array<T>(int_t, int_t, std::vector<T> const &, cl::Context)
INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE

// General
array::array(numeric_type dtype, cl::Buffer data, slice const & s1, slice const & s2, int_t ld, cl::Context context):
context_(context), data_(data),
infos_(init_infos(dtype, data_(), s1.size, s2.size, s1.start, s2.start, s1.stride, s2.stride, ld))
{ }

array::array(control const & x):
context_(x.expression().context()), data_(context_, CL_MEM_READ_WRITE, size_of(x.expression().dtype())*prod(x.expression().shape())),
infos_(init_infos(x.expression().dtype(), data_(), x.expression().shape()._1, x.expression().shape()._2, 0, 0, 1, 1, x.expression().shape()._1))
{
  *this = x;
}

array::array(array const & x) :
context_(x.context()), data_(context_, CL_MEM_READ_WRITE, size_of(x.dtype())*x.shape()._1*x.shape()._2),
infos_(init_infos(x.dtype(), data_(), x.shape()._1, x.shape()._2, 0, 0, 1, 1, x.shape()._1))
{
  *this = x;
}


/*--- Getters ---*/
numeric_type array::dtype() const
{ return infos_.dtype; }

size4 array::shape() const
{ return size4(infos_.shape1, infos_.shape2); }

int_t array::nshape() const
{ return int_t((infos_.shape1 > 1) + (infos_.shape2 > 1)); }

size4 array::start() const
{ return size4(infos_.start1, infos_.start2); }

size4 array::stride() const
{ return size4(infos_.stride1, infos_.stride2); }

int_t array::ld() const
{ return infos_.ld; }

cl::Context const & array::context() const
{ return context_; }

cl::Buffer const & array::data() const
{ return data_; }

int_t array::dsize() const
{ return infos_.ld*infos_.shape2; }

/*--- Assignment Operators ----*/
//---------------------------------------
array & array::operator=(array const & rhs)
{
  assert(dtype() == rhs.dtype());
  array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), context_, dtype(), shape());
  cl::CommandQueue & queue = cl_ext::get_queue(context_, 0);
  model_map_t & mmap = atidlas::get_model_map(queue);
  execute(expression, mmap);
  return *this;
}

array & array::operator=(control const & x)
{
  array_expression const & rhs = x.expression();

  assert(dtype() == rhs.dtype());
  array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), dtype(), shape());
  cl::CommandQueue & queue = cl_ext::get_queue(context_, 0);
  model_map_t & mmap = atidlas::get_model_map(queue);
  execute(expression, mmap);
  return *this;
}

template<class DT>
array & array::operator=(std::vector<DT> const & rhs)
{
  assert(nshape()==1);
  atidlas::copy(rhs, *this);
  return *this;
}

#define INSTANTIATE(T) template array & array::operator=<T>(std::vector<T> const &)

INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE

array_expression array::operator-()
{ return array_expression(*this, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype(), shape()); }

array_expression array::operator!()
{ return array_expression(*this, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_NEGATE_TYPE), context_, INT_TYPE, shape()); }

//
array & array::operator+=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype(), shape()); }

array & array::operator+=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype(), shape()); }

array & array::operator+=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), dtype(), shape()); }
//----
array & array::operator-=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype(), shape()); }

array & array::operator-=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype(), shape()); }

array & array::operator-=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), dtype(), shape()); }
//----
array & array::operator*=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype(), shape()); }

array & array::operator*=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype(), shape()); }

array & array::operator*=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), dtype(), shape()); }
//----
array & array::operator/=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype(), shape()); }

array & array::operator/=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype(), shape()); }

array & array::operator/=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), dtype(), shape()); }

array_expression array::T() const
{ return atidlas::trans(*this) ;}

/*--- Indexing operators -----*/
//---------------------------------------
scalar array::operator [](int_t idx)
{
  assert(nshape()==1);
  return scalar(dtype(), data_, idx, context_);
}

const scalar array::operator [](int_t idx) const
{
  assert(nshape()==1);
  return scalar(dtype(), data_, idx, context_);
}


array array::operator[](slice const & e1)
{
  assert(nshape()==1);
  return array(*this, e1);
}

array array::operator()(slice const & e1, slice const & e2)
{ return array(*this, e1, e2); }
//---------------------------------------

/*--- Scalar ---*/
namespace detail
{

template<class T>
void copy(cl::Context & ctx, cl::Buffer const & data, T value)
{
  cl_ext::get_queue(ctx, 0).enqueueWriteBuffer(data, CL_TRUE, 0, sizeof(T), (void*)&value);
}

}

scalar::scalar(numeric_type dtype, const cl::Buffer &data, int_t offset, cl::Context context): array(dtype, data, _(offset, offset+1), _(1,2), 1, context)
{ }

scalar::scalar(value_scalar value, cl::Context context) : array(1, value.dtype(), context)
{
  switch(dtype())
  {
//    case BOOL_TYPE: detail::copy(context_, data_, (cl_bool)value); break;
    case CHAR_TYPE: detail::copy(context_, data_, (cl_char)value); break;
    case UCHAR_TYPE: detail::copy(context_, data_, (cl_uchar)value); break;
    case SHORT_TYPE: detail::copy(context_, data_, (cl_short)value); break;
    case USHORT_TYPE: detail::copy(context_, data_, (cl_ushort)value); break;
    case INT_TYPE: detail::copy(context_, data_, (cl_int)value); break;
    case UINT_TYPE: detail::copy(context_, data_, (cl_uint)value); break;
    case LONG_TYPE: detail::copy(context_, data_, (cl_long)value); break;
    case ULONG_TYPE: detail::copy(context_, data_, (cl_ulong)value); break;
//    case HALF_TYPE: detail::copy(context_, data_, (cl_float)value); break;
    case FLOAT_TYPE: detail::copy(context_, data_, (cl_float)value); break;
    case DOUBLE_TYPE: detail::copy(context_, data_, (cl_double)value); break;
    default: throw unknown_datatype(dtype());
  }
}


scalar::scalar(numeric_type dtype, cl::Context context) : array(1, dtype, context)
{ }

scalar::scalar(control const &proxy) : array(proxy){ }

template<class T>
T scalar::cast() const
{
  values_holder v;
  int_t dtsize = size_of(dtype());
#define HANDLE_CASE(DTYPE, VAL) \
case DTYPE:\
  cl_ext::get_queue(context_, 0).enqueueReadBuffer(data_, CL_TRUE, infos_.start1*dtsize, dtsize, (void*)&v.VAL);\
  return v.VAL

  switch(dtype())
  {
//    HANDLE_CASE(BOOL_TYPE, bool8);
    HANDLE_CASE(CHAR_TYPE, int8);
    HANDLE_CASE(UCHAR_TYPE, uint8);
    HANDLE_CASE(SHORT_TYPE, int16);
    HANDLE_CASE(USHORT_TYPE, uint16);
    HANDLE_CASE(INT_TYPE, int32);
    HANDLE_CASE(UINT_TYPE, uint32);
    HANDLE_CASE(LONG_TYPE, int64);
    HANDLE_CASE(ULONG_TYPE, uint64);
//    HANDLE_CASE(HALF_TYPE, float16);
    HANDLE_CASE(FLOAT_TYPE, float32);
    HANDLE_CASE(DOUBLE_TYPE, float64);
    default: throw unknown_datatype(dtype());
  }
#undef HANDLE_CASE

}

scalar& scalar::operator=(value_scalar const & s)
{
  cl::CommandQueue& queue = cl_ext::get_queue(context_, 0);
  int_t dtsize = size_of(dtype());

#define HANDLE_CASE(TYPE, CLTYPE) case TYPE:\
                            {\
                              CLTYPE v = s;\
                              queue.enqueueWriteBuffer(data_, CL_TRUE, infos_.start1*dtsize, dtsize, (void*)&v);\
                              return *this;\
                            }
  switch(dtype())
  {
//    HANDLE_CASE(BOOL_TYPE, cl_bool)
    HANDLE_CASE(CHAR_TYPE, cl_char)
    HANDLE_CASE(UCHAR_TYPE, cl_uchar)
    HANDLE_CASE(SHORT_TYPE, cl_short)
    HANDLE_CASE(USHORT_TYPE, cl_ushort)
    HANDLE_CASE(INT_TYPE, cl_int)
    HANDLE_CASE(UINT_TYPE, cl_uint)
    HANDLE_CASE(LONG_TYPE, cl_long)
    HANDLE_CASE(ULONG_TYPE, cl_ulong)
//    HANDLE_CASE(HALF_TYPE, cl_half)
    HANDLE_CASE(FLOAT_TYPE, cl_float)
    HANDLE_CASE(DOUBLE_TYPE, cl_double)
    default: throw unknown_datatype(dtype());
  }
}

//scalar& scalar::operator=(scalar const & s)
//{
//  return scalar::operator =(value_scalar(s));
//}

#define INSTANTIATE(type) scalar::operator type() const { return cast<type>(); }
  INSTANTIATE(cl_char)
  INSTANTIATE(cl_uchar)
  INSTANTIATE(cl_short)
  INSTANTIATE(cl_ushort)
  INSTANTIATE(cl_int)
  INSTANTIATE(cl_uint)
  INSTANTIATE(cl_long)
  INSTANTIATE(cl_ulong)
  INSTANTIATE(cl_float)
  INSTANTIATE(cl_double)
#undef INSTANTIATE

std::ostream & operator<<(std::ostream & os, scalar const & s)
{
  switch(s.dtype())
  {
//    case BOOL_TYPE: return os << static_cast<cl_bool>(s);
    case CHAR_TYPE: return os << static_cast<cl_char>(s);
    case UCHAR_TYPE: return os << static_cast<cl_uchar>(s);
    case SHORT_TYPE: return os << static_cast<cl_short>(s);
    case USHORT_TYPE: return os << static_cast<cl_ushort>(s);
    case INT_TYPE: return os << static_cast<cl_int>(s);
    case UINT_TYPE: return os << static_cast<cl_uint>(s);
    case LONG_TYPE: return os << static_cast<cl_long>(s);
    case ULONG_TYPE: return os << static_cast<cl_ulong>(s);
//    case HALF_TYPE: return os << static_cast<cl_half>(s);
    case FLOAT_TYPE: return os << static_cast<cl_float>(s);
    case DOUBLE_TYPE: return os << static_cast<cl_double>(s);
    default: throw unknown_datatype(s.dtype());
  }
}

/*--- Binary Operators ----*/
//-----------------------------------
template<class U, class V>
size4 elementwise_size(U const & u, V const & v)
{
  if(max(u.shape())==1)
    return v.shape();
  return u.shape();
}

template<class U, class V>
bool check_elementwise(U const & u, V const & v)
{
  return max(u.shape())==1 || max(v.shape())==1 || u.shape()==v.shape();
}

#define DEFINE_ELEMENT_BINARY_OPERATOR(OP, OPNAME, DTYPE) \
array_expression OPNAME (array_expression const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), DTYPE, elementwise_size(x, y)); } \
 \
array_expression OPNAME (array const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), DTYPE, elementwise_size(x, y)); } \
\
array_expression OPNAME (array_expression const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), DTYPE, elementwise_size(x, y)); } \
\
array_expression OPNAME (array const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, elementwise_size(x, y)); }\
\
array_expression OPNAME (array_expression const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), DTYPE, x.shape()); } \
\
array_expression OPNAME (array const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, x.shape()); }\
\
array_expression OPNAME (value_scalar const & y, array_expression const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), DTYPE, x.shape()); } \
\
array_expression OPNAME (value_scalar const & y, array const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, x.shape()); }

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ADD_TYPE, operator +, x.dtype())
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_SUB_TYPE, operator -, x.dtype())
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_MULT_TYPE, operator *, x.dtype())
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_DIV_TYPE, operator /, x.dtype())

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MAX_TYPE, maximum, x.dtype())
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MIN_TYPE, minimum, x.dtype())
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_POW_TYPE, pow, x.dtype())

namespace detail
{ DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ASSIGN_TYPE, assign, x.dtype()) }


DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GREATER_TYPE, operator >, INT_TYPE)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GEQ_TYPE, operator >=, INT_TYPE)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LESS_TYPE, operator <, INT_TYPE)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LEQ_TYPE, operator <=, INT_TYPE)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_EQ_TYPE, operator ==, INT_TYPE)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_NEQ_TYPE, operator !=, INT_TYPE)


array_expression outer(array const & x, array const & y)
{
  assert(x.nshape()==1 && y.nshape()==1);
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_OUTER_PROD_TYPE), x.context(), x.dtype(), size4(max(x.shape()), max(y.shape())) );
}


#undef DEFINE_ELEMENT_BINARY_OPERATOR
//---------------------------------------

/*--- Math Operators----*/
//---------------------------------------
#define DEFINE_ELEMENT_UNARY_OPERATOR(OP, OPNAME) \
array_expression OPNAME (array  const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }\
\
array_expression OPNAME (array_expression const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.dtype(), x.shape()); }

DEFINE_ELEMENT_UNARY_OPERATOR((x.dtype()==FLOAT_TYPE || x.dtype()==DOUBLE_TYPE)?OPERATOR_FABS_TYPE:OPERATOR_ABS_TYPE,  abs)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ACOS_TYPE, acos)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ASIN_TYPE, asin)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ATAN_TYPE, atan)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_CEIL_TYPE, ceil)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_COS_TYPE,  cos)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_COSH_TYPE, cosh)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_EXP_TYPE,  exp)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_FLOOR_TYPE, floor)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_LOG_TYPE,  log)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_LOG10_TYPE,log10)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SIN_TYPE,  sin)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SINH_TYPE, sinh)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SQRT_TYPE, sqrt)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_TAN_TYPE,  tan)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_TANH_TYPE, tanh)
#undef DEFINE_ELEMENT_UNARY_OPERATOR
//---------------------------------------


///*--- Misc----*/
////---------------------------------------
inline operation_node_type casted(numeric_type dtype)
{
  switch(dtype)
  {
//    case BOOL_TYPE: return OPERATOR_CAST_BOOL_TYPE;
    case CHAR_TYPE: return OPERATOR_CAST_CHAR_TYPE;
    case UCHAR_TYPE: return OPERATOR_CAST_UCHAR_TYPE;
    case SHORT_TYPE: return OPERATOR_CAST_SHORT_TYPE;
    case USHORT_TYPE: return OPERATOR_CAST_USHORT_TYPE;
    case INT_TYPE: return OPERATOR_CAST_INT_TYPE;
    case UINT_TYPE: return OPERATOR_CAST_UINT_TYPE;
    case LONG_TYPE: return OPERATOR_CAST_LONG_TYPE;
    case ULONG_TYPE: return OPERATOR_CAST_ULONG_TYPE;
//    case FLOAT_TYPE: return OPERATOR_CAST_HALF_TYPE;
    case FLOAT_TYPE: return OPERATOR_CAST_FLOAT_TYPE;
    case DOUBLE_TYPE: return OPERATOR_CAST_DOUBLE_TYPE;
    default: throw unknown_datatype(dtype);
  }
}

array_expression cast(array const & x, numeric_type dtype)
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, casted(dtype)), x.context(), dtype, x.shape()); }

array_expression cast(array_expression const & x, numeric_type dtype)
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, casted(dtype)), dtype, x.shape()); }

atidlas::array_expression eye(std::size_t M, std::size_t N, atidlas::numeric_type dtype, cl::Context ctx)
{ return array_expression(value_scalar(1), value_scalar(0), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_VDIAG_TYPE), ctx, dtype, size4(M, N)); }

atidlas::array_expression zeros(std::size_t M, std::size_t N, atidlas::numeric_type dtype, cl::Context ctx)
{ return array_expression(value_scalar(0), invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), ctx, dtype, size4(M, N)); }

inline size4 flip(size4 const & shape)
{ return size4(shape._2, shape._1);}

inline size4 prod(size4 const & shape1, size4 const & shape2)
{ return size4(shape1._1*shape2._1, shape1._2*shape2._2);}

array_expression trans(array  const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.context(), x.dtype(), flip(x.shape())); }\
\
array_expression trans(array_expression const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.dtype(), flip(x.shape())); }

array_expression repmat(array const & A, int_t const & rep1, int_t const & rep2)
{
  repeat_infos infos;
  infos.rep1 = rep1;
  infos.rep2 = rep2;
  infos.sub1 = A.shape()._1;
  infos.sub2 = A.shape()._2;
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_REPEAT_TYPE), A.context(), A.dtype(), size4(infos.rep1*infos.sub1, infos.rep2*infos.sub2));
}

array_expression repmat(array_expression const & A, int_t const & rep1, int_t const & rep2)
{
  repeat_infos infos;
  infos.rep1 = rep1;
  infos.rep2 = rep2;
  infos.sub1 = A.shape()._1;
  infos.sub2 = A.shape()._2;
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_REPEAT_TYPE), A.dtype(), size4(infos.rep1*infos.sub1, infos.rep2*infos.sub2));
}

////---------------------------------------

///*--- Reductions ---*/
////---------------------------------------
#define DEFINE_REDUCTION(OP, OPNAME)\
array_expression OPNAME(array const & x, int_t axis)\
{\
  if(axis < -1 || axis > x.nshape())\
    throw std::out_of_range("The axis entry is out of bounds");\
  else if(axis==-1)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(1));\
  else if(axis==0)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._1));\
  else\
    return array_expression(x, invalid_node(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._2));\
}\
\
array_expression OPNAME(array_expression const & x, int_t axis)\
{\
  if(axis < -1 || axis > x.nshape())\
    throw std::out_of_range("The axis entry is out of bounds");\
  if(axis==-1)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), x.dtype(), size4(1));\
  else if(axis==0)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.dtype(), size4(x.shape()._1));\
  else\
    return array_expression(x, invalid_node(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.dtype(), size4(x.shape()._2));\
}

DEFINE_REDUCTION(OPERATOR_ADD_TYPE, sum)
DEFINE_REDUCTION(OPERATOR_ELEMENT_ARGMAX_TYPE, argmax)
DEFINE_REDUCTION(OPERATOR_ELEMENT_MAX_TYPE, max)
DEFINE_REDUCTION(OPERATOR_ELEMENT_MIN_TYPE, min)
DEFINE_REDUCTION(OPERATOR_ELEMENT_ARGMIN_TYPE, argmin)

#undef DEFINE_REDUCTION

namespace detail
{

  array_expression matmatprod(array const & A, array const & B)
  {
    size4 shape(A.shape()._1, B.shape()._2);
    return array_expression(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, OPERATOR_MATRIX_PRODUCT_NN_TYPE), A.context(), A.dtype(), shape);
  }

  array_expression matmatprod(array_expression const & A, array const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()._1, B.shape()._2);

    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans){
      type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
      shape._1 = A.shape()._2;
    }

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    return res;
  }

  array_expression matmatprod(array const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()._1, B.shape()._2);

    array_expression::node & B_root = const_cast<array_expression::node &>(B.tree()[B.root()]);
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(B_trans){
      type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
      shape._2 = B.shape()._1;
    }
    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  array_expression matmatprod(array_expression const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    array_expression::node & B_root = const_cast<array_expression::node &>(B.tree()[B.root()]);
    size4 shape(A.shape()._1, B.shape()._2);

    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans) shape._1 = A.shape()._2;
    if(B_trans) shape._2 = B.shape()._1;
    if(A_trans && B_trans)  type = OPERATOR_MATRIX_PRODUCT_TT_TYPE;
    else if(A_trans && !B_trans) type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
    else if(!A_trans && B_trans) type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
    else type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  template<class T>
  array_expression matvecprod(array const & A, T const & x)
  {
    int_t M = A.shape()._1;
    int_t N = A.shape()._2;
    return sum(A*repmat(reshape(x, 1, N), M, 1), 0);
  }

  template<class T>
  array_expression matvecprod(array_expression const & A, T const & x)
  {
    int_t M = A.shape()._1;
    int_t N = A.shape()._2;
    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans)
    {
      array_expression tmp(A, repmat(x, 1, M), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ELEMENT_PROD_TYPE), A.dtype(), size4(N, M));
      //Remove trans
      tmp.tree()[tmp.root()].lhs = A.tree()[A.root()].lhs;
      return sum(tmp, 1);
    }
    else
      return sum(A*repmat(reshape(x, 1, N), M, 1), 0);

  }

  array_expression matvecprod(array_expression const & A, array_expression const & x)
  {
    return matvecprod(A, array(x));
  }


}

array reshape(array const & a, int_t size1, int_t size2)
{
  array tmp(a);
  tmp.infos_.shape1 = size1;
  tmp.infos_.shape2 = size2;
  return tmp;
}

array reshape(array_expression const & a, int_t size1, int_t size2)
{
  array tmp(a);
  tmp.infos_.shape1 = size1;
  tmp.infos_.shape2 = size2;
  return tmp;
}

#define DEFINE_DOT(LTYPE, RTYPE) \
array_expression dot(LTYPE const & x, RTYPE const & y)\
{\
  if(x.nshape()==1 && y.nshape()==1)\
  {\
    return sum(x*y);\
  }\
  else if(x.nshape()==2 && y.nshape()==1)\
  {\
    return detail::matvecprod(x, y);\
  }\
  else if(x.nshape()==1 && y.nshape()==2)\
  {\
    return detail::matvecprod(trans(y), x);\
  }\
  else /*if(x.nshape()==2 && y.nshape()==2)*/\
  {\
    return detail::matmatprod(x, y);\
  }\
}

DEFINE_DOT(array, array)
DEFINE_DOT(array_expression, array)
DEFINE_DOT(array, array_expression)
DEFINE_DOT(array_expression, array_expression)

#undef DEFINE_DOT


#define DEFINE_NORM(TYPE)\
array_expression norm(TYPE const & x, unsigned int order)\
{\
  assert(order > 0 && order < 3);\
  switch(order)\
  {\
    case 1: return sum(abs(x));\
    default: return sqrt(sum(pow(x,2)));\
  }\
}

DEFINE_NORM(array)
DEFINE_NORM(array_expression)

#undef DEFINE_NORM

/*--- Copy ----*/
//---------------------------------------

//void*
void copy(void const * data, array& x, cl::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()._1)
  {
    queue.enqueueWriteBuffer(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()._1, x.shape()._2, x.dtype(), x.context());
    queue.enqueueWriteBuffer(x.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
    x = tmp;
  }
  if(blocking)
    cl_ext::synchronize(x.context());
}

void copy(array const & x, void* data, cl::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()._1)
  {
    queue.enqueueReadBuffer(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()._1, x.shape()._2, x.dtype(), x.context());
    tmp = x;
    queue.enqueueReadBuffer(tmp.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
  }
  if(blocking)
    cl_ext::synchronize(x.context());
}

void copy(void const *data, array &x, bool blocking)
{ copy(data, x, cl_ext::get_queue(x.context(), 0), blocking); }

void copy(array const & x, void* data, bool blocking)
{ copy(x, data, cl_ext::get_queue(x.context(), 0), blocking); }

//std::vector<>
template<class T>
void copy(std::vector<T> const & cx, array & x, cl::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()._1)
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy((void const*)cx.data(), x, queue, blocking);
}

template<class T>
void copy(array const & x, std::vector<T> & cx, cl::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()._1)
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy(x, (void*)cx.data(), queue, blocking);
}

template<class T>
void copy(std::vector<T> const & cx, array & x, bool blocking)
{ copy(cx, x, cl_ext::get_queue(x.context(), 0), blocking); }

template<class T>
void copy(array const & x, std::vector<T> & cx, bool blocking)
{ copy(x, cx, cl_ext::get_queue(x.context(), 0), blocking); }

#define INSTANTIATE(T) \
  template void copy<T>(std::vector<T> const &, array &, cl::CommandQueue&, bool);\
  template void copy<T>(array const &, std::vector<T> &, cl::CommandQueue&, bool);\
  template void copy<T>(std::vector<T> const &, array &, bool);\
  template void copy<T>(array const &, std::vector<T> &, bool)

INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);

#undef INSTANTIATE
/*--- Stream operators----*/
//---------------------------------------

namespace detail
{
  template<typename ItType>
  static std::ostream & prettyprint(std::ostream& os, ItType begin, ItType const & end, size_t stride = 1, bool col = false, size_t WINDOW = 10)
  {
    if(!col)
      os << "[ " ;
    size_t N = (end - begin)/stride;
    size_t upper = std::min(WINDOW,N);
    for(size_t j = 0; j < upper ; j++)
    {
      os << *begin;
      if(j<upper - 1)
        os << ",";
      begin+=stride;
    }
    if(upper < N)
    {
      if(N - upper > WINDOW)
        os << ", ... ";
      for(size_t j = std::max(N - WINDOW, upper) ; j < N ; j++)
      {
        os << "," << *begin;
        begin+=stride;
      }
    }
    if(!col)
      os << " ]" ;
    return os;
  }

}

std::ostream& operator<<(std::ostream & os, array const & a)
{
  size_t WINDOW = 10;
  numeric_type dtype = a.dtype();
  size_t M = a.shape()._1;
  size_t N = a.shape()._2;

  if(M>1 && N==1)
    std::swap(M, N);

  void* tmp = new char[M*N*size_of(dtype)];
  copy(a, (void*)tmp);

  os << "[ " ;
  size_t upper = std::min(WINDOW,M);

#define HANDLE(ADTYPE, CTYPE) case ADTYPE: detail::prettyprint(os, reinterpret_cast<CTYPE*>(tmp) + i, reinterpret_cast<CTYPE*>(tmp) + M*N + i, M, true, WINDOW); break;
  for(unsigned int i = 0 ; i < upper ; ++i)
  {
    if(i>0)
      os << "  ";
    switch(dtype)
    {
//      HANDLE(BOOL_TYPE, cl_bool)
      HANDLE(CHAR_TYPE, cl_char)
      HANDLE(UCHAR_TYPE, cl_uchar)
      HANDLE(SHORT_TYPE, cl_short)
      HANDLE(USHORT_TYPE, cl_ushort)
      HANDLE(INT_TYPE, cl_int)
      HANDLE(UINT_TYPE, cl_uint)
      HANDLE(LONG_TYPE, cl_long)
      HANDLE(ULONG_TYPE, cl_ulong)
//      HANDLE(HALF_TYPE, cl_half)
      HANDLE(FLOAT_TYPE, cl_float)
      HANDLE(DOUBLE_TYPE, cl_double)
      default: throw unknown_datatype(dtype);
    }
    if(i < upper-1)
      os <<  std::endl;
  }
  if(upper < M)
  {
    if(N - upper > WINDOW)
      os << std::endl << "  ... ";
    for(size_t i = std::max(N - WINDOW, upper) ; i < N ; i++)
    {
      os << std::endl << "  ";
      switch(dtype)
      {
//        HANDLE(BOOL_TYPE, cl_bool)
        HANDLE(CHAR_TYPE, cl_char)
        HANDLE(UCHAR_TYPE, cl_uchar)
        HANDLE(SHORT_TYPE, cl_short)
        HANDLE(USHORT_TYPE, cl_ushort)
        HANDLE(INT_TYPE, cl_int)
        HANDLE(UINT_TYPE, cl_uint)
        HANDLE(LONG_TYPE, cl_long)
        HANDLE(ULONG_TYPE, cl_ulong)
//        HANDLE(HALF_TYPE, cl_half)
        HANDLE(FLOAT_TYPE, cl_float)
        HANDLE(DOUBLE_TYPE, cl_double)
        default: throw unknown_datatype(dtype);
      }
    }
  }
  os << " ]";
  return os;
}

}
