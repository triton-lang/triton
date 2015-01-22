#include <cassert>

#include "atidlas/array.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/exception/unknown_datatype.h"
#include "atidlas/model/model.h"
#include "atidlas/symbolic/execute.h"

namespace atidlas
{


/*--- Constructors ---*/

//1D Constructors

array::array(int_t size1, numeric_type dtype, cl::Context context) :
  dtype_(dtype), shape_(size1, 1), start_(0, 0), stride_(1, 1), ld_(shape_._1),
  context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype)*dsize())
{ }

template<class DT>
array::array(std::vector<DT> const & x, cl::Context context):
  dtype_(to_numeric_type<DT>::value), shape_(x.size(), 1), start_(0, 0), stride_(1, 1), ld_(shape_._1),
  context_(context), data_(context, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{ *this = x; }

array::array(array & v, slice const & s1) : dtype_(v.dtype_), shape_(s1.size, 1), start_(v.start_._1 + v.stride_._1*s1.start, 0), stride_(v.stride_._1*s1.stride, 1),
                                            ld_(v.ld_), context_(v.data_.getInfo<CL_MEM_CONTEXT>()), data_(v.data_)
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
array::array(int_t size1, int_t size2, numeric_type dtype, cl::Context context) : dtype_(dtype), shape_(size1, size2), start_(0, 0), stride_(1, 1), ld_(size1),
                                                                              context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{}

array::array(array & M, slice const & s1, slice const & s2) :  dtype_(M.dtype_), shape_(s1.size, s2.size),
                                                          start_(M.start_._1 + M.stride_._1*s1.start, M.start_._2 + M.stride_._2*s2.start),
                                                          stride_(M.stride_._1*s1.stride, M.stride_._2*s2.stride), ld_(M.ld_),
                                                          context_(M.data_.getInfo<CL_MEM_CONTEXT>()), data_(M.data_)
{ }

template<typename DT>
array::array(int_t size1, int_t size2, std::vector<DT> const & data, cl::Context context)
  : dtype_(to_numeric_type<DT>::value),
    shape_(size1, size2), start_(0, 0), stride_(1, 1), ld_(size1),
    context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
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
  dtype_(dtype), shape_(s1.size, s2.size), start_(s1.start, s2.start), stride_(s1.stride, s2.stride),
   ld_(ld), context_(context), data_(data)
{ }

array::array(array_expression const & proxy) :
  dtype_(proxy.dtype()),
  shape_(proxy.shape()), start_(0,0), stride_(1, 1), ld_(shape_._1),
  context_(proxy.context()), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{
  *this = proxy;
}

array::array(array const & other) :
  dtype_(other.dtype()),
  shape_(other.shape()), start_(0,0), stride_(1, 1), ld_(shape_._1),
  context_(other.context()), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{
  *this = other;
}


/*--- Getters ---*/
numeric_type array::dtype() const
{ return dtype_; }

size4 array::shape() const
{ return shape_; }

int_t array::nshape() const
{ return int_t((shape_._1 > 1) + (shape_._2 > 1)); }

size4 array::start() const
{ return start_; }

size4 array::stride() const
{ return stride_; }

int_t array::ld() const
{ return ld_; }

cl::Context const & array::context() const
{ return context_; }

cl::Buffer const & array::data() const
{ return data_; }

int_t array::dsize() const
{ return ld_*shape_._2; }

/*--- Assignment Operators ----*/
//---------------------------------------
array & array::operator=(array const & rhs)
{
  array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), context_, dtype_, shape_);
  cl::CommandQueue & queue = cl::queues[context_].front();
  model_map_t & mmap = atidlas::get_model_map(queue);
  execute(expression, mmap);
  return *this;
}

array & array::operator=(array_expression const & rhs)
{
  array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), shape_);
  cl::CommandQueue & queue = cl::queues[context_].front();
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
{ return array_expression(*this, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

//
array & array::operator+=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype_, shape_); }

array & array::operator+=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype_, shape_); }

array & array::operator+=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), shape_); }
//----
array & array::operator-=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array & array::operator-=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array & array::operator-=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), shape_); }
//----
array & array::operator*=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), shape_); }
//----
array & array::operator/=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), shape_); }

array_expression array::T() const
{ return atidlas::trans(*this) ;}

/*--- Indexing operators -----*/
//---------------------------------------
scalar array::operator [](int_t idx)
{
  assert(nshape()==1);
  return scalar(dtype_, data_, idx, context_);
}

const scalar array::operator [](int_t idx) const
{
  assert(nshape()==1);
  return scalar(dtype_, data_, idx, context_);
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
  cl::queues[ctx].front().enqueueWriteBuffer(data, CL_TRUE, 0, sizeof(T), (void*)&value);
}

}

scalar::scalar(numeric_type dtype, const cl::Buffer &data, int_t offset, cl::Context context): array(dtype, data, _(offset, offset+1), _(1,2), 1, context)
{ }

scalar::scalar(value_scalar value, cl::Context context) : array(1, value.dtype(), context)
{
  switch(dtype_)
  {
    case CHAR_TYPE: detail::copy(context_, data_, (cl_char)value); break;
    case UCHAR_TYPE: detail::copy(context_, data_, (cl_uchar)value); break;
    case SHORT_TYPE: detail::copy(context_, data_, (cl_short)value); break;
    case USHORT_TYPE: detail::copy(context_, data_, (cl_ushort)value); break;
    case INT_TYPE: detail::copy(context_, data_, (cl_int)value); break;
    case UINT_TYPE: detail::copy(context_, data_, (cl_uint)value); break;
    case LONG_TYPE: detail::copy(context_, data_, (cl_long)value); break;
    case ULONG_TYPE: detail::copy(context_, data_, (cl_ulong)value); break;
    case FLOAT_TYPE: detail::copy(context_, data_, (cl_float)value); break;
    case DOUBLE_TYPE: detail::copy(context_, data_, (cl_double)value); break;
    default: throw unknown_datatype(dtype_);
  }
}


scalar::scalar(numeric_type dtype, cl::Context context) : array(1, dtype, context)
{ }

scalar::scalar(array_expression const & proxy) : array(proxy){ }

template<class T>
T scalar::cast() const
{
  values_holder v;
  int_t dtsize = size_of(dtype_);
#define HANDLE_CASE(DTYPE, VAL) \
case DTYPE:\
  cl::queues[context_].front().enqueueReadBuffer(data_, CL_TRUE, start_._1*dtsize, dtsize, (void*)&v.VAL);\
  return v.VAL

  switch(dtype_)
  {
    HANDLE_CASE(CHAR_TYPE, int8);
    HANDLE_CASE(UCHAR_TYPE, uint8);
    HANDLE_CASE(SHORT_TYPE, int16);
    HANDLE_CASE(USHORT_TYPE, uint16);
    HANDLE_CASE(INT_TYPE, int32);
    HANDLE_CASE(UINT_TYPE, uint32);
    HANDLE_CASE(LONG_TYPE, int64);
    HANDLE_CASE(ULONG_TYPE, uint64);
    HANDLE_CASE(FLOAT_TYPE, float32);
    HANDLE_CASE(DOUBLE_TYPE, float64);
    default: throw unknown_datatype(dtype_);
  }
#undef HANDLE_CASE

}

scalar& scalar::operator=(value_scalar const & s)
{
  cl::CommandQueue& queue = cl::queues[context_].front();
  int_t dtsize = size_of(dtype_);

#define HANDLE_CASE(TYPE, CLTYPE) case TYPE:\
                            {\
                              CLTYPE v = s;\
                              queue.enqueueWriteBuffer(data_, CL_TRUE, start_._1*dtsize, dtsize, (void*)&v);\
                              return *this;\
                            }
  switch(dtype_)
  {
    HANDLE_CASE(CHAR_TYPE, cl_char)
    HANDLE_CASE(UCHAR_TYPE, cl_uchar)
    HANDLE_CASE(SHORT_TYPE, cl_short)
    HANDLE_CASE(USHORT_TYPE, cl_ushort)
    HANDLE_CASE(INT_TYPE, cl_int)
    HANDLE_CASE(UINT_TYPE, cl_uint)
    HANDLE_CASE(LONG_TYPE, cl_long)
    HANDLE_CASE(ULONG_TYPE, cl_ulong)
    HANDLE_CASE(FLOAT_TYPE, cl_float)
    HANDLE_CASE(DOUBLE_TYPE, cl_double)
    default: throw unknown_datatype(dtype_);
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
    case CHAR_TYPE: return os << static_cast<cl_char>(s);
    case UCHAR_TYPE: return os << static_cast<cl_uchar>(s);
    case SHORT_TYPE: return os << static_cast<cl_short>(s);
    case USHORT_TYPE: return os << static_cast<cl_ushort>(s);
    case INT_TYPE: return os << static_cast<cl_int>(s);
    case UINT_TYPE: return os << static_cast<cl_uint>(s);
    case LONG_TYPE: return os << static_cast<cl_long>(s);
    case ULONG_TYPE: return os << static_cast<cl_ulong>(s);
    case HALF_TYPE: return os << static_cast<cl_half>(s);
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


#define DEFINE_ELEMENT_BINARY_OPERATOR(OP, OPNAME) \
array_expression OPNAME (array_expression const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), elementwise_size(x, y) ); } \
 \
array_expression OPNAME (array const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), elementwise_size(x, y)); } \
\
array_expression OPNAME (array_expression const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), elementwise_size(x, y)); } \
\
array_expression OPNAME (array const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), elementwise_size(x, y)); }\
\
array_expression OPNAME (array_expression const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.shape()); } \
\
array_expression OPNAME (array const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }\
\
array_expression OPNAME (value_scalar const & y, array_expression const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.shape()); } \
\
array_expression OPNAME (value_scalar const & y, array const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ADD_TYPE, operator +)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_SUB_TYPE, operator -)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_MULT_TYPE, operator *)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_DIV_TYPE, operator /)

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GREATER_TYPE, operator >)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GEQ_TYPE, operator >=)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LESS_TYPE, operator <)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LEQ_TYPE, operator <=)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_EQ_TYPE, operator ==)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_NEQ_TYPE, operator !=)

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MAX_TYPE, maximum)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MIN_TYPE, minimum)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_POW_TYPE, pow)

array_expression outer(array const & x, array const & y)
{
  assert(x.nshape()==1 && y.nshape()==1);
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_OUTER_PROD_TYPE), x.context(), x.dtype(), size4(max(x.shape()), max(y.shape())) );
}

namespace detail
{
  DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ASSIGN_TYPE, assign)
}

#undef DEFINE_ELEMENT_BINARY_OPERATOR
//---------------------------------------

/*--- Math Operators----*/
//---------------------------------------
#define DEFINE_ELEMENT_UNARY_OPERATOR(OP, OPNAME) \
array_expression OPNAME (array  const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }\
\
array_expression OPNAME (array_expression const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.shape()); }

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
atidlas::array_expression eye(std::size_t M, std::size_t N, atidlas::numeric_type dtype, cl::Context ctx)
{
  return array_expression(value_scalar(1), value_scalar(0), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_VDIAG_TYPE), ctx, dtype, size4(M, N));
}

atidlas::array_expression zeros(std::size_t M, std::size_t N, atidlas::numeric_type dtype, cl::Context ctx)
{
  return array_expression(value_scalar(0), lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), ctx, dtype, size4(M, N));
}

inline size4 flip(size4 const & shape)
{ return size4(shape._2, shape._1);}

inline size4 prod(size4 const & shape1, size4 const & shape2)
{ return size4(shape1._1*shape2._1, shape1._2*shape2._2);}

array_expression trans(array  const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.context(), x.dtype(), flip(x.shape())); }\
\
array_expression trans(array_expression const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), flip(x.shape())); }

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
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_REPEAT_TYPE), size4(infos.rep1*infos.sub1, infos.rep2*infos.sub2));
}

////---------------------------------------

///*--- Reductions ---*/
////---------------------------------------
#define DEFINE_REDUCTION(OP, OPNAME)\
array_expression OPNAME(array const & x, int_t axis)\
{\
  if(axis==-1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(1));\
  else if(axis==0)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._1));\
  else if(axis==1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._2));\
  else\
    throw ;\
}\
\
array_expression OPNAME(array_expression const & x, int_t axis)\
{\
  if(axis==-1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), size4(1));\
  else if(axis==0)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), size4(x.shape()._1));\
  else if(axis==1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), size4(x.shape()._2));\
  else\
    throw ;\
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

    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans){
      type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
      shape._1 = A.shape()._2;
    }

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.tree()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    return res;
  }

  array_expression matmatprod(array const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()._1, B.shape()._2);

    symbolic_expression_node & B_root = const_cast<symbolic_expression_node &>(B.tree()[B.root()]);
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(B_trans){
      type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
      shape._2 = B.shape()._1;
    }
    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.tree()[res.root()]);
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  array_expression matmatprod(array_expression const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.tree()[A.root()]);
    symbolic_expression_node & B_root = const_cast<symbolic_expression_node &>(B.tree()[B.root()]);
    size4 shape(A.shape()._1, B.shape()._2);

    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans) shape._1 = A.shape()._2;
    if(B_trans) shape._2 = B.shape()._1;
    if(A_trans && B_trans)  type = OPERATOR_MATRIX_PRODUCT_TT_TYPE;
    else if(A_trans && !B_trans) type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
    else if(!A_trans && B_trans) type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
    else type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.tree()[res.root()]);
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
    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans)
    {
      array_expression tmp(A, repmat(x, 1, M), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ELEMENT_PROD_TYPE), size4(N, M));
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
  tmp.shape_._1 = size1;
  tmp.shape_._2 = size2;
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
    cl::synchronize(x.context());
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
    cl::synchronize(x.context());
}

void copy(void const *data, array &x, bool blocking)
{ copy(data, x, cl::queues[x.context()].front(), blocking); }

void copy(array const & x, void* data, bool blocking)
{ copy(x, data, cl::queues[x.context()].front(), blocking); }

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
{ copy(cx, x, cl::queues[x.context()].front(), blocking); }

template<class T>
void copy(array const & x, std::vector<T> & cx, bool blocking)
{ copy(x, cx, cl::queues[x.context()].front(), blocking); }

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
      HANDLE(CHAR_TYPE, cl_char)
      HANDLE(UCHAR_TYPE, cl_uchar)
      HANDLE(SHORT_TYPE, cl_short)
      HANDLE(USHORT_TYPE, cl_ushort)
      HANDLE(INT_TYPE, cl_int)
      HANDLE(UINT_TYPE, cl_uint)
      HANDLE(LONG_TYPE, cl_long)
      HANDLE(ULONG_TYPE, cl_ulong)
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
        HANDLE(CHAR_TYPE, cl_char)
        HANDLE(UCHAR_TYPE, cl_uchar)
        HANDLE(SHORT_TYPE, cl_short)
        HANDLE(USHORT_TYPE, cl_ushort)
        HANDLE(INT_TYPE, cl_int)
        HANDLE(UINT_TYPE, cl_uint)
        HANDLE(LONG_TYPE, cl_long)
        HANDLE(ULONG_TYPE, cl_ulong)
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
