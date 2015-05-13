#include <cassert>

#include "isaac/array.h"
#include "isaac/exception/unknown_datatype.h"
#include "isaac/model/model.h"
#include "isaac/symbolic/execute.h"

#include <stdexcept>

namespace isaac
{


/*--- Constructors ---*/
//1D Constructors

array::array(int_t shape0, numeric_type dtype, driver::Context context) :
  dtype_(dtype), shape_(shape0, 1, 1, 1), start_(0, 0, 0, 0), stride_(1, 1, 1, 1), ld_(shape_[0]),
  context_(context), data_(context_, size_of(dtype)*dsize())
{ }

template<class DT>
array::array(std::vector<DT> const & x, driver::Context context):
  dtype_(to_numeric_type<DT>::value), shape_(x.size(), 1), start_(0, 0, 0, 0), stride_(1, 1, 1, 1), ld_(shape_[0]),
  context_(context), data_(context, size_of(dtype_)*dsize())
{ *this = x; }

array::array(array & v, slice const & s0) : dtype_(v.dtype_), shape_(s0.size, 1, 1, 1), start_(v.start_[0] + v.stride_[0]*s0.start, 0, 0, 0), stride_(v.stride_[0]*s0.stride, 1, 1, 1),
                                            ld_(v.ld_), context_(v.data_.context()), data_(v.data_)
{}

#define INSTANTIATE(T) template array::array(std::vector<T> const &, driver::Context)
INSTANTIATE(char);
INSTANTIATE(unsigned char);
INSTANTIATE(short);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(unsigned int);
INSTANTIATE(long);
INSTANTIATE(unsigned long);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE

// 2D
array::array(int_t shape0, int_t shape1, numeric_type dtype, driver::Context context) : dtype_(dtype), shape_(shape0, shape1, 1, 1), start_(0, 0, 0, 0), stride_(1, 1, 1, 1), ld_(shape0),
                                                                                        context_(context), data_(context_, size_of(dtype_)*dsize())
{}

array::array(array & M, slice const & s0, slice const & s1) :  dtype_(M.dtype_), shape_(s0.size, s1.size, 1, 1),
                                                          start_(M.start_[0] + M.stride_[0]*s0.start, M.start_[1] + M.stride_[1]*s1.start, 0, 0),
                                                          stride_(M.stride_[0]*s0.stride, M.stride_[1]*s1.stride, 1, 1), ld_(M.ld_),
                                                          context_(M.data_.context()), data_(M.data_)
{ }

// 3D
array::array(int_t shape0, int_t shape1, int_t shape2, numeric_type dtype, driver::Context context) : dtype_(dtype), shape_(shape0, shape1, shape2, 1), start_(0, 0, 0, 0), stride_(1, 1, 1, 1), ld_(shape0),
                                                                                        context_(context), data_(context_, size_of(dtype_)*dsize())
{}

template<typename DT>
array::array(int_t shape0, int_t shape1, std::vector<DT> const & data, driver::Context context)
  : dtype_(to_numeric_type<DT>::value),
    shape_(shape0, shape1), start_(0, 0), stride_(1, 1), ld_(shape0),
    context_(context), data_(context_, size_of(dtype_)*dsize())
{
  isaac::copy(data, *this);
}

#define INSTANTIATE(T) template array::array(int_t, int_t, std::vector<T> const &, driver::Context)
INSTANTIATE(char);
INSTANTIATE(unsigned char);
INSTANTIATE(short);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(unsigned int);
INSTANTIATE(long);
INSTANTIATE(unsigned long);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE

// General
array::array(numeric_type dtype, driver::Buffer data, slice const & s0, slice const & s1, int_t ld, driver::Context context):
  dtype_(dtype), shape_(s0.size, s1.size), start_(s0.start, s1.start), stride_(s0.stride, s1.stride),
   ld_(ld), context_(context), data_(data)
{ }

array::array(array_expression const & proxy) : array(control(proxy)){}
array::array(array const & other) : array(control(other)){}

template<class TYPE>
array::array(controller<TYPE> const & other) :
  dtype_(other.x().dtype()),
  shape_(other.x().shape()), start_(0,0), stride_(1, 1), ld_(shape_[0]),
  context_(other.x().context()), data_(context_, size_of(dtype_)*dsize())
{
  *this = other;
}


template array::array(controller<array> const&);
template array::array(controller<array_expression> const&);

/*--- Getters ---*/
numeric_type array::dtype() const
{ return dtype_; }

size4 const & array::shape() const
{ return shape_; }

int_t array::nshape() const
{ return int_t((shape_[0] > 1) + (shape_[1] > 1)); }

size4 const & array::start() const
{ return start_; }

size4 const & array::stride() const
{ return stride_; }

int_t const & array::ld() const
{ return ld_; }

driver::Context const & array::context() const
{ return context_; }

driver::Buffer const & array::data() const
{ return data_; }

driver::Buffer & array::data()
{ return data_; }


int_t array::dsize() const
{ return ld_*shape_[1]*shape_[2]*shape_[3]; }

/*--- Assignment Operators ----*/
//---------------------------------------
array & array::operator=(array const & rhs)
{ return *this = controller<array>(rhs); }

array & array::operator=(array_expression const & rhs)
{ return *this = controller<array_expression>(rhs); }

template<class TYPE>
array& array::operator=(controller<TYPE> const & c)
{
  assert(dtype_ == c.x().dtype());
  array_expression expression(*this, c.x(), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), context_, dtype_, shape_);
  execute(controller<array_expression>(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()),
          isaac::get_model_map(driver::queues[context_][c.execution_options().queue_id]));
  return *this;
}

template<class DT>
array & array::operator=(std::vector<DT> const & rhs)
{
  assert(nshape()==1);
  isaac::copy(rhs, *this);
  return *this;
}

#define INSTANTIATE(T) template array & array::operator=<T>(std::vector<T> const &)

INSTANTIATE(char);
INSTANTIATE(unsigned char);
INSTANTIATE(short);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(unsigned int);
INSTANTIATE(long);
INSTANTIATE(unsigned long);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE

array_expression array::operator-()
{ return array_expression(*this, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array_expression array::operator!()
{ return array_expression(*this, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_NEGATE_TYPE), context_, INT_TYPE, shape_); }

//
array & array::operator+=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype_, shape_); }

array & array::operator+=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), context_, dtype_, shape_); }

array & array::operator+=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), rhs.context(), dtype_, shape_); }
//----
array & array::operator-=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array & array::operator-=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), context_, dtype_, shape_); }

array & array::operator-=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_SUB_TYPE), rhs.context(), dtype_, shape_); }
//----
array & array::operator*=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), rhs.context(), dtype_, shape_); }
//----
array & array::operator/=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), rhs.context(), dtype_, shape_); }

array_expression array::T() const
{ return isaac::trans(*this) ;}

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
void copy(driver::Context & ctx, driver::Buffer const & data, T value)
{
  driver::queues[ctx][0].write(data, CL_TRUE, 0, sizeof(T), (void*)&value);
}

}

scalar::scalar(numeric_type dtype, const driver::Buffer &data, int_t offset, driver::Context context): array(dtype, data, _(offset, offset+1), _(1,2), 1, context)
{ }

scalar::scalar(value_scalar value, driver::Context context) : array(1, value.dtype(), context)
{
  switch(dtype_)
  {
    case CHAR_TYPE: detail::copy(context_, data_, (char)value); break;
    case UCHAR_TYPE: detail::copy(context_, data_, (unsigned char)value); break;
    case SHORT_TYPE: detail::copy(context_, data_, (short)value); break;
    case USHORT_TYPE: detail::copy(context_, data_, (unsigned short)value); break;
    case INT_TYPE: detail::copy(context_, data_, (int)value); break;
    case UINT_TYPE: detail::copy(context_, data_, (unsigned int)value); break;
    case LONG_TYPE: detail::copy(context_, data_, (long)value); break;
    case ULONG_TYPE: detail::copy(context_, data_, (unsigned long)value); break;
    case FLOAT_TYPE: detail::copy(context_, data_, (float)value); break;
    case DOUBLE_TYPE: detail::copy(context_, data_, (double)value); break;
    default: throw unknown_datatype(dtype_);
  }
}


scalar::scalar(numeric_type dtype, driver::Context context) : array(1, dtype, context)
{ }

scalar::scalar(array_expression const & proxy) : array(proxy){ }

void scalar::inject(values_holder & v) const
{
    int_t dtsize = size_of(dtype_);
  #define HANDLE_CASE(DTYPE, VAL) \
  case DTYPE:\
    driver::queues[context_][0].read(data_, CL_TRUE, start_[0]*dtsize, dtsize, (void*)&v.VAL); break;\

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

template<class T>
T scalar::cast() const
{
  values_holder v;
  inject(v);

#define HANDLE_CASE(DTYPE, VAL) case DTYPE: return v.VAL

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
  driver::CommandQueue& queue = driver::queues[context_][0];
  int_t dtsize = size_of(dtype_);

#define HANDLE_CASE(TYPE, CLTYPE) case TYPE:\
                            {\
                              CLTYPE v = s;\
                              queue.write(data_, CL_TRUE, start_[0]*dtsize, dtsize, (void*)&v);\
                              return *this;\
                            }
  switch(dtype_)
  {
    HANDLE_CASE(CHAR_TYPE, char)
    HANDLE_CASE(UCHAR_TYPE, unsigned char)
    HANDLE_CASE(SHORT_TYPE, short)
    HANDLE_CASE(USHORT_TYPE, unsigned short)
    HANDLE_CASE(INT_TYPE, int)
    HANDLE_CASE(UINT_TYPE, unsigned int)
    HANDLE_CASE(LONG_TYPE, long)
    HANDLE_CASE(ULONG_TYPE, unsigned long)
    HANDLE_CASE(FLOAT_TYPE, float)
    HANDLE_CASE(DOUBLE_TYPE, double)
    default: throw unknown_datatype(dtype_);
  }
}

#define INSTANTIATE(type) scalar::operator type() const { return cast<type>(); }
  INSTANTIATE(char)
  INSTANTIATE(unsigned char)
  INSTANTIATE(short)
  INSTANTIATE(unsigned short)
  INSTANTIATE(int)
  INSTANTIATE(unsigned int)
  INSTANTIATE(long)
  INSTANTIATE(unsigned long)
  INSTANTIATE(long long)
  INSTANTIATE(unsigned long long)
  INSTANTIATE(float)
  INSTANTIATE(double)
#undef INSTANTIATE

std::ostream & operator<<(std::ostream & os, scalar const & s)
{
  switch(s.dtype())
  {
//    case BOOL_TYPE: return os << static_cast<cl_bool>(s);
    case CHAR_TYPE: return os << static_cast<char>(s);
    case UCHAR_TYPE: return os << static_cast<unsigned char>(s);
    case SHORT_TYPE: return os << static_cast<short>(s);
    case USHORT_TYPE: return os << static_cast<unsigned short>(s);
    case INT_TYPE: return os << static_cast<int>(s);
    case UINT_TYPE: return os << static_cast<unsigned int>(s);
    case LONG_TYPE: return os << static_cast<long>(s);
    case ULONG_TYPE: return os << static_cast<unsigned long>(s);
//    case HALF_TYPE: return os << static_cast<cl_half>(s);
    case FLOAT_TYPE: return os << static_cast<float>(s);
    case DOUBLE_TYPE: return os << static_cast<double>(s);
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
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, elementwise_size(x, y)); } \
 \
array_expression OPNAME (array const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, elementwise_size(x, y)); } \
\
array_expression OPNAME (array_expression const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, elementwise_size(x, y)); } \
\
array_expression OPNAME (array const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, elementwise_size(x, y)); }\
\
array_expression OPNAME (array_expression const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, x.shape()); } \
\
array_expression OPNAME (array const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, x.shape()); }\
\
array_expression OPNAME (value_scalar const & y, array_expression const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), DTYPE, x.shape()); } \
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
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }

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
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, casted(dtype)), x.context(), dtype, x.shape()); }

isaac::array_expression eye(std::size_t M, std::size_t N, isaac::numeric_type dtype, driver::Context ctx)
{ return array_expression(value_scalar(1), value_scalar(0), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_VDIAG_TYPE), ctx, dtype, size4(M, N)); }

isaac::array_expression zeros(std::size_t M, std::size_t N, isaac::numeric_type dtype, driver::Context ctx)
{ return array_expression(value_scalar(0, dtype), invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_ADD_TYPE), ctx, dtype, size4(M, N)); }

inline size4 flip(size4 const & shape)
{ return size4(shape[1], shape[0]);}

inline size4 prod(size4 const & shape1, size4 const & shape2)
{ return size4(shape1[0]*shape2[0], shape1[1]*shape2[1]);}

array_expression trans(array  const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.context(), x.dtype(), flip(x.shape())); }\
\
array_expression trans(array_expression const & x) \
{ return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.context(), x.dtype(), flip(x.shape())); }

array_expression repmat(array const & A, int_t const & rep1, int_t const & rep2)
{
  repeat_infos infos;
  infos.rep1 = rep1;
  infos.rep2 = rep2;
  infos.sub1 = A.shape()[0];
  infos.sub2 = A.shape()[1];
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_REPEAT_TYPE), A.context(), A.dtype(), size4(infos.rep1*infos.sub1, infos.rep2*infos.sub2));
}

array_expression repmat(array_expression const & A, int_t const & rep1, int_t const & rep2)
{
  repeat_infos infos;
  infos.rep1 = rep1;
  infos.rep2 = rep2;
  infos.sub1 = A.shape()[0];
  infos.sub2 = A.shape()[1];
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_REPEAT_TYPE), A.context(), A.dtype(), size4(infos.rep1*infos.sub1, infos.rep2*infos.sub2));
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
    return array_expression(x, invalid_node(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()[1]));\
  else\
    return array_expression(x, invalid_node(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()[0]));\
}\
\
array_expression OPNAME(array_expression const & x, int_t axis)\
{\
  if(axis < -1 || axis > x.nshape())\
    throw std::out_of_range("The axis entry is out of bounds");\
  if(axis==-1)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(1));\
  else if(axis==0)\
    return array_expression(x, invalid_node(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()[1]));\
  else\
    return array_expression(x, invalid_node(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()[0]));\
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
    size4 shape(A.shape()[0], B.shape()[1]);
    return array_expression(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, OPERATOR_MATRIX_PRODUCT_NN_TYPE), A.context(), A.dtype(), shape);
  }

  array_expression matmatprod(array_expression const & A, array const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()[0], B.shape()[1]);

    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans){
      type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
    }

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.context(), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    return res;
  }

  array_expression matmatprod(array const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()[0], B.shape()[1]);

    array_expression::node & B_root = const_cast<array_expression::node &>(B.tree()[B.root()]);
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(B_trans){
      type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
    }

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.context(), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  array_expression matmatprod(array_expression const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    array_expression::node & B_root = const_cast<array_expression::node &>(B.tree()[B.root()]);
    size4 shape(A.shape()[0], B.shape()[1]);

    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans) shape[0] = A.shape()[1];
    if(B_trans) shape[1] = B.shape()[0];
    if(A_trans && B_trans)  type = OPERATOR_MATRIX_PRODUCT_TT_TYPE;
    else if(A_trans && !B_trans) type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
    else if(!A_trans && B_trans) type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
    else type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), A.context(), A.dtype(), shape);
    array_expression::node & res_root = const_cast<array_expression::node &>(res.tree()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  template<class T>
  array_expression matvecprod(array const & A, T const & x)
  {
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    return sum(A*repmat(reshape(x, 1, N), M, 1), 1);
  }

  template<class T>
  array_expression matvecprod(array_expression const & A, T const & x)
  {
    int_t M = A.shape()[0];
    int_t N = A.shape()[1];
    array_expression::node & A_root = const_cast<array_expression::node &>(A.tree()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans)
    {
      array_expression tmp(A, repmat(x, 1, M), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ELEMENT_PROD_TYPE), A.context(), A.dtype(), size4(N, M));
      //Remove trans
      tmp.tree()[tmp.root()].lhs = A.tree()[A.root()].lhs;
      return sum(tmp, 0);
    }
    else
      return sum(A*repmat(reshape(x, 1, N), M, 1), 1);

  }

  array_expression matvecprod(array_expression const & A, array_expression const & x)
  {
    return matvecprod(A, array(x));
  }


}

array_expression reshape(array const & x, int_t shape0, int_t shape1)
{  return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_RESHAPE_TYPE), x.context(), x.dtype(), size4(shape0, shape1)); }

array_expression reshape(array_expression const & x, int_t shape0, int_t shape1)
{  return array_expression(x, invalid_node(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_RESHAPE_TYPE), x.context(), x.dtype(), size4(shape0, shape1)); }


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
void copy(void const * data, array& x, driver::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()[0])
  {
    queue.write(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()[0], x.shape()[1], x.dtype(), x.context());
    queue.write(x.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
    x = tmp;
  }
  if(blocking)
    driver::synchronize(x.context());
}

void copy(array const & x, void* data, driver::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()[0])
  {
    queue.read(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()[0], x.shape()[1], x.dtype(), x.context());
    tmp = x;
    queue.read(tmp.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
  }
  if(blocking)
    driver::synchronize(x.context());
}

void copy(void const *data, array &x, bool blocking)
{ copy(data, x, driver::queues[x.context()][0], blocking); }

void copy(array const & x, void* data, bool blocking)
{ copy(x, data, driver::queues[x.context()][0], blocking); }

//std::vector<>
template<class T>
void copy(std::vector<T> const & cx, array & x, driver::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()[0])
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy((void const*)cx.data(), x, queue, blocking);
}

template<class T>
void copy(array const & x, std::vector<T> & cx, driver::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()[0])
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy(x, (void*)cx.data(), queue, blocking);
}

template<class T>
void copy(std::vector<T> const & cx, array & x, bool blocking)
{ copy(cx, x, driver::queues[x.context()][0], blocking); }

template<class T>
void copy(array const & x, std::vector<T> & cx, bool blocking)
{ copy(x, cx, driver::queues[x.context()][0], blocking); }

#define INSTANTIATE(T) \
  template void copy<T>(std::vector<T> const &, array &, driver::CommandQueue&, bool);\
  template void copy<T>(array const &, std::vector<T> &, driver::CommandQueue&, bool);\
  template void copy<T>(std::vector<T> const &, array &, bool);\
  template void copy<T>(array const &, std::vector<T> &, bool)

INSTANTIATE(char);
INSTANTIATE(unsigned char);
INSTANTIATE(short);
INSTANTIATE(unsigned short);
INSTANTIATE(int);
INSTANTIATE(unsigned int);
INSTANTIATE(long);
INSTANTIATE(unsigned long);
INSTANTIATE(long long);
INSTANTIATE(unsigned long long);
INSTANTIATE(float);
INSTANTIATE(double);

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
  size_t M = a.shape()[0];
  size_t N = a.shape()[1];

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
      HANDLE(CHAR_TYPE, char)
      HANDLE(UCHAR_TYPE, unsigned char)
      HANDLE(SHORT_TYPE, short)
      HANDLE(USHORT_TYPE, unsigned short)
      HANDLE(INT_TYPE, int)
      HANDLE(UINT_TYPE, unsigned int)
      HANDLE(LONG_TYPE, long)
      HANDLE(ULONG_TYPE, unsigned long)
//      HANDLE(HALF_TYPE, cl_half)
      HANDLE(FLOAT_TYPE, float)
      HANDLE(DOUBLE_TYPE, double)
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
        HANDLE(CHAR_TYPE, char)
        HANDLE(UCHAR_TYPE, unsigned char)
        HANDLE(SHORT_TYPE, short)
        HANDLE(USHORT_TYPE, unsigned short)
        HANDLE(INT_TYPE, int)
        HANDLE(UINT_TYPE, unsigned int)
        HANDLE(LONG_TYPE, long)
        HANDLE(ULONG_TYPE, unsigned long)
//        HANDLE(HALF_TYPE, cl_half)
        HANDLE(FLOAT_TYPE, float)
        HANDLE(DOUBLE_TYPE, double)
        default: throw unknown_datatype(dtype);
      }
    }
  }
  os << " ]";
  return os;
}

}
