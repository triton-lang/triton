#ifndef ATIDLAS_UTILS_HPP
#define ATIDLAS_UTILS_HPP


#include <sstream>

#include "viennacl/matrix_def.hpp"
#include "viennacl/vector_def.hpp"

#include "viennacl/ocl/forwards.h"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/traits/size.hpp"
#include "viennacl/traits/handle.hpp"

#include "atidlas/tools/to_string.hpp"
#include "atidlas/forwards.h"

namespace atidlas
{
namespace tools
{

template <typename T>
class make_vector {
public:
  typedef make_vector<T> my_type;
  my_type& operator<< (const T& val) {
    data_.push_back(val);
    return *this;
  }
  operator std::vector<T>() const {
    return data_;
  }
private:
  std::vector<T> data_;
};

//CUDA Conversion
inline std::string opencl_source_to_cuda_source(std::string const & opencl_src)
{
  std::string res = opencl_src;

  viennacl::tools::find_and_replace(res,"__attribute__","//__attribute__");

  //Pointer
  viennacl::tools::find_and_replace(res, "__global float*", "float*");
  viennacl::tools::find_and_replace(res, "__local float*", "float*");

  viennacl::tools::find_and_replace(res, "__global double*", "double*");
  viennacl::tools::find_and_replace(res, "__local double*", "double*");

  //Qualifiers
  viennacl::tools::find_and_replace(res,"__global","__device__");
  viennacl::tools::find_and_replace(res,"__kernel","__global__");
  viennacl::tools::find_and_replace(res,"__constant","__constant__");
  viennacl::tools::find_and_replace(res,"__local","__shared__");

  //Indexing
  viennacl::tools::find_and_replace(res,"get_num_groups(0)","gridDim.x");
  viennacl::tools::find_and_replace(res,"get_num_groups(1)","gridDim.y");

  viennacl::tools::find_and_replace(res,"get_local_size(0)","blockDim.x");
  viennacl::tools::find_and_replace(res,"get_local_size(1)","blockDim.y");

  viennacl::tools::find_and_replace(res,"get_group_id(0)","blockIdx.x");
  viennacl::tools::find_and_replace(res,"get_group_id(1)","blockIdx.y");

  viennacl::tools::find_and_replace(res,"get_local_id(0)","threadIdx.x");
  viennacl::tools::find_and_replace(res,"get_local_id(1)","threadIdx.y");

  viennacl::tools::find_and_replace(res,"get_global_id(0)","(blockIdx.x*blockDim.x + threadIdx.x)");
  viennacl::tools::find_and_replace(res,"get_global_id(1)","(blockIdx.y*blockDim.y + threadIdx.y)");

  //Synchronization
  viennacl::tools::find_and_replace(res,"barrier(CLK_LOCAL_MEM_FENCE)","__syncthreads()");
  viennacl::tools::find_and_replace(res,"barrier(CLK_GLOBAL_MEM_FENCE)","__syncthreads()");


  return res;
}

static std::string numeric_type_to_string(viennacl::scheduler::statement_node_numeric_type const & type){
  switch (type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return "char";
  //case viennacl::scheduler::UCHAR_TYPE: return "unsigned char";
  //case viennacl::scheduler::SHORT_TYPE: return "short";
  //case viennacl::scheduler::USHORT_TYPE: return "unsigned short";
  case viennacl::scheduler::INT_TYPE:  return "int";
  case viennacl::scheduler::UINT_TYPE: return "unsigned int";
  case viennacl::scheduler::LONG_TYPE:  return "long";
  case viennacl::scheduler::ULONG_TYPE: return "unsigned long";
  case viennacl::scheduler::FLOAT_TYPE : return "float";
  case viennacl::scheduler::DOUBLE_TYPE : return "double";
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}


template<class Fun>
static typename Fun::result_type call_on_host_scalar(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == viennacl::scheduler::SCALAR_TYPE_FAMILY && bool("Must be called on a host scalar"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(element.host_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(element.host_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(element.host_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(element.host_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(element.host_int);
  case viennacl::scheduler::UINT_TYPE: return fun(element.host_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(element.host_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(element.host_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(element.host_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(element.host_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_scalar(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == viennacl::scheduler::SCALAR_TYPE_FAMILY && bool("Must be called on a scalar"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(*element.scalar_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(*element.scalar_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(*element.scalar_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(*element.scalar_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(*element.scalar_int);
  case viennacl::scheduler::UINT_TYPE: return fun(*element.scalar_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(*element.scalar_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(*element.scalar_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(*element.scalar_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(*element.scalar_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_vector(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == viennacl::scheduler::VECTOR_TYPE_FAMILY && bool("Must be called on a vector"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(*element.vector_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(*element.vector_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(*element.vector_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(*element.vector_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(*element.vector_int);
  case viennacl::scheduler::UINT_TYPE: return fun(*element.vector_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(*element.vector_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(*element.vector_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(*element.vector_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(*element.vector_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_implicit_vector(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == viennacl::scheduler::VECTOR_TYPE_FAMILY   && bool("Must be called on a implicit_vector"));
  assert(element.subtype     == viennacl::scheduler::IMPLICIT_VECTOR_TYPE && bool("Must be called on a implicit_vector"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(*element.implicit_vector_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(*element.implicit_vector_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(*element.implicit_vector_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(*element.implicit_vector_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(*element.implicit_vector_int);
  case viennacl::scheduler::UINT_TYPE: return fun(*element.implicit_vector_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(*element.implicit_vector_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(*element.implicit_vector_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(*element.implicit_vector_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(*element.implicit_vector_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_matrix(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == viennacl::scheduler::MATRIX_TYPE_FAMILY && bool("Must be called on a matrix"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(*element.matrix_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(*element.matrix_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(*element.matrix_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(*element.matrix_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(*element.matrix_int);
  case viennacl::scheduler::UINT_TYPE: return fun(*element.matrix_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(*element.matrix_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(*element.matrix_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(*element.matrix_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(*element.matrix_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}


template<class Fun>
static typename Fun::result_type call_on_implicit_matrix(viennacl::scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.subtype     == viennacl::scheduler::IMPLICIT_MATRIX_TYPE && bool("Must be called on a implicit matrix"));
  switch (element.numeric_type)
  {
  //case viennacl::scheduler::CHAR_TYPE: return fun(*element.implicit_matrix_char);
  //case viennacl::scheduler::UCHAR_TYPE: return fun(*element.implicit_matrix_uchar);
  //case viennacl::scheduler::SHORT_TYPE: return fun(*element.implicit_matrix_short);
  //case viennacl::scheduler::USHORT_TYPE: return fun(*element.implicit_matrix_ushort);
  case viennacl::scheduler::INT_TYPE:  return fun(*element.implicit_matrix_int);
  case viennacl::scheduler::UINT_TYPE: return fun(*element.implicit_matrix_uint);
  case viennacl::scheduler::LONG_TYPE:  return fun(*element.implicit_matrix_long);
  case viennacl::scheduler::ULONG_TYPE: return fun(*element.implicit_matrix_ulong);
  case viennacl::scheduler::FLOAT_TYPE : return fun(*element.implicit_matrix_float);
  case viennacl::scheduler::DOUBLE_TYPE : return fun(*element.implicit_matrix_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_element(viennacl::scheduler::lhs_rhs_element const & element, Fun const & fun){
  switch (element.type_family)
  {
  case viennacl::scheduler::SCALAR_TYPE_FAMILY:
    if (element.subtype == viennacl::scheduler::HOST_SCALAR_TYPE)
      return call_on_host_scalar(element, fun);
    else
      return call_on_scalar(element, fun);
  case viennacl::scheduler::VECTOR_TYPE_FAMILY :
    if (element.subtype == viennacl::scheduler::IMPLICIT_VECTOR_TYPE)
      return call_on_implicit_vector(element, fun);
    else
      return call_on_vector(element, fun);
  case viennacl::scheduler::MATRIX_TYPE_FAMILY:
    if (element.subtype == viennacl::scheduler::IMPLICIT_MATRIX_TYPE)
      return call_on_implicit_matrix(element, fun);
    else
      return call_on_matrix(element,fun);
  default:
    throw generator_not_supported_exception("Unsupported datastructure type : Not among {Scalar, Vector, Matrix}");
  }
}

struct scalartype_size_fun
{
  typedef atidlas_int_t result_type;
  result_type operator()(float const &) const { return sizeof(float); }
  result_type operator()(double const &) const { return sizeof(double); }
  template<class T> result_type operator()(T const &) const { return sizeof(typename viennacl::result_of::cpu_value_type<T>::type); }
};

struct internal_size_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::internal_size(t); }
};

struct size_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::size(t); }
};

struct start_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start(t); }
};


struct stride_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride(t); }
};

struct start1_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start1(t); }
};

struct start2_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start2(t); }
};

struct leading_stride_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride1(t); }
};

struct leading_start_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start1(t); }
};

struct stride1_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride1(t); }
};

struct stride2_fun
{
  typedef atidlas_int_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride2(t); }
};

struct handle_fun
{
  typedef cl_mem result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::opencl_handle(t); }
};

struct internal_size1_fun
{
  typedef atidlas_int_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::internal_size1(t); }
};

struct internal_size2_fun
{
  typedef atidlas_int_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::internal_size2(t); }
};

struct size1_fun
{
  typedef atidlas_int_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::size1(t); }
};

struct size2_fun
{
  typedef atidlas_int_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::size2(t); }
};

template<class T, class U>
struct is_same_type { enum { value = 0 }; };

template<class T>
struct is_same_type<T,T> { enum { value = 1 }; };

inline bool is_reduction(viennacl::scheduler::statement_node const & node)
{
  return node.op.type_family==viennacl::scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
      || node.op.type_family==viennacl::scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY
      || node.op.type_family==viennacl::scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY
      || node.op.type==viennacl::scheduler::OPERATION_BINARY_INNER_PROD_TYPE
      || node.op.type==viennacl::scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE;
}

inline bool is_index_reduction(viennacl::scheduler::op_element const & op)
{
  return op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE
      || op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE
      || op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE
      || op.type==viennacl::scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE;
}
template<class T>
struct type_to_string;
template<> struct type_to_string<unsigned char> { static const char * value() { return "unsigned char"; } };
template<> struct type_to_string<char> { static const char * value() { return "char"; } };
template<> struct type_to_string<unsigned short> { static const char * value() { return "unsigned short"; } };
template<> struct type_to_string<short> { static const char * value() { return "short"; } };
template<> struct type_to_string<unsigned int> { static const char * value() { return "unsigned int"; } };
template<> struct type_to_string<int> { static const char * value() { return "int"; } };
template<> struct type_to_string<unsigned long> { static const char * value() { return "unsigned long"; } };
template<> struct type_to_string<long> { static const char * value() { return "long"; } };
template<> struct type_to_string<float> { static const char * value() { return "float"; } };
template<> struct type_to_string<double> { static const char * value() { return "double"; } };


template<class T>
struct first_letter_of_type;
template<> struct first_letter_of_type<char> { static char value() { return 'c'; } };
template<> struct first_letter_of_type<unsigned char> { static char value() { return 'd'; } };
template<> struct first_letter_of_type<short> { static char value() { return 's'; } };
template<> struct first_letter_of_type<unsigned short> { static char value() { return 't'; } };
template<> struct first_letter_of_type<int> { static char value() { return 'i'; } };
template<> struct first_letter_of_type<unsigned int> { static char value() { return 'j'; } };
template<> struct first_letter_of_type<long> { static char value() { return 'l'; } };
template<> struct first_letter_of_type<unsigned long> { static char value() { return 'm'; } };
template<> struct first_letter_of_type<float> { static char value() { return 'f'; } };
template<> struct first_letter_of_type<double> { static char value() { return 'd'; } };

class kernel_generation_stream : public std::ostream
{
  class kgenstream : public std::stringbuf
  {
  public:
    kgenstream(std::ostringstream& oss,unsigned int const & tab_count) : oss_(oss), tab_count_(tab_count){ }
    int sync() {
      for (unsigned int i=0; i<tab_count_;++i)
        oss_ << "    ";
      oss_ << str();
      str("");
      return !oss_;
    }
    ~kgenstream() {  pubsync(); }
  private:
    std::ostream& oss_;
    unsigned int const & tab_count_;
  };

public:
  kernel_generation_stream() : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0){ }
  ~kernel_generation_stream(){ delete rdbuf(); }

  std::string str(){ return oss.str(); }
  void inc_tab(){ ++tab_count_; }
  void dec_tab(){ --tab_count_; }
private:
  unsigned int tab_count_;
  std::ostringstream oss;
};

inline bool node_leaf(viennacl::scheduler::op_element const & op)
{
  using namespace viennacl::scheduler;
  return op.type==OPERATION_UNARY_NORM_1_TYPE
      || op.type==OPERATION_UNARY_NORM_2_TYPE
      || op.type==OPERATION_UNARY_NORM_INF_TYPE
      || op.type==OPERATION_UNARY_TRANS_TYPE
      || op.type==OPERATION_BINARY_MAT_VEC_PROD_TYPE
      || op.type==OPERATION_BINARY_MAT_MAT_PROD_TYPE
      || op.type==OPERATION_BINARY_INNER_PROD_TYPE
      || op.type==OPERATION_BINARY_MATRIX_DIAG_TYPE
      || op.type==OPERATION_BINARY_VECTOR_DIAG_TYPE
      || op.type==OPERATION_BINARY_MATRIX_ROW_TYPE
      || op.type==OPERATION_BINARY_MATRIX_COLUMN_TYPE
      || op.type_family==OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
      || op.type_family==OPERATION_ROWS_REDUCTION_TYPE_FAMILY
      || op.type_family==OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY;
}

inline bool elementwise_operator(viennacl::scheduler::op_element const & op)
{
  using namespace viennacl::scheduler;
  return op.type== OPERATION_BINARY_ASSIGN_TYPE
      || op.type== OPERATION_BINARY_INPLACE_ADD_TYPE
      || op.type== OPERATION_BINARY_INPLACE_SUB_TYPE
      || op.type== OPERATION_BINARY_ADD_TYPE
      || op.type== OPERATION_BINARY_SUB_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_PROD_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_DIV_TYPE
      || op.type== OPERATION_BINARY_MULT_TYPE
      || op.type== OPERATION_BINARY_DIV_TYPE;
}

inline bool elementwise_function(viennacl::scheduler::op_element const & op)
{
  using namespace viennacl::scheduler;
  return

      op.type == OPERATION_UNARY_CAST_CHAR_TYPE
      || op.type == OPERATION_UNARY_CAST_UCHAR_TYPE
      || op.type == OPERATION_UNARY_CAST_SHORT_TYPE
      || op.type == OPERATION_UNARY_CAST_USHORT_TYPE
      || op.type == OPERATION_UNARY_CAST_INT_TYPE
      || op.type == OPERATION_UNARY_CAST_UINT_TYPE
      || op.type == OPERATION_UNARY_CAST_LONG_TYPE
      || op.type == OPERATION_UNARY_CAST_ULONG_TYPE
      || op.type == OPERATION_UNARY_CAST_HALF_TYPE
      || op.type == OPERATION_UNARY_CAST_FLOAT_TYPE
      || op.type == OPERATION_UNARY_CAST_DOUBLE_TYPE

      || op.type== OPERATION_UNARY_ABS_TYPE
      || op.type== OPERATION_UNARY_ACOS_TYPE
      || op.type== OPERATION_UNARY_ASIN_TYPE
      || op.type== OPERATION_UNARY_ATAN_TYPE
      || op.type== OPERATION_UNARY_CEIL_TYPE
      || op.type== OPERATION_UNARY_COS_TYPE
      || op.type== OPERATION_UNARY_COSH_TYPE
      || op.type== OPERATION_UNARY_EXP_TYPE
      || op.type== OPERATION_UNARY_FABS_TYPE
      || op.type== OPERATION_UNARY_FLOOR_TYPE
      || op.type== OPERATION_UNARY_LOG_TYPE
      || op.type== OPERATION_UNARY_LOG10_TYPE
      || op.type== OPERATION_UNARY_SIN_TYPE
      || op.type== OPERATION_UNARY_SINH_TYPE
      || op.type== OPERATION_UNARY_SQRT_TYPE
      || op.type== OPERATION_UNARY_TAN_TYPE
      || op.type== OPERATION_UNARY_TANH_TYPE

      || op.type== OPERATION_BINARY_ELEMENT_POW_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_EQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_NEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_GREATER_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_LESS_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_GEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_LEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_FMAX_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_FMIN_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_MAX_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_MIN_TYPE;

}

inline viennacl::scheduler::lhs_rhs_element & lhs_rhs_element(viennacl::scheduler::statement const & st, atidlas_int_t idx, leaf_t leaf)
{
  using namespace tools;
  assert(leaf==LHS_NODE_TYPE || leaf==RHS_NODE_TYPE);
  if (leaf==LHS_NODE_TYPE)
    return const_cast<viennacl::scheduler::lhs_rhs_element &>(st.array()[idx].lhs);
  return const_cast<viennacl::scheduler::lhs_rhs_element &>(st.array()[idx].rhs);
}

inline unsigned int size_of(viennacl::scheduler::statement_node_numeric_type type)
{
  using namespace viennacl::scheduler;
  switch (type)
  {
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

  case USHORT_TYPE:
  case SHORT_TYPE:
  case HALF_TYPE: return 2;

  case UINT_TYPE:
  case INT_TYPE:
  case FLOAT_TYPE: return 4;

  case ULONG_TYPE:
  case LONG_TYPE:
  case DOUBLE_TYPE: return 8;

  default: throw generator_not_supported_exception("Unsupported scalartype");
  }
}

inline std::string append_width(std::string const & str, unsigned int width)
{
  if (width==1)
    return str;
  return str + tools::to_string(width);
}

template<typename MapT>
class create_map
{
    typedef typename MapT::key_type T;
    typedef typename MapT::mapped_type U;
public:
    create_map(const T& key, const U& val)
    {
        map_.insert(std::make_pair(key,val));
    }

    create_map<MapT>& operator()(const T& key, const U& val)
    {
        map_.insert(std::make_pair(key,val));
        return *this;
    }

    operator MapT()
    {
        return map_;
    }
private:
    MapT map_;
};

typedef create_map<std::multimap<std::string, std::string> > create_process_accessors;
typedef create_map<std::map<std::string, std::string> > create_evaluate_accessors;


}
}
#endif
