#ifndef ATIDLAS_UTILS_HPP
#define ATIDLAS_UTILS_HPP


#include <sstream>

#include "atidlas/tools/to_string.hpp"
#include "atidlas/forwards.h"
#include "atidlas/backend/forwards.h"
#include "atidlas/scheduler/forwards.h"
#include "atidlas/tools/find_and_replace.hpp"

namespace atidlas
{
namespace tools
{

inline std::string numeric_type_to_string(numeric_type const & type)
{
  switch (type)
  {
  //case CHAR_TYPE: return "char";
  //case UCHAR_TYPE: return "uchar";
  //case SHORT_TYPE: return "short";
  //case USHORT_TYPE: return "ushort";
  case INT_TYPE:  return "int";
  case UINT_TYPE: return "uint";
  case LONG_TYPE:  return "long";
  case ULONG_TYPE: return "ulong";
  case FLOAT_TYPE : return "float";
  case DOUBLE_TYPE : return "double";
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class T, class U>
struct is_same_type { enum { value = 0 }; };

template<class T>
struct is_same_type<T,T> { enum { value = 1 }; };

inline bool is_reduction(scheduler::statement_node const & node)
{
  return node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
      || node.op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY
      || node.op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY
      || node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE
      || node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE;
}

inline bool is_index_reduction(scheduler::op_element const & op)
{
  return op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE;
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

inline bool node_leaf(scheduler::op_element const & op)
{
  using namespace scheduler;
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

inline bool elementwise_operator(scheduler::op_element const & op)
{
  using namespace scheduler;
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

inline bool elementwise_function(scheduler::op_element const & op)
{
  using namespace scheduler;
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

inline scheduler::lhs_rhs_element & lhs_rhs_element(scheduler::statement const & st, atidlas_int_t idx, leaf_t leaf)
{
  using namespace tools;
  assert(leaf==LHS_NODE_TYPE || leaf==RHS_NODE_TYPE);
  if (leaf==LHS_NODE_TYPE)
    return const_cast<scheduler::lhs_rhs_element &>(st.array()[idx].lhs);
  return const_cast<scheduler::lhs_rhs_element &>(st.array()[idx].rhs);
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

}
}
#endif
