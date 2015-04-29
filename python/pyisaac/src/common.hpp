#ifndef PYTHON_ISAAC_SRC_COMMON_HPP
#define PYTHON_ISAAC_SRC_COMMON_HPP

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "isaac/array.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)

namespace bp = boost::python;
namespace np = boost::numpy;
namespace atd = isaac;

namespace detail
{
  template<class IT>
  bp::list to_list(IT const & begin, IT const & end)
  {
    bp::list res;
    for (IT it = begin; it != end; ++it)
      res.append(*it);
    return res;
  }

  template<class T>
  std::vector<T> to_vector(bp::list const & list)
  {
    std::size_t len = bp::len(list);
    std::vector<T> res; res.reserve(len);
    for(std::size_t i = 0 ; i < len ; ++i)
      res.push_back(boost::python::extract<T>(list[i]));
    return res;
  }

  inline atd::numeric_type extract_dtype(bp::object const & odtype)
  {
      std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
      if(name=="class")
        name = bp::extract<std::string>(odtype.attr("__name__"))();
      else
        name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

      if(name=="int8") return atd::CHAR_TYPE;
      else if(name=="uint8") return atd::UCHAR_TYPE;
      else if(name=="int16") return atd::SHORT_TYPE;
      else if(name=="uint16") return atd::USHORT_TYPE;
      else if(name=="int32") return atd::INT_TYPE;
      else if(name=="uint32") return atd::UINT_TYPE;
      else if(name=="int64") return atd::LONG_TYPE;
      else if(name=="uint64") return atd::ULONG_TYPE;
      else if(name=="float32") return atd::FLOAT_TYPE;
      else if(name=="float64") return atd::DOUBLE_TYPE;
      else
      {
          PyErr_SetString(PyExc_TypeError, "Data type not understood");
          bp::throw_error_already_set();
          throw;
      }
  }

  inline atd::expression_type extract_template_type(bp::object const & odtype)
  {
      std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
      if(name=="class")
        name = bp::extract<std::string>(odtype.attr("__name__"))();
      else
        name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

      if(name=="vaxpy") return atd::VECTOR_AXPY_TYPE;
      else if(name=="maxpy") return atd::MATRIX_AXPY_TYPE;
      else if(name=="reduction") return atd::REDUCTION_TYPE;
      else if(name=="mreduction_rows") return atd::ROW_WISE_REDUCTION_TYPE;
      else if(name=="mreduction_cols") return atd::COL_WISE_REDUCTION_TYPE;
      else if(name=="mproduct_nn") return atd::MATRIX_PRODUCT_NN_TYPE;
      else if(name=="mproduct_tn") return atd::MATRIX_PRODUCT_TN_TYPE;
      else if(name=="mproduct_nt") return atd::MATRIX_PRODUCT_NT_TYPE;
      else if(name=="mproduct_tt") return atd::MATRIX_PRODUCT_TT_TYPE;
      else
      {
          PyErr_SetString(PyExc_TypeError, "Template type not understood");
          bp::throw_error_already_set();
          throw;
      }
  }

  inline atd::numeric_type to_atd_dtype(np::dtype const & T)
  {
    if(T==np::detail::get_int_dtype<8, false>()) return atd::CHAR_TYPE;
    else if(T==np::detail::get_int_dtype<8, true>()) return atd::UCHAR_TYPE;
    else if(T==np::detail::get_int_dtype<16, false>()) return atd::SHORT_TYPE;
    else if(T==np::detail::get_int_dtype<16, true>()) return atd::USHORT_TYPE;
    else if(T==np::detail::get_int_dtype<32, false>()) return atd::INT_TYPE;
    else if(T==np::detail::get_int_dtype<32, true>()) return atd::UINT_TYPE;
    else if(T==np::detail::get_int_dtype<64, false>()) return atd::LONG_TYPE;
    else if(T==np::detail::get_int_dtype<64, true>()) return atd::ULONG_TYPE;
  //  else if(T==np::detail::get_float_dtype<16>()) return atd::HALF_TYPE;
    else if(T==np::detail::get_float_dtype<32>()) return atd::FLOAT_TYPE;
    else if(T==np::detail::get_float_dtype<64>()) return atd::DOUBLE_TYPE;
    else{
      PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
      bp::throw_error_already_set();
      throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
    }
  }

  inline np::dtype to_np_dtype(atd::numeric_type const & T) throw()
  {
    if(T==atd::CHAR_TYPE) return np::detail::get_int_dtype<8, false>();
    else if(T==atd::UCHAR_TYPE) return np::detail::get_int_dtype<8, true>();
    else if(T==atd::SHORT_TYPE) return np::detail::get_int_dtype<16, false>();
    else if(T==atd::USHORT_TYPE) return np::detail::get_int_dtype<16, true>();
    else if(T==atd::INT_TYPE) return np::detail::get_int_dtype<32, false>();
    else if(T==atd::UINT_TYPE) return np::detail::get_int_dtype<32, true>();
    else if(T==atd::LONG_TYPE) return np::detail::get_int_dtype<64, false>();
    else if(T==atd::ULONG_TYPE) return np::detail::get_int_dtype<64, true>();
  //  else if(T==atd::HALF_TYPE) return np::detail::get_float_dtype<16>();
    else if(T==atd::FLOAT_TYPE) return np::detail::get_float_dtype<32>();
    else if(T==atd::DOUBLE_TYPE) return np::detail::get_float_dtype<64>();
    else{
      PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
      bp::throw_error_already_set();
      throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
    }
  }


}

#endif
