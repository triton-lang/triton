#ifndef PYTHON_ISAAC_SRC_COMMON_HPP
#define PYTHON_ISAAC_SRC_COMMON_HPP

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "isaac/array.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)

namespace bp = boost::python;
namespace np = boost::numpy;
namespace isc = isaac;

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

  inline isc::numeric_type extract_dtype(bp::object const & odtype)
  {
      std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
      if(name=="class")
        name = bp::extract<std::string>(odtype.attr("__name__"))();
      else
        name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

      if(name=="int8") return isc::CHAR_TYPE;
      else if(name=="uint8") return isc::UCHAR_TYPE;
      else if(name=="int16") return isc::SHORT_TYPE;
      else if(name=="uint16") return isc::USHORT_TYPE;
      else if(name=="int32") return isc::INT_TYPE;
      else if(name=="uint32") return isc::UINT_TYPE;
      else if(name=="int64") return isc::LONG_TYPE;
      else if(name=="uint64") return isc::ULONG_TYPE;
      else if(name=="float32") return isc::FLOAT_TYPE;
      else if(name=="float64") return isc::DOUBLE_TYPE;
      else
      {
          PyErr_SetString(PyExc_TypeError, "Data type not understood");
          bp::throw_error_already_set();
          throw;
      }
  }

  inline isc::expression_type extract_template_type(bp::object const & odtype)
  {
      std::string name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();
      if(name=="class")
        name = bp::extract<std::string>(odtype.attr("__name__"))();
      else
        name = bp::extract<std::string>(odtype.attr("__class__").attr("__name__"))();

      if(name=="vaxpy") return isc::VECTOR_AXPY_TYPE;
      else if(name=="maxpy") return isc::MATRIX_AXPY_TYPE;
      else if(name=="reduction") return isc::REDUCTION_TYPE;
      else if(name=="mreduction_rows") return isc::ROW_WISE_REDUCTION_TYPE;
      else if(name=="mreduction_cols") return isc::COL_WISE_REDUCTION_TYPE;
      else if(name=="mproduct_nn") return isc::MATRIX_PRODUCT_NN_TYPE;
      else if(name=="mproduct_tn") return isc::MATRIX_PRODUCT_TN_TYPE;
      else if(name=="mproduct_nt") return isc::MATRIX_PRODUCT_NT_TYPE;
      else if(name=="mproduct_tt") return isc::MATRIX_PRODUCT_TT_TYPE;
      else
      {
          PyErr_SetString(PyExc_TypeError, "Template type not understood");
          bp::throw_error_already_set();
          throw;
      }
  }

  inline isc::numeric_type to_isc_dtype(np::dtype const & T)
  {
    if(T==np::detail::get_int_dtype<8, false>()) return isc::CHAR_TYPE;
    else if(T==np::detail::get_int_dtype<8, true>()) return isc::UCHAR_TYPE;
    else if(T==np::detail::get_int_dtype<16, false>()) return isc::SHORT_TYPE;
    else if(T==np::detail::get_int_dtype<16, true>()) return isc::USHORT_TYPE;
    else if(T==np::detail::get_int_dtype<32, false>()) return isc::INT_TYPE;
    else if(T==np::detail::get_int_dtype<32, true>()) return isc::UINT_TYPE;
    else if(T==np::detail::get_int_dtype<64, false>()) return isc::LONG_TYPE;
    else if(T==np::detail::get_int_dtype<64, true>()) return isc::ULONG_TYPE;
  //  else if(T==np::detail::get_float_dtype<16>()) return isc::HALF_TYPE;
    else if(T==np::detail::get_float_dtype<32>()) return isc::FLOAT_TYPE;
    else if(T==np::detail::get_float_dtype<64>()) return isc::DOUBLE_TYPE;
    else{
      PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
      bp::throw_error_already_set();
      throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
    }
  }

  inline np::dtype to_np_dtype(isc::numeric_type const & T) throw()
  {
    if(T==isc::CHAR_TYPE) return np::detail::get_int_dtype<8, false>();
    else if(T==isc::UCHAR_TYPE) return np::detail::get_int_dtype<8, true>();
    else if(T==isc::SHORT_TYPE) return np::detail::get_int_dtype<16, false>();
    else if(T==isc::USHORT_TYPE) return np::detail::get_int_dtype<16, true>();
    else if(T==isc::INT_TYPE) return np::detail::get_int_dtype<32, false>();
    else if(T==isc::UINT_TYPE) return np::detail::get_int_dtype<32, true>();
    else if(T==isc::LONG_TYPE) return np::detail::get_int_dtype<64, false>();
    else if(T==isc::ULONG_TYPE) return np::detail::get_int_dtype<64, true>();
  //  else if(T==isc::HALF_TYPE) return np::detail::get_float_dtype<16>();
    else if(T==isc::FLOAT_TYPE) return np::detail::get_float_dtype<32>();
    else if(T==isc::DOUBLE_TYPE) return np::detail::get_float_dtype<64>();
    else{
      PyErr_SetString(PyExc_TypeError, "Unrecognized datatype");
      bp::throw_error_already_set();
      throw; // suppress warning; throw_error_already_set() never returns but isn't marked noreturn: https://svn.boost.org/trac/boost/ticket/1482
    }
  }


}

#endif
