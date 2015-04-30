#include <list>
#include <functional>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "isaac/array.h"

#include "isaac/backend/templates/vaxpy.h"
#include "isaac/backend/templates/maxpy.h"
#include "isaac/backend/templates/reduction.h"
#include "isaac/backend/templates/mreduction.h"
#include "isaac/backend/templates/mproduct.h"

#include "isaac/model/model.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)
namespace bp = boost::python;
namespace isc = isaac;
namespace np = boost::numpy;

namespace detail
{
  
isc::numeric_type to_isc_dtype(np::dtype const & T)
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

np::dtype to_np_dtype(isc::numeric_type const & T) throw()
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

bp::tuple get_shape(isc::array const & x)
{
  return bp::make_tuple(x.shape()[0], x.shape()[1]);
}

//void set_shape(isc::array & x, bp::tuple const & t)
//{
//  unsigned int len = bp::len(t);
//  isc::int_t size1 = bp::extract<isc::int_t>(t[0]);
//  isc::int_t size2 = len<2?1:bp::extract<isc::int_t>(t[1]);
//  x.reshape(size1, size2);
//}

//boost::python::dict create_queues(isc::cl_ext::queues_t queues)
//{
//  boost::python::dict dictionary;
//  for (isc::cl_ext::queues_t::iterator it = queues.begin(); it != queues.end(); ++it) {
//    bp::list list;
//    for (isc::cl_ext::queues_t::mapped_type::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
//      list.append(*itt);
//    dictionary[it->first] = list;
//  }
//  return dictionary;
//}

template<class T>
struct datatype : public isc::value_scalar
{
  datatype(T t) : isc::value_scalar(t){ }

};

template<class T>
unsigned int size(datatype<T> const & dt)
{ return isc::size_of(dt.dtype()) ; }

#define INSTANTIATE(name, clname) \
  struct name : public detail::datatype<clname> {  name(clname value) : detail::datatype<clname>(value){} };
  INSTANTIATE(int8, cl_char)
  INSTANTIATE(uint8, cl_uchar)
  INSTANTIATE(int16, cl_short)
  INSTANTIATE(uint16, cl_ushort)
  INSTANTIATE(int32, cl_int)
  INSTANTIATE(uint32, cl_uint)
  INSTANTIATE(int64, cl_long)
  INSTANTIATE(uint64, cl_ulong)
  INSTANTIATE(float32, cl_float)
  INSTANTIATE(float64, cl_double)
#undef INSTANTIATE

}



void export_core()
{

#define INSTANTIATE(name, clname) \
  bp::class_<detail::datatype<clname>, bp::bases<isc::value_scalar> >(#name, bp::init<clname>());\
  bp::class_<detail::name, bp::bases<detail::datatype<clname> > >(#name, bp::init<clname>())\
    .add_property("size", &detail::size<clname>)\
    ;


  INSTANTIATE(int8, cl_char)
  INSTANTIATE(uint8, cl_uchar)
  INSTANTIATE(int16, cl_short)
  INSTANTIATE(uint16, cl_ushort)
  INSTANTIATE(int32, cl_int)
  INSTANTIATE(uint32, cl_uint)
  INSTANTIATE(int64, cl_long)
  INSTANTIATE(uint64, cl_ulong)
  INSTANTIATE(float32, cl_float)
  INSTANTIATE(float64, cl_double)
  #undef INSTANTIATE

  bp::enum_<isc::expression_type>("operations")
    MAP_ENUM(VECTOR_AXPY_TYPE, isc)
    MAP_ENUM(MATRIX_AXPY_TYPE, isc)
    MAP_ENUM(REDUCTION_TYPE, isc)
    MAP_ENUM(ROW_WISE_REDUCTION_TYPE, isc)
    MAP_ENUM(COL_WISE_REDUCTION_TYPE, isc)
    MAP_ENUM(VECTOR_AXPY_TYPE, isc)
    MAP_ENUM(VECTOR_AXPY_TYPE, isc)
    MAP_ENUM(VECTOR_AXPY_TYPE, isc)
    MAP_ENUM(VECTOR_AXPY_TYPE, isc)
    ;
}


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

  bp::list nv_compute_capability(isc::driver::Device const & device)
  {
    bp::list res;
    std::pair<unsigned int, unsigned int> cc = device.nv_compute_capability();
    res.append(cc.first);
    res.append(cc.second);
    return res;
  }

  bp::list get_platforms()
  {
    std::vector<isc::driver::Platform> platforms(isc::driver::Platform::get());
    return to_list(platforms.begin(), platforms.end());
  }

  bp::list get_devices(isc::driver::Platform const & platform)
  {
    std::vector<isc::driver::Device> devices(platform.devices());
    return to_list(devices.begin(), devices.end());
  }

  isc::numeric_type extract_dtype(bp::object const & odtype)
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

  isc::expression_type extract_template_type(bp::object const & odtype)
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

  struct model_map_indexing
  {
      static isc::model& get_item(isc::model_map_t& container, bp::tuple i_)
      {
          isc::expression_type expression = extract_template_type(i_[0]);
          isc::numeric_type dtype = extract_dtype(i_[1]);
          isc::model_map_t::iterator i = container.find(std::make_pair(expression, dtype));
          if (i == container.end())
          {
              PyErr_SetString(PyExc_KeyError, "Invalid key");
              bp::throw_error_already_set();
          }
          return *i->second;
      }

      static void set_item(isc::model_map_t& container, bp::tuple i_, isc::model const & v)
      {
          isc::expression_type expression = extract_template_type(i_[0]);
          isc::numeric_type dtype = extract_dtype(i_[1]);
          container[std::make_pair(expression, dtype)].reset(new isc::model(v));
      }
  };

  std::string to_string(isc::driver::device_type type)
  {
    if(type==isc::driver::DEVICE_TYPE_CPU) return "CPU";
    if(type==isc::driver::DEVICE_TYPE_GPU) return "GPU";
    if(type==isc::driver::DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    throw;
  }

  std::shared_ptr<isc::driver::Context> make_context(isc::driver::Device const & dev)
  { return std::shared_ptr<isc::driver::Context>(new isc::driver::Context(dev)); }

  bp::tuple flush(isc::array_expression const & expression, unsigned int queue_id, bp::list dependencies, bool tune, int label, std::string const & program_name, bool force_recompile)
  {
      std::list<isc::driver::Event> events;
      std::vector<isc::driver::Event> cdependencies = to_vector<isc::driver::Event>(dependencies);
      std::shared_ptr<isc::array> parray(new isc::array(isc::control(expression,
                                                                    isc::execution_options_type(queue_id, &events, &cdependencies),
                                                                    isc::dispatcher_options_type(tune, label),
                                                                    isc::compilation_options_type(program_name, force_recompile))));
      return bp::make_tuple(parray, to_list(events.begin(), events.end()));
  }
}

struct state_type{ };
state_type state;

void export_cl()
{
  typedef std::vector<isc::driver::CommandQueue> queues_t;
  bp::class_<queues_t>("queues")
      .def("__len__", &queues_t::size)
      .def("__getitem__", &bp::vector_indexing_suite<queues_t>::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &bp::vector_indexing_suite<queues_t>::set_item, bp::with_custodian_and_ward<1,2>())
      .def("append", &bp::vector_indexing_suite<queues_t>::append)

      ;

  bp::class_<isc::model_map_t>("models")
      .def("__getitem__", &detail::model_map_indexing::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &detail::model_map_indexing::set_item, bp::with_custodian_and_ward<1,2>())
      ;

  bp::enum_<isc::driver::backend_type>
      ("backend_type")
      .value("OPENCL", isc::driver::OPENCL)
  #ifdef ISAAC_WITH_CUDA
      .value("CUDA", isc::driver::CUDA)
  #endif
      ;

  bp::enum_<isc::driver::device_type>
      ("device_type")
      .value("DEVICE_TYPE_GPU", isc::driver::DEVICE_TYPE_GPU)
      .value("DEVICE_TYPE_CPU", isc::driver::DEVICE_TYPE_CPU)
      ;


  bp::class_<isc::driver::Platform>("platform", bp::no_init)
      .def("get_devices", &detail::get_devices)
      .add_property("name",&isc::driver::Platform::name)
      ;

  bp::enum_<isaac::driver::Device::VENDOR>
      ("vendor")
      .value("AMD", isc::driver::Device::AMD)
      .value("INTEL", isc::driver::Device::INTEL)
      .value("NVIDIA", isc::driver::Device::NVIDIA)
      .value("UNKNOWN", isc::driver::Device::UNKNOWN)
      ;

  bp::class_<isc::driver::Device>("device", bp::no_init)
      .add_property("clock_rate", &isc::driver::Device::clock_rate)
      .add_property("name", &isc::driver::Device::name)
      .add_property("type", &isc::driver::Device::type)
      .add_property("platform", &isc::driver::Device::platform)
      .add_property("vendor", &isc::driver::Device::vendor)
      .add_property("nv_compute_capability", &detail::nv_compute_capability)
      ;

  bp::class_<isc::driver::Context>("context", bp::no_init)
      .def("__init__", bp::make_constructor(&detail::make_context))
      .add_property("queues", bp::make_function(static_cast<std::vector<isc::driver::CommandQueue> & (*)(const isc::driver::Context&)>( [](const isc::driver::Context & ctx) -> std::vector<isc::driver::CommandQueue> & { return isc::driver::queues[ctx]; }) , bp::return_internal_reference<>()))
      .add_property("backend", &isc::driver::Context::backend)
      ;

  bp::class_<isc::driver::CommandQueue>("command_queue", bp::init<isc::driver::Context, isc::driver::Device>())
      .def("synchronize", &isc::driver::CommandQueue::synchronize)
      .add_property("models", bp::make_function(&isc::get_model_map, bp::return_internal_reference<>()))
      .add_property("device", bp::make_function(&isc::driver::CommandQueue::device, bp::return_internal_reference<>()))
      ;

  bp::class_<isc::driver::Event>("event", bp::init<isc::driver::backend_type>())
      .add_property("elapsed_time", &isc::driver::Event::elapsed_time)
     ;

  bp::def("device_type_to_string", &detail::to_string);

  bp::def("get_platforms", &detail::get_platforms);

  bp::def("flush", &detail::flush, (bp::arg("expression"), bp::arg("queue_id") = 0, bp::arg("dependencies")=bp::list(), bp::arg("tune") = false, bp::arg("label")=-1, bp::arg("program_name")="", bp::arg("recompile") = false));

  bp::class_<state_type>("state_type")
          .def_readwrite("queue_properties",&isc::driver::queues.queue_properties)
      ;

  bp::scope().attr("state") = bp::object(bp::ptr(&state));

  bp::scope().attr("CL_QUEUE_PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
  bp::scope().attr("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE") = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}

namespace detail
{
  std::shared_ptr<isc::array>
  ndarray_to_iscarray(const np::ndarray& array, const isc::driver::Context& ctx)
  {

    int d = array.get_nd();
    if (d > 2) {
      PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
      bp::throw_error_already_set();
    }

    isc::numeric_type dtype = to_isc_dtype(array.get_dtype());
    isc::int_t size = (isc::int_t)array.shape(0);
    isc::array* v = new isc::array(size, dtype, ctx);

    void* data = (void*)array.get_data();
    isc::copy(data, *v);

    return std::shared_ptr<isc::array>(v);
  }



  std::shared_ptr<isc::array> create_array(bp::object const & obj, bp::object odtype, isc::driver::Context context)
  {
    return ndarray_to_iscarray(np::from_object(obj, to_np_dtype(extract_dtype(odtype))), context);
  }

  std::shared_ptr<isc::array> create_empty_array(bp::object sizes, bp::object odtype, isc::driver::Context context)
  {
      typedef std::shared_ptr<isc::array> result_type;

      std::size_t len;
      int size1;
      int size2;
      try{
        len = bp::len(sizes);
        size1 = bp::extract<int>(sizes[0])();
        size2 = bp::extract<int>(sizes[1])();
      }catch(bp::error_already_set const &){
        PyErr_Clear();
        len = 1;
        size1 = bp::extract<int>(sizes)();
      }

      isc::numeric_type dtype = extract_dtype(odtype);
      if(len < 1 || len > 2)
      {
          PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
          bp::throw_error_already_set();
      }
      if(len==1)
          return result_type(new isc::array(size1, dtype, context));
      return result_type(new isc::array(size1, size2, dtype, context));
  }

  std::string type_name(bp::object const & obj)
  {
    std::string name = bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
    if(name=="class")
      return bp::extract<std::string>(obj.attr("__name__"))();
    else
      return bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
  }

  std::shared_ptr<isc::scalar> construct_scalar(bp::object obj, isc::driver::Context const & context)
  {
    typedef std::shared_ptr<isc::scalar> result_type;
    std::string name = type_name(obj);
    if(name=="int") return result_type(new isc::scalar(bp::extract<int>(obj)(), context));
    else if(name=="float") return result_type(new isc::scalar(bp::extract<double>(obj)(), context));
    else if(name=="long") return result_type(new isc::scalar(bp::extract<long>(obj)(), context));
    else if(name=="int") return result_type(new isc::scalar(bp::extract<int>(obj)(), context));

    else if(name=="int8") return result_type(new isc::scalar(isc::CHAR_TYPE, context));
    else if(name=="uint8") return result_type(new isc::scalar(isc::UCHAR_TYPE, context));
    else if(name=="int16") return result_type(new isc::scalar(isc::SHORT_TYPE, context));
    else if(name=="uint16") return result_type(new isc::scalar(isc::USHORT_TYPE, context));
    else if(name=="int32") return result_type(new isc::scalar(isc::INT_TYPE, context));
    else if(name=="uint32") return result_type(new isc::scalar(isc::UINT_TYPE, context));
    else if(name=="int64") return result_type(new isc::scalar(isc::LONG_TYPE, context));
    else if(name=="uint64") return result_type(new isc::scalar(isc::ULONG_TYPE, context));
    else if(name=="float32") return result_type(new isc::scalar(isc::FLOAT_TYPE, context));
    else if(name=="float64") return result_type(new isc::scalar(isc::DOUBLE_TYPE, context));
    else{
        PyErr_SetString(PyExc_TypeError, "Data type not understood");
        bp::throw_error_already_set();
        throw;
    }

  }
}

void export_array()
{
#define ADD_SCALAR_HANDLING(OP)\
  .def(bp::self                                    OP int())\
  .def(bp::self                                    OP long())\
  .def(bp::self                                    OP double())\
  .def(bp::self                                    OP bp::other<isc::value_scalar>())\
  .def(int()                                       OP bp::self)\
  .def(long()                                      OP bp::self)\
  .def(double()                                     OP bp::self)\
  .def(bp::other<isc::value_scalar>()              OP bp::self)

#define ADD_ARRAY_OPERATOR(OP)\
  .def(bp::self OP bp::self)\
  ADD_SCALAR_HANDLING(OP)

  bp::class_<isc::expressions_tuple>
      ("array_expression_container", bp::init<isc::array_expression const &>())
  ;

  bp::class_<isc::array_expression >("array_expression", bp::no_init)
      ADD_ARRAY_OPERATOR(+)
      ADD_ARRAY_OPERATOR(-)
      ADD_ARRAY_OPERATOR(*)
      ADD_ARRAY_OPERATOR(/)
      ADD_ARRAY_OPERATOR(>)
      ADD_ARRAY_OPERATOR(>=)
      ADD_ARRAY_OPERATOR(<)
      ADD_ARRAY_OPERATOR(<=)
      ADD_ARRAY_OPERATOR(==)
      ADD_ARRAY_OPERATOR(!=)
      .add_property("context", bp::make_function(&isc::array_expression::context, bp::return_internal_reference<>()))
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
  ;
#undef ADD_ARRAY_OPERATOR

#define ADD_ARRAY_OPERATOR(OP) \
  .def(bp::self                            OP bp::self)\
  .def(bp::self                            OP bp::other<isc::array_expression>())\
  .def(bp::other<isc::array_expression>() OP bp::self) \
  ADD_SCALAR_HANDLING(OP)

  bp::class_<isc::array,
          std::shared_ptr<isc::array> >
  ( "array", bp::no_init)
      .def("__init__", bp::make_constructor(detail::create_array, bp::default_call_policies(), (bp::arg("obj"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=isc::driver::queues.default_context())))
      .def(bp::init<isc::array_expression>())
      .add_property("dtype", &isc::array::dtype)
      .add_property("context", bp::make_function(&isc::array::context, bp::return_internal_reference<>()))
      .add_property("T", &isc::array::T)
      .add_property("shape", &detail::get_shape)
      ADD_ARRAY_OPERATOR(+)
      ADD_ARRAY_OPERATOR(-)
      ADD_ARRAY_OPERATOR(*)
      ADD_ARRAY_OPERATOR(/)
      ADD_ARRAY_OPERATOR(>)
      ADD_ARRAY_OPERATOR(>=)
      ADD_ARRAY_OPERATOR(<)
      ADD_ARRAY_OPERATOR(<=)
      ADD_ARRAY_OPERATOR(==)
      ADD_ARRAY_OPERATOR(!=)
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
      .def(bp::self_ns::str(bp::self_ns::self))
  ;

  bp::class_<isc::scalar, bp::bases<isc::array> >
      ("scalar", bp::no_init)
      .def("__init__", bp::make_constructor(detail::construct_scalar, bp::default_call_policies(), (bp::arg(""), bp::arg("context")=isc::driver::queues.default_context())))
      ;

  //Other numpy-like initializers
  bp::def("empty", &detail::create_empty_array, (bp::arg("shape"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=isc::driver::queues.default_context()));

//Binary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array const &, isc::array const &)>(&isc::name));\
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array_expression const &, isc::array const &)>(&isc::name));\
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array const &, isc::array_expression const &)>(&isc::name));\
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array_expression const &, isc::array_expression const &)>(&isc::name));

  MAP_FUNCTION(maximum)
  MAP_FUNCTION(minimum)
  MAP_FUNCTION(pow)
  MAP_FUNCTION(dot)
#undef MAP_FUNCTION

//Unary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array const &)>(&isc::name));\
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array_expression const &)>(&isc::name));

  MAP_FUNCTION(abs)
  MAP_FUNCTION(acos)
  MAP_FUNCTION(asin)
  MAP_FUNCTION(atan)
  MAP_FUNCTION(ceil)
  MAP_FUNCTION(cos)
  MAP_FUNCTION(cosh)
  MAP_FUNCTION(exp)
  MAP_FUNCTION(floor)
  MAP_FUNCTION(log)
  MAP_FUNCTION(log10)
  MAP_FUNCTION(sin)
  MAP_FUNCTION(sinh)
  MAP_FUNCTION(sqrt)
  MAP_FUNCTION(tan)
  MAP_FUNCTION(tanh)
#undef MAP_FUNCTION

  /*--- Reduction operators----*/
  //---------------------------------------
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array const &, isc::int_t)>(&isc::name));\
      bp::def(#name, static_cast<isc::array_expression (*)(isc::array_expression const &, isc::int_t)>(&isc::name));

  MAP_FUNCTION(sum)
  MAP_FUNCTION(max)
  MAP_FUNCTION(min)
  MAP_FUNCTION(argmax)
  MAP_FUNCTION(argmin)
#undef MAP_FUNCTION
}

void export_scalar()
{
  bp::class_<isc::value_scalar>("value_scalar", bp::no_init)
          .add_property("dtype", &isc::value_scalar::dtype);
}


namespace detail
{
  bp::list input_sizes(isaac::base & temp, isc::expressions_tuple const & tree)
  {
      std::vector<int> tmp = temp.input_sizes(tree);
      return detail::to_list(tmp.begin(), tmp.end());
  }

  std::shared_ptr<isc::model> construct_model(bp::object dtype, bp::object const & tp, isc::driver::CommandQueue & queue)
  {
      return std::shared_ptr<isc::model>(new isc::model(extract_template_type(tp), extract_dtype(dtype), (isc::base const &)bp::extract<isc::base>(tp), queue));
  }
}

void export_model()
{

  bp::class_<isaac::model>("model", bp::no_init)
                  .def("__init__", bp::make_constructor(detail::construct_model))
                  .def("execute", &isc::model::execute);
  
  bp::enum_<isaac::fetching_policy_type>
      ("fetching_policy_type")
      .value("FETCH_FROM_LOCAL", isc::FETCH_FROM_LOCAL)
      .value("FETCH_FROM_GLOBAL_STRIDED", isc::FETCH_FROM_GLOBAL_STRIDED)
      .value("FETCH_FROM_GLOBAL_CONTIGUOUS", isc::FETCH_FROM_GLOBAL_CONTIGUOUS)
      ;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &isaac::base::parameters_type::name)
    bp::class_<isaac::base, boost::noncopyable>("base", bp::no_init)
            .def("lmem_usage", &isaac::base::lmem_usage)
            .def("registers_usage", &isaac::base::registers_usage)
            .def("is_invalid", &isaac::base::is_invalid)
            .def("input_sizes", &detail::input_sizes)
        ;
    #undef __PROP
  }

  #define WRAP_BASE(name) bp::class_<isaac::base_impl<isaac::name, isaac::name::parameters_type>, bp::bases<isaac::base>, boost::noncopyable>(#name "_base_impl", bp::no_init);
  #define WRAP_TEMPLATE(name, basename, ...) bp::class_<isaac::name, bp::bases<isaac::base_impl<isaac::basename, isaac::basename::parameters_type> > >(#name, bp::init<__VA_ARGS__>())\
                                      .add_property("local_size_0", &isc::name::local_size_0)\
                                      .add_property("local_size_1", &isc::name::local_size_1);
  #define WRAP_SINGLE_TEMPLATE(name, ...) WRAP_BASE(name) WRAP_TEMPLATE(name, name, __VA_ARGS__)

  //Vector AXPY
  WRAP_SINGLE_TEMPLATE(vaxpy, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(maxpy, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(reduction, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_BASE(mreduction)
  WRAP_TEMPLATE(mreduction_rows, mreduction, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_TEMPLATE(mreduction_cols, mreduction, uint, uint, uint, uint, uint, isaac::fetching_policy_type)
  WRAP_BASE(mproduct)
  WRAP_TEMPLATE(mproduct_nn, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tn, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_nt, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tt, mproduct, uint, uint, uint, uint, uint, uint, uint, uint, isaac::fetching_policy_type, isaac::fetching_policy_type, uint, uint)


}

BOOST_PYTHON_MODULE(_isaac)
{
  Py_Initialize();
  np::initialize();

  // specify that this module is actually a package
  bp::object package = bp::scope();
  package.attr("__path__") = "_isaac";

  export_scalar();
  export_core();
  export_cl();
  export_model();
  export_array();
}
