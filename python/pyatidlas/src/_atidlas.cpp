#include <list>
#include <functional>

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/numpy.hpp>
#include <boost/numpy/dtype.hpp>

#include "atidlas/array.h"

#include "atidlas/backend/templates/vaxpy.h"
#include "atidlas/backend/templates/maxpy.h"
#include "atidlas/backend/templates/reduction.h"
#include "atidlas/backend/templates/mreduction.h"
#include "atidlas/backend/templates/mproduct.h"

#include "atidlas/model/model.h"

#define MAP_ENUM(v, ns) .value(#v, ns::v)
namespace bp = boost::python;
namespace atd = atidlas;
namespace np = boost::numpy;

namespace detail
{
  
atd::numeric_type to_atd_dtype(np::dtype const & T)
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

np::dtype to_np_dtype(atd::numeric_type const & T) throw()
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

bp::tuple get_shape(atd::array const & x)
{
  return bp::make_tuple(x.shape()._1, x.shape()._2);
}

//void set_shape(atd::array & x, bp::tuple const & t)
//{
//  unsigned int len = bp::len(t);
//  atd::int_t size1 = bp::extract<atd::int_t>(t[0]);
//  atd::int_t size2 = len<2?1:bp::extract<atd::int_t>(t[1]);
//  x.reshape(size1, size2);
//}

//boost::python::dict create_queues(atd::cl_ext::queues_t queues)
//{
//  boost::python::dict dictionary;
//  for (atd::cl_ext::queues_t::iterator it = queues.begin(); it != queues.end(); ++it) {
//    bp::list list;
//    for (atd::cl_ext::queues_t::mapped_type::iterator itt = it->second.begin(); itt != it->second.end(); ++itt)
//      list.append(*itt);
//    dictionary[it->first] = list;
//  }
//  return dictionary;
//}

template<class T>
struct datatype : public atd::value_scalar
{
  datatype(T t) : atd::value_scalar(t){ }

};

template<class T>
unsigned int size(datatype<T> const & dt)
{ return atd::size_of(dt.dtype()) ; }

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
  bp::class_<detail::datatype<clname>, bp::bases<atd::value_scalar> >(#name, bp::init<clname>());\
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

  bp::enum_<atd::expression_type>("operations")
    MAP_ENUM(VECTOR_AXPY_TYPE, atd)
    MAP_ENUM(MATRIX_AXPY_TYPE, atd)
    MAP_ENUM(REDUCTION_TYPE, atd)
    MAP_ENUM(ROW_WISE_REDUCTION_TYPE, atd)
    MAP_ENUM(COL_WISE_REDUCTION_TYPE, atd)
    MAP_ENUM(VECTOR_AXPY_TYPE, atd)
    MAP_ENUM(VECTOR_AXPY_TYPE, atd)
    MAP_ENUM(VECTOR_AXPY_TYPE, atd)
    MAP_ENUM(VECTOR_AXPY_TYPE, atd)
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
    for(int i = 0 ; i < len ; ++i)
      res.push_back(boost::python::extract<T>(list[i]));
    return res;
  }

  bp::list nv_compute_capability(cl::Device const & device)
  {
    bp::list res;
    cl_uint cmin, cmaj;
    clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(cl_uint), (cl_uint*)&cmaj, NULL);
    clGetDeviceInfo(device(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, sizeof(cl_uint), (cl_uint*)&cmin, NULL);
    res.append(cmaj);
    res.append(cmin);
    return res;
  }

  bp::list get_platforms()
  {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    return to_list(platforms.begin(), platforms.end());
  }

  bp::list get_devices(cl::Platform const & platform)
  {
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    return to_list(devices.begin(), devices.end());
  }

  atd::numeric_type extract_dtype(bp::object const & odtype)
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

  atd::expression_type extract_template_type(bp::object const & odtype)
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

  struct model_map_indexing
  {
      static atd::model& get_item(atd::model_map_t& container, bp::tuple i_)
      {
          atd::expression_type expression = extract_template_type(i_[0]);
          atd::numeric_type dtype = extract_dtype(i_[1]);
          atd::model_map_t::iterator i = container.find(std::make_pair(expression, dtype));
          if (i == container.end())
          {
              PyErr_SetString(PyExc_KeyError, "Invalid key");
              bp::throw_error_already_set();
          }
          return *i->second;
      }

      static void set_item(atd::model_map_t& container, bp::tuple i_, atd::model const & v)
      {
          atd::expression_type expression = extract_template_type(i_[0]);
          atd::numeric_type dtype = extract_dtype(i_[1]);
          container[std::make_pair(expression, dtype)].reset(new atd::model(v));
      }
  };

  cl::Platform get_platform(cl::Device const & device)
  {  return cl::Platform(device.getInfo<CL_DEVICE_PLATFORM>());  }

  template<cl_int INFO>
  typename cl::detail::param_traits<cl::detail::cl_device_info, INFO>::param_type
  wrap_device_info(cl::Device const & x)
  { return x.getInfo<INFO>(NULL); }

  template<cl_int INFO>
  typename cl::detail::param_traits<cl::detail::cl_context_info, INFO>::param_type
  wrap_context_info(cl::Context const & x)
  { return x.getInfo<INFO>(NULL); }

  template<cl_int INFO>
  typename cl::detail::param_traits<cl::detail::cl_platform_info, INFO>::param_type
  wrap_platform_info(cl::Platform const & x)
  { return x.getInfo<INFO>(NULL); }

  template<cl_int INFO>
  typename cl::detail::param_traits<cl::detail::cl_command_queue_info, INFO>::param_type
  wrap_command_queue_info(cl::CommandQueue const & x)
  { return x.getInfo<INFO>(NULL); }

  template<cl_int INFO>
  typename cl::detail::param_traits<cl::detail::cl_profiling_info, INFO>::param_type
  wrap_profiling_info(cl::Event const & x)
  { return x.getProfilingInfo<INFO>(NULL); }

  std::string to_string(cl_device_type type)
  {
    if(type==CL_DEVICE_TYPE_ALL) return "ALL";
    if(type==CL_DEVICE_TYPE_CPU) return "CPU";
    if(type==CL_DEVICE_TYPE_GPU) return "GPU";
    if(type==CL_DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
    throw;
  }

  boost::shared_ptr<cl::Context> make_context(cl::Device const & dev)
  { return boost::shared_ptr<cl::Context>(new cl::Context(std::vector<cl::Device>(1, dev))); }

  bp::tuple flush(atd::array_expression const & expression, unsigned int queue_id, bp::list dependencies, bool tune, int label, std::string const & program_name, bool force_recompile)
  {
      std::list<cl::Event> events;
      atd::operation_cache cache;
      std::vector<cl::Event> cdependencies = to_vector<cl::Event>(dependencies);
      boost::shared_ptr<atd::array> parray(new atd::array(atd::control(expression, atd::execution_options_type(queue_id, &events, &cache, &cdependencies),
                                                                       atd::dispatcher_options_type(tune, label), atd::compilation_options_type(program_name, force_recompile))));
      return bp::make_tuple(parray, to_list(events.begin(), events.end()), cache);
  }
}

struct state_type{ };
state_type state;

void export_cl()
{
  typedef std::vector<cl::CommandQueue> queues_t;
  bp::class_<queues_t>("queues")
      .def("__len__", &queues_t::size)
      .def("__getitem__", &bp::vector_indexing_suite<queues_t>::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &bp::vector_indexing_suite<queues_t>::set_item, bp::with_custodian_and_ward<1,2>())
      .def("append", &bp::vector_indexing_suite<queues_t>::append)

      ;

  bp::class_<atd::model_map_t>("models")
      .def("__getitem__", &detail::model_map_indexing::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &detail::model_map_indexing::set_item, bp::with_custodian_and_ward<1,2>())
      ;

  bp::enum_<cl_device_type>("device_type")
      .value("CL_DEVICE_TYPE_ALL", CL_DEVICE_TYPE_ALL)
      .value("CL_DEVICE_TYPE_CPU", CL_DEVICE_TYPE_CPU)
      .value("CL_DEVICE_TYPE_GPU", CL_DEVICE_TYPE_GPU)
      .value("CL_DEVICE_TYPE_ACCELERATOR", CL_DEVICE_TYPE_ACCELERATOR)
      ;

  bp::def("device_type_to_string", &detail::to_string);


  bp::class_<cl::Platform>("platform", bp::no_init)
    #define WRAP(PYNAME, NAME) .add_property(PYNAME, &detail::wrap_platform_info<NAME>)
      WRAP("name", CL_PLATFORM_NAME)
    #undef WRAP
      .def("get_devices", &detail::get_devices)
      ;

  bp::class_<cl::Device>("device", bp::no_init)
    #define WRAP(PYNAME, NAME) .add_property(PYNAME, &detail::wrap_device_info<NAME>)
      .add_property("nv_compute_capability", &detail::nv_compute_capability)
      .add_property("platform", &detail::get_platform)
      WRAP("double_fp_config", CL_DEVICE_DOUBLE_FP_CONFIG)
      WRAP("name", CL_DEVICE_NAME)
      WRAP("type", CL_DEVICE_TYPE)
      WRAP("vendor", CL_DEVICE_VENDOR)
    #undef WRAP
      ;

  bp::class_<cl::Context>("context", bp::no_init)
      .def("__init__", bp::make_constructor(&detail::make_context))
    #define WRAP(PYNAME, NAME) .add_property(PYNAME, &detail::wrap_context_info<NAME>)
    #undef WRAP
      .add_property("queues", bp::make_function(static_cast<std::vector<cl::CommandQueue> & (*)(const cl::Context&)>( [](const cl::Context & ctx) -> std::vector<cl::CommandQueue> & { return atd::cl_ext::queues[ctx]; }) , bp::return_internal_reference<>()))
      ;

  bp::class_<cl::CommandQueue>("command_queue", bp::init<cl::Context, cl::Device>())
    #define WRAP(PYNAME, NAME) .add_property(PYNAME, &detail::wrap_command_queue_info<NAME>)
      WRAP("device", CL_QUEUE_DEVICE)
    #undef WRAP
      .add_property("models", bp::make_function(&atd::get_model_map, bp::return_internal_reference<>()));
      ;

  bp::class_<cl::Event>("event")
    #define WRAP(PYNAME, NAME) .add_property(PYNAME, &detail::wrap_profiling_info<NAME>)
      WRAP("start", CL_PROFILING_COMMAND_START)
      WRAP("submit", CL_PROFILING_COMMAND_SUBMIT)
      WRAP("end", CL_PROFILING_COMMAND_END)
     ;

  bp::class_<atd::operation_cache>("operation_cache", bp::no_init)
      .def("enqueue", &atd::operation_cache::enqueue)
      ;

  bp::def("synchronize", &atd::cl_ext::synchronize);
  bp::def("get_platforms", &detail::get_platforms);

  bp::def("flush", &detail::flush, (bp::arg("expression"), bp::arg("queue_id") = 0, bp::arg("dependencies")=bp::list(), bp::arg("tune") = false, bp::arg("label")=-1, bp::arg("program_name")="", bp::arg("recompile") = false));

  bp::class_<state_type>("state_type")
          .def_readwrite("queue_properties",&atd::cl_ext::queue_properties)
      ;

  bp::scope().attr("state") = bp::object(bp::ptr(&state));

  bp::scope().attr("CL_QUEUE_PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
  bp::scope().attr("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE") = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}

namespace detail
{
  boost::shared_ptr<atd::array>
  ndarray_to_atdarray(const np::ndarray& array, const cl::Context& ctx)
  {

    int d = array.get_nd();
    if (d > 2) {
      PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
      bp::throw_error_already_set();
    }

    atd::numeric_type dtype = to_atd_dtype(array.get_dtype());
    atd::int_t size = (atd::int_t)array.shape(0);
    atd::array* v = new atd::array(size, dtype, ctx);

    void* data = (void*)array.get_data();
    atd::copy(data, *v);

    return boost::shared_ptr<atd::array>(v);
  }



  boost::shared_ptr<atd::array> create_array(bp::object const & obj, bp::object odtype, cl::Context context)
  {
    return ndarray_to_atdarray(np::from_object(obj, to_np_dtype(extract_dtype(odtype))), context);
  }

  boost::shared_ptr<atd::array> create_empty_array(bp::object sizes, bp::object odtype, cl::Context context)
  {
      typedef boost::shared_ptr<atd::array> result_type;

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

      atd::numeric_type dtype = extract_dtype(odtype);
      if(len < 1 || len > 2)
      {
          PyErr_SetString(PyExc_TypeError, "Only 1-D and 2-D arrays are supported!");
          bp::throw_error_already_set();
      }
      if(len==1)
          return result_type(new atd::array(size1, dtype, context));
      return result_type(new atd::array(size1, size2, dtype, context));
  }

  std::string type_name(bp::object const & obj)
  {
    std::string name = bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
    if(name=="class")
      return bp::extract<std::string>(obj.attr("__name__"))();
    else
      return bp::extract<std::string>(obj.attr("__class__").attr("__name__"))();
  }

  boost::shared_ptr<atd::scalar> construct_scalar(bp::object obj, cl::Context const & context)
  {
    typedef boost::shared_ptr<atd::scalar> result_type;
    std::string name = type_name(obj);
    if(name=="int") return result_type(new atd::scalar(bp::extract<int>(obj)(), context));
    else if(name=="float") return result_type(new atd::scalar(bp::extract<double>(obj)(), context));
    else if(name=="long") return result_type(new atd::scalar(bp::extract<long>(obj)(), context));
    else if(name=="int") return result_type(new atd::scalar(bp::extract<int>(obj)(), context));

    else if(name=="int8") return result_type(new atd::scalar(atd::CHAR_TYPE, context));
    else if(name=="uint8") return result_type(new atd::scalar(atd::UCHAR_TYPE, context));
    else if(name=="int16") return result_type(new atd::scalar(atd::SHORT_TYPE, context));
    else if(name=="uint16") return result_type(new atd::scalar(atd::USHORT_TYPE, context));
    else if(name=="int32") return result_type(new atd::scalar(atd::INT_TYPE, context));
    else if(name=="uint32") return result_type(new atd::scalar(atd::UINT_TYPE, context));
    else if(name=="int64") return result_type(new atd::scalar(atd::LONG_TYPE, context));
    else if(name=="uint64") return result_type(new atd::scalar(atd::ULONG_TYPE, context));
    else if(name=="float32") return result_type(new atd::scalar(atd::FLOAT_TYPE, context));
    else if(name=="float64") return result_type(new atd::scalar(atd::DOUBLE_TYPE, context));
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
  .def(bp::self                                    OP bp::other<atd::value_scalar>())\
  .def(int()                                       OP bp::self)\
  .def(long()                                      OP bp::self)\
  .def(double()                                     OP bp::self)\
  .def(bp::other<atd::value_scalar>()              OP bp::self)

#define ADD_ARRAY_OPERATOR(OP)\
  .def(bp::self OP bp::self)\
  ADD_SCALAR_HANDLING(OP)

  bp::class_<atd::expressions_tuple>
      ("array_expression_container", bp::init<atd::array_expression const &>())
  ;

  bp::class_<atd::array_expression >("array_expression", bp::no_init)
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
      .add_property("context", bp::make_function(&atd::array_expression::context, bp::return_internal_reference<>()))
      .def(bp::self_ns::abs(bp::self))
//      .def(bp::self_ns::pow(bp::self))
  ;
#undef ADD_ARRAY_OPERATOR

#define ADD_ARRAY_OPERATOR(OP) \
  .def(bp::self                            OP bp::self)\
  .def(bp::self                            OP bp::other<atd::array_expression>())\
  .def(bp::other<atd::array_expression>() OP bp::self) \
  ADD_SCALAR_HANDLING(OP)

  bp::class_<atd::array,
          boost::shared_ptr<atd::array> >
  ( "array", bp::no_init)
      .def("__init__", bp::make_constructor(detail::create_array, bp::default_call_policies(), (bp::arg("obj"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=atd::cl_ext::default_context())))
      .def(bp::init<atd::array_expression>())
      .add_property("dtype", &atd::array::dtype)
      .add_property("context", bp::make_function(&atd::array::context, bp::return_internal_reference<>()))
      .add_property("T", &atd::array::T)
//      .add_property("shape", &detail::get_shape, &detail::set_shape)
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

  bp::class_<atd::scalar, bp::bases<atd::array> >
      ("scalar", bp::no_init)
      .def("__init__", bp::make_constructor(detail::construct_scalar, bp::default_call_policies(), (bp::arg(""), bp::arg("context")=atd::cl_ext::default_context())))
      ;

  //Other numpy-like initializers
  bp::def("empty", &detail::create_empty_array, (bp::arg("shape"), bp::arg("dtype") = bp::scope().attr("float32"), bp::arg("context")=atd::cl_ext::default_context()));

//Binary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::array const &)>(&atd::name));\
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::array const &)>(&atd::name));\
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::array_expression const &)>(&atd::name));\
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::array_expression const &)>(&atd::name));

  MAP_FUNCTION(maximum)
  MAP_FUNCTION(minimum)
  MAP_FUNCTION(pow)
  MAP_FUNCTION(dot)
#undef MAP_FUNCTION

//Unary
#define MAP_FUNCTION(name) \
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &)>(&atd::name));\
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &)>(&atd::name));

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
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array const &, atd::int_t)>(&atd::name));\
      bp::def(#name, static_cast<atd::array_expression (*)(atd::array_expression const &, atd::int_t)>(&atd::name));

  MAP_FUNCTION(sum)
  MAP_FUNCTION(max)
  MAP_FUNCTION(min)
  MAP_FUNCTION(argmax)
  MAP_FUNCTION(argmin)
#undef MAP_FUNCTION
}

void export_scalar()
{
  bp::class_<atd::value_scalar>("value_scalar", bp::no_init)
          .add_property("dtype", &atd::value_scalar::dtype);
}


void export_model()
{

  bp::class_<atidlas::model>("model", bp::init<atd::base const &, cl::CommandQueue&>())
                  .def("execute", &atd::model::execute);
  
  bp::enum_<atidlas::fetching_policy_type>
      ("fetching_policy_type")
      .value("FETCH_FROM_LOCAL", atd::FETCH_FROM_LOCAL)
      .value("FETCH_FROM_GLOBAL_STRIDED", atd::FETCH_FROM_GLOBAL_STRIDED)
      .value("FETCH_FROM_GLOBAL_CONTIGUOUS", atd::FETCH_FROM_GLOBAL_CONTIGUOUS)
      ;

  //Base
  {
    #define __PROP(name) .def_readonly(#name, &atidlas::base::parameters_type::name)
    bp::class_<atidlas::base, boost::noncopyable>("base", bp::no_init)
            .def("lmem_usage", &atidlas::base::lmem_usage)
            .def("registers_usage", &atidlas::base::registers_usage)
            .def("check_invalid", &atidlas::base::check_invalid)
        ;
    #undef __PROP
  }

  #define WRAP_BASE(name) bp::class_<atidlas::base_impl<atidlas::name, atidlas::name::parameters_type>, bp::bases<atidlas::base>, boost::noncopyable>(#name "_base_impl", bp::no_init);
  #define WRAP_TEMPLATE(name, basename, ...) bp::class_<atidlas::name, bp::bases<atidlas::base_impl<atidlas::basename, atidlas::basename::parameters_type> > >(#name, bp::init<__VA_ARGS__>())\
                                      .add_property("local_size_0", &atd::name::local_size_0)\
                                      .add_property("local_size_1", &atd::name::local_size_1);
  #define WRAP_SINGLE_TEMPLATE(name, ...) WRAP_BASE(name) WRAP_TEMPLATE(name, name, __VA_ARGS__)

  //Vector AXPY
  WRAP_SINGLE_TEMPLATE(vaxpy, uint, uint, uint, atidlas::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(maxpy, uint, uint, uint, uint, uint, atidlas::fetching_policy_type)
  WRAP_SINGLE_TEMPLATE(reduction, uint, uint, uint, atidlas::fetching_policy_type)
  WRAP_BASE(mreduction)
  WRAP_TEMPLATE(mreduction_rows, mreduction, uint, uint, uint, uint, uint, atidlas::fetching_policy_type)
  WRAP_TEMPLATE(mreduction_cols, mreduction, uint, uint, uint, uint, uint, atidlas::fetching_policy_type)
  WRAP_BASE(mproduct)
  WRAP_TEMPLATE(mproduct_nn, mproduct, uint, uint, uint, uint, uint, uint, uint, atidlas::fetching_policy_type, atidlas::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tn, mproduct, uint, uint, uint, uint, uint, uint, uint, atidlas::fetching_policy_type, atidlas::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_nt, mproduct, uint, uint, uint, uint, uint, uint, uint, atidlas::fetching_policy_type, atidlas::fetching_policy_type, uint, uint)
  WRAP_TEMPLATE(mproduct_tt, mproduct, uint, uint, uint, uint, uint, uint, uint, atidlas::fetching_policy_type, atidlas::fetching_policy_type, uint, uint)


}

BOOST_PYTHON_MODULE(_atidlas)
{
  Py_Initialize();
  np::initialize();

  // specify that this module is actually a package
  bp::object package = bp::scope();
  package.attr("__path__") = "_atidlas";

  export_scalar();
  export_core();
  export_cl();
  export_model();
  export_array();
}
