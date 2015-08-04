#include <memory>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "isaac/model/model.h"
#include "isaac/symbolic/execute.h"

#include "common.hpp"
#include "driver.h"

namespace detail
{

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
    std::vector<isc::driver::Platform> platforms;
    isc::driver::backend::platforms(platforms);
    return tools::to_list(platforms.begin(), platforms.end());
  }

  bp::list get_devices(isc::driver::Platform const & platform)
  {
    std::vector<isc::driver::Device> devices;
    platform.devices(devices);
    return tools::to_list(devices.begin(), devices.end());
  }

  bp::list get_queues(isc::driver::Context const & context)
  {
    std::vector<isc::driver::CommandQueue*> queues;
    isc::driver::backend::queues::get(context, queues);
    bp::list res;
    for(isc::driver::CommandQueue* queue:queues)
        res.append(*queue);
    return res;
  }

  std::shared_ptr< isc::driver::CommandQueue> create_queue(isc::driver::Context const & context, isc::driver::Device const & device)
  {
      return std::shared_ptr<isc::driver::CommandQueue>(new isc::driver::CommandQueue(context, device));
  }

  struct model_map_indexing
  {
      static isc::model& get_item(isc::model_map_t& container, bp::tuple i_)
      {
          isc::expression_type expression = tools::extract_template_type(i_[0]);
          isc::numeric_type dtype = tools::extract_dtype(i_[1]);
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
          isc::expression_type expression = tools::extract_template_type(i_[0]);
          isc::numeric_type dtype = tools::extract_dtype(i_[1]);
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

  bp::object enqueue(isc::array_expression const & expression, unsigned int queue_id, bp::list dependencies, bool tune, int label, std::string const & program_name, bool force_recompile)
  {
      std::list<isc::driver::Event> events;
      std::vector<isc::driver::Event> cdependencies = tools::to_vector<isc::driver::Event>(dependencies);

      isc::execution_options_type execution_options(queue_id, &events, &cdependencies);
      isc::dispatcher_options_type dispatcher_options(tune, label);
      isc::compilation_options_type compilation_options(program_name, force_recompile);
      isc::array_expression::container_type::value_type root = expression.tree()[expression.root()];
      if(isc::detail::is_assignment(root.op))
      {
          isc::execute(isc::control(expression, execution_options, dispatcher_options, compilation_options), isaac::models(execution_options.queue(expression.context())));
          return bp::make_tuple(bp::ptr(root.lhs.array), tools::to_list(events.begin(), events.end()));
      }
      else
      {
          std::shared_ptr<isc::array> parray(new isc::array(isc::control(expression, execution_options, dispatcher_options, compilation_options)));
          return bp::make_tuple(parray, tools::to_list(events.begin(), events.end()));
      }
  }
}

struct state_type{ };
state_type state;

void export_driver()
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

  bp::class_<isc::driver::Context, boost::noncopyable>("context", bp::no_init)
      .def("__init__", bp::make_constructor(&detail::make_context))
      .add_property("queues", &detail::get_queues)
      .add_property("backend", &isc::driver::Context::backend)
      ;

  bp::class_<isc::driver::CommandQueue>("command_queue", bp::init<isc::driver::Context const &, isc::driver::Device const &>())
      .def("synchronize", &isc::driver::CommandQueue::synchronize)
      .add_property("models", bp::make_function(&isc::models, bp::return_internal_reference<>()))
      .add_property("device", bp::make_function(&isc::driver::CommandQueue::device, bp::return_internal_reference<>()))
      ;

  bp::class_<isc::driver::Event>("event", bp::init<isc::driver::backend_type>())
      .add_property("elapsed_time", &isc::driver::Event::elapsed_time)
     ;

  bp::def("device_type_to_string", &detail::to_string);

  bp::def("get_platforms", &detail::get_platforms);

  bp::def("enqueue", &detail::enqueue, (bp::arg("expression"), bp::arg("queue_id") = 0, bp::arg("dependencies")=bp::list(), bp::arg("tune") = false, bp::arg("label")=-1, bp::arg("program_name")="", bp::arg("recompile") = false));

  bp::class_<state_type>("state_type")
          .def_readwrite("queue_properties",&isc::driver::backend::queue_properties)
      ;

  bp::scope().attr("state") = bp::object(bp::ptr(&state));

  bp::scope().attr("CL_QUEUE_PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
  bp::scope().attr("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE") = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}
