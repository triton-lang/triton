#include <memory>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

#include "isaac/driver/device.h"
#include "isaac/symbolic/execute.h"

#include "common.hpp"
#include "driver.h"


namespace detail
{

  bp::list nv_compute_capability(sc::driver::Device const & device)
  {
    bp::list res;
    std::pair<unsigned int, unsigned int> cc = device.nv_compute_capability();
    res.append(cc.first);
    res.append(cc.second);
    return res;
  }

  bp::list get_platforms()
  {
    std::vector<sc::driver::Platform> platforms;
    sc::driver::backend::platforms(platforms);
    return tools::to_list(platforms.begin(), platforms.end());
  }

  bp::list get_devices(sc::driver::Platform const & platform)
  {
    std::vector<sc::driver::Device> devices;
    platform.devices(devices);
    return tools::to_list(devices.begin(), devices.end());
  }

  bp::list get_queues(sc::driver::Context const & context)
  {
    std::vector<sc::driver::CommandQueue*> queues;
    sc::driver::backend::queues::get(context, queues);
    bp::list res;
    for(sc::driver::CommandQueue* queue:queues)
        res.append(*queue);
    return res;
  }

  std::shared_ptr< sc::driver::CommandQueue> create_queue(sc::driver::Context const & context, sc::driver::Device const & device)
  {
      return std::shared_ptr<sc::driver::CommandQueue>(new sc::driver::CommandQueue(context, device));
  }



  std::string to_string(sc::driver::Device::Type type)
  {
    if(type==sc::driver::Device::Type::CPU) return "CPU";
    if(type==sc::driver::Device::Type::GPU) return "GPU";
    if(type==sc::driver::Device::Type::ACCELERATOR) return "ACCELERATOR";
    throw;
  }

  std::shared_ptr<sc::driver::Context> make_context(sc::driver::Device const & dev)
  { return std::shared_ptr<sc::driver::Context>(new sc::driver::Context(dev)); }

  bp::object enqueue(sc::math_expression const & expression, unsigned int queue_id, bp::list dependencies, bool tune, int label, std::string const & program_name, bool force_recompile)
  {
      std::list<sc::driver::Event> events;
      std::vector<sc::driver::Event> cdependencies = tools::to_vector<sc::driver::Event>(dependencies);

      sc::execution_options_type execution_options(queue_id, &events, &cdependencies);
      sc::dispatcher_options_type dispatcher_options(tune, label);
      sc::compilation_options_type compilation_options(program_name, force_recompile);
      sc::math_expression::container_type::value_type root = expression.tree()[expression.root()];
      if(sc::detail::is_assignment(root.op))
      {
          sc::execute(sc::execution_handler(expression, execution_options, dispatcher_options, compilation_options), isaac::profiles::get(execution_options.queue(expression.context())));
          return bp::make_tuple(bp::ptr(root.lhs.array), tools::to_list(events.begin(), events.end()));
      }
      else
      {
          std::shared_ptr<sc::array> parray(new sc::array(sc::execution_handler(expression, execution_options, dispatcher_options, compilation_options)));
          return bp::make_tuple(parray, tools::to_list(events.begin(), events.end()));
      }
  }
}

struct default_driver_values_type{ };
default_driver_values_type default_driver_parameters;

void export_driver()
{
  typedef std::vector<sc::driver::CommandQueue> queues_t;

  bp::object driver_module(bp::handle<>(bp::borrowed(PyImport_AddModule("isaac.driver"))));
  bp::scope().attr("driver") = driver_module;
  bp::scope driver_scope = driver_module;

  bp::class_<queues_t>("queues")
      .def("__len__", &queues_t::size)
      .def("__getitem__", &bp::vector_indexing_suite<queues_t>::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &bp::vector_indexing_suite<queues_t>::set_item, bp::with_custodian_and_ward<1,2>())
      .def("append", &bp::vector_indexing_suite<queues_t>::append)
      ;



  bp::enum_<sc::driver::backend_type>
      ("backend_type")
      .value("OPENCL", sc::driver::OPENCL)
      .value("CUDA", sc::driver::CUDA)
      ;

  bp::enum_<sc::driver::Device::Type>
      ("device_type")
      .value("DEVICE_TYPE_GPU", sc::driver::Device::Type::GPU)
      .value("DEVICE_TYPE_CPU", sc::driver::Device::Type::CPU)
      ;


  bp::class_<sc::driver::Platform>("platform", bp::no_init)
      .def("get_devices", &detail::get_devices)
      .add_property("name",&sc::driver::Platform::name)
      ;

  bp::enum_<isaac::driver::Device::Vendor>
      ("vendor")
      .value("AMD", sc::driver::Device::Vendor::AMD)
      .value("INTEL", sc::driver::Device::Vendor::INTEL)
      .value("NVIDIA", sc::driver::Device::Vendor::NVIDIA)
      .value("UNKNOWN", sc::driver::Device::Vendor::UNKNOWN)
      ;

  bp::class_<sc::driver::Device>("device", bp::no_init)
      .add_property("clock_rate", &sc::driver::Device::clock_rate)
      .add_property("name", &sc::driver::Device::name)
      .add_property("type", &sc::driver::Device::type)
      .add_property("platform", &sc::driver::Device::platform)
      .add_property("vendor", &sc::driver::Device::vendor)
      .add_property("nv_compute_capability", &detail::nv_compute_capability)
      .add_property("infos", &sc::driver::Device::infos)
      ;

  bp::class_<sc::driver::Context, boost::noncopyable>("context", bp::no_init)
      .def("__init__", bp::make_constructor(&detail::make_context))
      .def("synchronize", &sc::driver::backend::synchronize)
      .add_property("queues", &detail::get_queues)
      .add_property("backend", &sc::driver::Context::backend)
      ;

  bp::class_<sc::driver::CommandQueue>("command_queue", bp::init<sc::driver::Context const &, sc::driver::Device const &>())
      .def("synchronize", &sc::driver::CommandQueue::synchronize)
      .add_property("profiles", bp::make_function(&sc::profiles::get, bp::return_internal_reference<>()))
      .add_property("device", bp::make_function(&sc::driver::CommandQueue::device, bp::return_internal_reference<>()))
      ;

  bp::class_<sc::driver::Event>("event", bp::init<sc::driver::backend_type>())
      .add_property("elapsed_time", &sc::driver::Event::elapsed_time)
     ;

  bp::def("device_type_to_string", &detail::to_string);

  bp::def("get_platforms", &detail::get_platforms);

  bp::def("enqueue", &detail::enqueue, (bp::arg("expression"), bp::arg("queue_id") = 0, bp::arg("dependencies")=bp::list(), bp::arg("tune") = false, bp::arg("label")=-1, bp::arg("program_name")="", bp::arg("recompile") = false));

  bp::class_<default_driver_values_type>("default_type")
          .def_readwrite("queue_properties",&sc::driver::backend::default_queue_properties)
          .def_readwrite("device", &sc::driver::backend::default_device)
      ;

  bp::scope().attr("default") = bp::object(bp::ptr(&default_driver_parameters));
  bp::scope().attr("PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
}
