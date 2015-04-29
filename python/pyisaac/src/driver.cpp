#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include "isaac/array.h"
#include "isaac/model/model.h"

#include "common.hpp"
#include "driver.h"


bp::list nv_compute_capability(atd::driver::Device const & device)
{
  bp::list res;
  std::pair<unsigned int, unsigned int> cc = device.nv_compute_capability();
  res.append(cc.first);
  res.append(cc.second);
  return res;
}

bp::list get_platforms()
{
  std::vector<atd::driver::Platform> platforms(atd::driver::Platform::get());
  return detail::to_list(platforms.begin(), platforms.end());
}

bp::list get_devices(atd::driver::Platform const & platform)
{
  std::vector<atd::driver::Device> devices(platform.devices());
  return detail::to_list(devices.begin(), devices.end());
}

struct model_map_indexing
{
    static atd::model& get_item(atd::model_map_t& container, bp::tuple i_)
    {
        atd::expression_type expression = detail::extract_template_type(i_[0]);
        atd::numeric_type dtype = detail::extract_dtype(i_[1]);
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
        atd::expression_type expression = detail::extract_template_type(i_[0]);
        atd::numeric_type dtype = detail::extract_dtype(i_[1]);
        container[std::make_pair(expression, dtype)].reset(new atd::model(v));
    }
};

std::string to_string(atd::driver::device_type type)
{
  if(type==atd::driver::DEVICE_TYPE_CPU) return "CPU";
  if(type==atd::driver::DEVICE_TYPE_GPU) return "GPU";
  if(type==atd::driver::DEVICE_TYPE_ACCELERATOR) return "ACCELERATOR";
  throw;
}

std::shared_ptr<atd::driver::Context> make_context(atd::driver::Device const & dev)
{ return std::shared_ptr<atd::driver::Context>(new atd::driver::Context(dev)); }

bp::tuple flush(atd::array_expression const & expression, unsigned int queue_id, bp::list dependencies, bool tune, int label, std::string const & program_name, bool force_recompile)
{
    std::list<atd::driver::Event> events;
    std::vector<atd::driver::Event> cdependencies = detail::to_vector<atd::driver::Event>(dependencies);
    std::shared_ptr<atd::array> parray(new atd::array(atd::control(expression,
                                                                  atd::execution_options_type(queue_id, &events, &cdependencies),
                                                                  atd::dispatcher_options_type(tune, label),
                                                                  atd::compilation_options_type(program_name, force_recompile))));
    return bp::make_tuple(parray, detail::to_list(events.begin(), events.end()));
}


struct state_type{ };
state_type state;

void export_cl()
{
  typedef std::vector<atd::driver::CommandQueue> queues_t;
  bp::class_<queues_t>("queues")
      .def("__len__", &queues_t::size)
      .def("__getitem__", &bp::vector_indexing_suite<queues_t>::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &bp::vector_indexing_suite<queues_t>::set_item, bp::with_custodian_and_ward<1,2>())
      .def("append", &bp::vector_indexing_suite<queues_t>::append)

      ;

  bp::class_<atd::model_map_t>("models")
      .def("__getitem__", &model_map_indexing::get_item, bp::return_internal_reference<>())
      .def("__setitem__", &model_map_indexing::set_item, bp::with_custodian_and_ward<1,2>())
      ;

  bp::enum_<atd::driver::backend_type>
      ("backend_type")
      .value("OPENCL", atd::driver::OPENCL)
  #ifdef ISAAC_WITH_CUDA
      .value("CUDA", atd::driver::CUDA)
  #endif
      ;

  bp::enum_<atd::driver::device_type>
      ("device_type")
      .value("DEVICE_TYPE_GPU", atd::driver::DEVICE_TYPE_GPU)
      .value("DEVICE_TYPE_CPU", atd::driver::DEVICE_TYPE_CPU)
      ;


  bp::class_<atd::driver::Platform>("platform", bp::no_init)
      .def("get_devices", &get_devices)
      .add_property("name",&atd::driver::Platform::name)
      ;

  bp::enum_<isaac::driver::Device::VENDOR>
      ("vendor")
      .value("AMD", atd::driver::Device::AMD)
      .value("INTEL", atd::driver::Device::INTEL)
      .value("NVIDIA", atd::driver::Device::NVIDIA)
      .value("UNKNOWN", atd::driver::Device::UNKNOWN)
      ;

  bp::class_<atd::driver::Device>("device", bp::no_init)
      .add_property("clock_rate", &atd::driver::Device::clock_rate)
      .add_property("name", &atd::driver::Device::name)
      .add_property("type", &atd::driver::Device::type)
      .add_property("platform", &atd::driver::Device::platform)
      .add_property("vendor", &atd::driver::Device::vendor)
      .add_property("nv_compute_capability", &nv_compute_capability)
      ;

  bp::class_<atd::driver::Context>("context", bp::no_init)
      .def("__init__", bp::make_constructor(&make_context))
      .add_property("queues", bp::make_function(static_cast<std::vector<atd::driver::CommandQueue> & (*)(const atd::driver::Context&)>( [](const atd::driver::Context & ctx) -> std::vector<atd::driver::CommandQueue> & { return atd::driver::queues[ctx]; }) , bp::return_internal_reference<>()))
      .add_property("backend", &atd::driver::Context::backend)
      ;

  bp::class_<atd::driver::CommandQueue>("command_queue", bp::init<atd::driver::Context, atd::driver::Device>())
      .def("synchronize", &atd::driver::CommandQueue::synchronize)
      .add_property("models", bp::make_function(&atd::get_model_map, bp::return_internal_reference<>()))
      .add_property("device", bp::make_function(&atd::driver::CommandQueue::device, bp::return_internal_reference<>()))
      ;

  bp::class_<atd::driver::Event>("event", bp::init<atd::driver::backend_type>())
      .add_property("elapsed_time", &atd::driver::Event::elapsed_time)
     ;

  bp::def("device_type_to_string", &to_string);

  bp::def("get_platforms", &get_platforms);

  bp::def("flush", &flush, (bp::arg("expression"), bp::arg("queue_id") = 0, bp::arg("dependencies")=bp::list(), bp::arg("tune") = false, bp::arg("label")=-1, bp::arg("program_name")="", bp::arg("recompile") = false));

  bp::class_<state_type>("state_type")
          .def_readwrite("queue_properties",&atd::driver::queues.queue_properties)
      ;

  bp::scope().attr("state") = bp::object(bp::ptr(&state));

  bp::scope().attr("CL_QUEUE_PROFILING_ENABLE") = CL_QUEUE_PROFILING_ENABLE;
  bp::scope().attr("CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE") = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
}
