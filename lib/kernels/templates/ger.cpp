#include <cstring>
#include <iostream>
#include "isaac/kernels/templates/ger.h"
#include "isaac/symbolic/io.h"
#include "isaac/kernels/keywords.h"

#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/vector_types.hpp"

namespace isaac
{
namespace templates
{

ger_parameters::ger_parameters(unsigned int _simd_width,
                          unsigned int _local_size_0, unsigned int _local_size_1,
                          unsigned int _num_groups_0, unsigned int _num_groups_1,
                          fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }



int ger::is_invalid_impl(driver::Device const &, expressions_tuple const &) const
{
  if (p_.simd_width>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  if(p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string ger::generate_impl(std::string const & suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const
{
  kernel_generation_stream stream;
  std::string _size_t = size_type(device);
  std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
  std::string data_type = append_width("#scalartype",p_.simd_width);
  driver::backend_type backend = device.backend();

  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void axpy" << suffix << "(" << _size_t << " M, " << _size_t << " N, " << generate_arguments("#scalartype", device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE, { {"array0", "#scalartype #namereg = #pointer[#start];"},
                                      {"array1", "#pointer += #start;"},
                                      {"array2", "#pointer += #start;"}}
                                  , expressions, mappings);

  fetching_loop_info(p_.fetching_policy, "M", stream, init0, upper_bound0, inc0,  GlobalIdx0(backend).get(), GlobalSize0(backend).get(), device);
  stream << "for(" << _size_t << " i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  fetching_loop_info(p_.fetching_policy, "N", stream, init1, upper_bound1, inc1, GlobalIdx1(backend).get(), GlobalSize1(backend).get(), device);
  stream << "for(" << _size_t << " j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE, { {"array2", data_type + " #namereg = $VALUE{i*#stride,j};"},
                                      {"vdiag", "#scalartype #namereg = ((i + ((#diag_offset<0)?#diag_offset:0))!=(j-((#diag_offset>0)?#diag_offset:0)))?0:$VALUE{min(i*#stride, j*#stride)};"},
                                      {"repeat", "#scalartype #namereg = $VALUE{(i%#tuplearg0)*#stride, (j%#tuplearg1)};"},
                                      {"outer", "#scalartype #namereg = ($LVALUE{i*#stride})*($RVALUE{j*#stride});"} }
                                    , expressions, mappings);

  evaluate(stream, PARENT_NODE_TYPE, { {"array2", "#namereg"},
                                        {"vdiag", "#namereg"},
                                        {"repeat", "#namereg"},
                                        {"array0", "#namereg"},
                                        {"outer", "#namereg"},
                                        {"cast", CastPrefix(backend, data_type).get()},
                                        {"host_scalar", p_.simd_width==1?"#name": InitPrefix(backend, data_type).get() + "(#name)"}}
                                    , expressions, mappings);

  process(stream, LHS_NODE_TYPE, { {"array2", "$VALUE{i*#stride,j} = #namereg;"} } , expressions, mappings);

  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;

  stream << "if(" << GlobalIdx0(backend) << "==0 &&" << GlobalIdx1(backend) << "==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  process(stream, LHS_NODE_TYPE, { {"array0", "#pointer[#start] = #namereg;"} }, expressions, mappings);
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

ger::ger(parameters_type const & parameters, binding_policy_t binding_policy) :
  base_impl<ger, ger_parameters>(parameters, binding_policy){ }

ger::ger(unsigned int simd, unsigned int ls1, unsigned int ls2,
                               unsigned int ng1, unsigned int ng2, fetching_policy_type fetch,
                               binding_policy_t bind):
    base_impl<ger, ger_parameters>(ger_parameters(simd, ls1, ls2, ng1, ng2, fetch), bind)
{}

std::vector<int_t> ger::input_sizes(expressions_tuple const & expressions) const
{
  isaac::array_expression const & array_expression = *(expressions.data().front());
  std::pair<int_t, int_t> size = matrix_size(lhs_most(array_expression.tree(), array_expression.root()));
  return {size.first, size.second};
}

void ger::enqueue(driver::CommandQueue & /*queue*/, driver::Program const & program, std::string const & suffix, base &, controller<expressions_tuple> const & controller)
{
  expressions_tuple const & expressions = controller.x();
  std::string name = "axpy";
  name +=suffix;
  driver::Kernel kernel(program, name.c_str());
  driver::NDRange global(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1);
  driver::NDRange local(p_.local_size_0, p_.local_size_1);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(expressions);
  kernel.setSizeArg(current_arg++, MN[0]);
  kernel.setSizeArg(current_arg++, MN[1]);
  set_arguments(expressions, kernel, current_arg, binding_policy_);

  controller.execution_options().enqueue(program.context(), kernel, global, local);
}

}
}
