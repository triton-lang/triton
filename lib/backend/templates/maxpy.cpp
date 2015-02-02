#include "atidlas/backend/templates/maxpy.h"
#include "atidlas/tools/make_map.hpp"
#include "atidlas/tools/make_vector.hpp"
#include "atidlas/symbolic/io.h"
#include <iostream>

namespace atidlas
{

maxpy_parameters::maxpy_parameters(unsigned int _simd_width,
                          unsigned int _local_size_0, unsigned int _local_size_1,
                          unsigned int _num_groups_0, unsigned int _num_groups_1,
                          fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1), num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetching_policy(_fetching_policy){ }



int maxpy::check_invalid_impl(cl::Device const &, expressions_tuple const &) const
{
  if (p_.simd_width>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  if(p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string maxpy::generate_impl(unsigned int label, expressions_tuple const & expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width) const
{
  kernel_generation_stream stream;

  std::string init0, upper_bound0, inc0, init1, upper_bound1, inc1;
  std::string data_type = append_width("#scalartype",simd_width);
  char kprefix[10];
  fill_kernel_name(kprefix, label, "d");

  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
  stream << "__kernel void " << kprefix << "(unsigned int M, unsigned int N, " << generate_arguments("#scalartype", mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array0", "#scalartype #namereg = #pointer[#start];")
                                                                                        ("array1", "#pointer += #start;")
                                                                                        ("array2", "#pointer = &$VALUE{#start1, #start2};"), expressions, mappings);

  fetching_loop_info(p_.fetching_policy, "M", stream, init0, upper_bound0, inc0, "get_global_id(0)", "get_global_size(0)");
  stream << "for(unsigned int i = " << init0 << "; i < " << upper_bound0 << "; i += " << inc0 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  fetching_loop_info(p_.fetching_policy, "N", stream, init1, upper_bound1, inc1, "get_global_id(1)", "get_global_size(1)");
  stream << "for(unsigned int j = " << init1 << "; j < " << upper_bound1 << "; j += " << inc1 << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >
                                                                          ("array2", data_type + " #namereg = $VALUE{i*#stride1,j*#stride2};")
                                                                          ("vdiag", "#scalartype #namereg = ((i + ((#diag_offset<0)?#diag_offset:0))!=(j-((#diag_offset>0)?#diag_offset:0)))?0:$VALUE{min(i*#stride1, j*#stride1)};")
                                                                          ("repeat", "#scalartype #namereg = $VALUE{(i%#tuplearg0)*#stride1, (j%#tuplearg1)*#stride2};")
                                                                          ("outer", "#scalartype #namereg = ($LVALUE{i*#stride})*($RVALUE{j*#stride});")
           , expressions, mappings);

  evaluate(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >
                                                                            ("array2", "#namereg")
                                                                            ("vdiag", "#namereg")
                                                                            ("repeat", "#namereg")
                                                                            ("array0", "#namereg")
                                                                            ("outer", "#namereg")
                                                                            ("cast", "convert_"+data_type)
                                                  , expressions, mappings);

  process(stream, LHS_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array2", "$VALUE{i*#stride1,j*#stride2} = #namereg;")
                                             , expressions, mappings);

  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;
  return stream.str();
}

std::vector<std::string> maxpy::generate_impl(unsigned int label, expressions_tuple const & expressions, std::vector<mapping_type> const & mappings) const
{
  std::vector<std::string> res;
  res.push_back(generate_impl(label, expressions, mappings, 1));
  return res;
}

maxpy::maxpy(parameters_type const & parameters, binding_policy_t binding_policy) :
  base_impl<maxpy, maxpy_parameters>(parameters, binding_policy){ }

maxpy::maxpy(unsigned int simd, unsigned int ls1, unsigned int ls2,
                               unsigned int ng1, unsigned int ng2, fetching_policy_type fetch,
                               binding_policy_t bind):
    base_impl<maxpy, maxpy_parameters>(maxpy_parameters(simd, ls1, ls2, ng1, ng2, fetch), bind)
{}

std::vector<int_t> maxpy::input_sizes(expressions_tuple const & expressions)
{
  atidlas::array_expression const & array_expression = *(expressions.data().front());
  std::pair<int_t, int_t> size = matrix_size(lhs_most(array_expression.tree(), array_expression.root()));
  return tools::make_vector<int_t>() << size.first << size.second;
}

void maxpy::enqueue(cl::CommandQueue & queue, std::vector<cl_ext::lazy_compiler> & programs,
                    unsigned int label, expressions_tuple const & expressions, operation_cache * cache)
{
  char kname[10];
  fill_kernel_name(kname, label, "d");
  cl::Program & program = programs[0].program();
  cl::Kernel kernel(program, kname);
  cl::NDRange grange(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1);
  cl::NDRange lrange(p_.local_size_0, p_.local_size_1);
  unsigned int current_arg = 0;
  std::vector<int_t> MN = input_sizes(expressions);
  kernel.setArg(current_arg++, cl_uint(MN[0]));
  kernel.setArg(current_arg++, cl_uint(MN[1]));
  set_arguments(expressions, kernel, current_arg);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, grange, lrange);

  if(cache)
    cache->push_back(queue, kernel, cl::NullRange, grange, lrange);
}

template class base_impl<maxpy, maxpy_parameters>;

}
