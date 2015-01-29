#include "atidlas/backend/templates/vaxpy.h"
#include "atidlas/cl/queues.h"
#include "atidlas/tools/make_map.hpp"
#include "atidlas/tools/make_vector.hpp"
#include "atidlas/tools/to_string.hpp"
#include <iostream>

namespace atidlas
{


vaxpy_parameters::vaxpy_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy) :
      base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), fetching_policy(_fetching_policy)
{ }


int vaxpy::check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::vector<std::string> vaxpy::generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const
{
  std::vector<std::string> result;
  for (unsigned int i = 0; i < 2; ++i)
  {
    kernel_generation_stream stream;
    unsigned int simd_width = (i==0)?1:p_.simd_width;
    std::string str_simd_width = tools::to_string(simd_width);
    std::string data_type = append_width("#scalartype",simd_width);

    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
    char kprefix[10];
    fill_kernel_name(kprefix, label, (i==0?"f":"o"));
    stream << "__kernel void " << kprefix << "(unsigned int N," << generate_arguments(data_type, mappings, symbolic_expressions) << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    process(stream, PARENT_NODE_TYPE,
                          tools::make_map<std::map<std::string, std::string> >("array0", "#scalartype #namereg = #pointer[#start];")
                                                                     ("array1", "#pointer += #start;")
                                                                     ("array1", "#start1/=" + str_simd_width + ";"), symbolic_expressions, mappings);

    std::string init, upper_bound, inc;
    fetching_loop_info(p_.fetching_policy, "N/"+str_simd_width, stream, init, upper_bound, inc, "get_global_id(0)", "get_global_size(0)");
    stream << "for(unsigned int i = " << init << "; i < " << upper_bound << "; i += " << inc << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    process(stream, PARENT_NODE_TYPE,
                          tools::make_map<std::map<std::string, std::string> >("array1", data_type + " #namereg = #pointer[i*#stride];")
                                                                     ("matrix_row", "#scalartype #namereg = $VALUE{#row*#stride1, i*#stride2};")
                                                                     ("matrix_column", "#scalartype #namereg = $VALUE{i*#stride1,#column*#stride2};")
                                                                     ("matrix_diag", "#scalartype #namereg = #pointer[#diag_offset<0?$OFFSET{(i - #diag_offset)*#stride1, i*#stride2}:$OFFSET{i*#stride1, (i + #diag_offset)*#stride2}];")
                                                                     , symbolic_expressions, mappings);

    evaluate(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array1", "#namereg")
                                                                                                ("matrix_row", "#namereg")
                                                                                                ("matrix_column", "#namereg")
                                                                                                ("matrix_diag", "#namereg")
                                                                                                ("array0", "#namereg")
                                                                                                ("cast", "convert_"+data_type)
             , symbolic_expressions, mappings);

    process(stream, LHS_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array1", "#pointer[i*#stride] = #namereg;")
                                                                                           ("matrix_row", "$VALUE{#row, i} = #namereg;")
                                                                                           ("matrix_column", "$VALUE{i, #column} = #namereg;")
                                                                                           ("matrix_diag", "#diag_offset<0?$VALUE{(i - #diag_offset)*#stride1, i*#stride2}:$VALUE{i*#stride1, (i + #diag_offset)*#stride2} = #namereg;")
                                                                                           ,symbolic_expressions, mappings);

    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "if(get_global_id(0)==0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    process(stream, LHS_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array0", "#pointer[#start] = #namereg;"), symbolic_expressions, mappings);
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;

//    std::cout << stream.str() << std::endl;
    result.push_back(stream.str());
  }

  return result;
}

vaxpy::vaxpy(vaxpy_parameters const & parameters,
                               binding_policy_t binding_policy) :
    base_impl<vaxpy, vaxpy_parameters>(parameters, binding_policy)
{}

vaxpy::vaxpy(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, binding_policy_t bind):
    base_impl<vaxpy, vaxpy_parameters>(vaxpy_parameters(simd,ls,ng,fetch), bind)
{}


std::vector<int_t> vaxpy::input_sizes(symbolic_expressions_container const & symbolic_expressions)
{
  int_t size = static_cast<array_expression const *>(symbolic_expressions.data().front().get())->shape()._1;
  return tools::make_vector<int_t>() << size;
}

void vaxpy::enqueue(cl::CommandQueue & queue,
             std::vector<cl_ext::lazy_compiler> & programs,
             unsigned int label,
             symbolic_expressions_container const & symbolic_expressions)
{
  //Size
  int_t size = input_sizes(symbolic_expressions)[0];
  //Kernel
  char kfb[10];
  char kopt[10];
  fill_kernel_name(kfb, label, "f");
  fill_kernel_name(kopt, label, "o");
  bool fallback = p_.simd_width > 1 && (requires_fallback(symbolic_expressions) || (size%p_.simd_width>0));

  cl::Program const & program = programs[fallback?0:1].program();
  cl_ext::kernels_t::key_type key(program(), label);
  cl_ext::kernels_t::iterator it = cl_ext::kernels.find(key);
  if(it==cl_ext::kernels.end())
    it = cl_ext::kernels.insert(std::make_pair(key, cl::Kernel(program, fallback?kfb:kopt))).first;
  cl::Kernel & kernel = it->second;

  //NDRange
  cl::NDRange grange(p_.local_size_0*p_.num_groups);
  cl::NDRange lrange(p_.local_size_0);
  //Arguments
  unsigned int current_arg = 0;
  kernel.setArg(current_arg++, cl_uint(size));
  set_arguments(symbolic_expressions, kernel, current_arg);
  queue.enqueueNDRangeKernel(kernel, cl::NullRange, grange, lrange);
}


}
