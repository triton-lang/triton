#include <iostream>
#include <cstring>
#include <algorithm>

#include "isaac/kernels/templates/axpy.h"
#include "isaac/kernels/keywords.h"
#include "isaac/driver/backend.h"

#include "tools/loop.hpp"
#include "tools/vector_types.hpp"
#include "tools/arguments.hpp"
#include "isaac/symbolic/io.h"

#include <string>

namespace isaac
{
namespace templates
{

axpy_parameters::axpy_parameters(unsigned int _simd_width,
                       unsigned int _group_size, unsigned int _num_groups,
                       fetching_policy_type _fetching_policy) :
      base::parameters_type(_simd_width, _group_size, 1, 1), num_groups(_num_groups), fetching_policy(_fetching_policy)
{
}


int axpy::is_invalid_impl(driver::Device const &, math_expression const &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

std::string axpy::generate_impl(std::string const & suffix, math_expression const & expressions, driver::Device const & device, mapping_type const & mappings) const
{
  driver::backend_type backend = device.backend();
  std::string _size_t = size_type(device);

  kernel_generation_stream stream;
  std::string str_simd_width = tools::to_string(p_.simd_width);
  std::string dtype = append_width("#scalartype",p_.simd_width);


  std::vector<size_t> assigned_scalar = filter_nodes([](math_expression::node const & node) {
                                                        return  detail::is_assignment(node.op) && node.lhs.subtype==DENSE_ARRAY_TYPE && node.lhs.array->nshape()==0;
                                                        }, expressions, expressions.root(), true);
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"helper_math.h\"" << std::endl; break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void " << "axpy" << suffix << "(" << _size_t << " N," << generate_arguments(dtype, device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE, {{"array0", "#scalartype #namereg = #pointer[#start];"},
                                      {"array1", "#pointer += #start;"}}, expressions, mappings);

  stream << _size_t << " idx = " << GlobalIdx0(backend) << ";" << std::endl;
  stream << _size_t << " gsize = " << GlobalSize0(backend) << ";" << std::endl;

  std::string init, upper_bound, inc;
  fetching_loop_info(p_.fetching_policy, "N/"+str_simd_width, stream, init, upper_bound, inc, "idx", "gsize", device);

  stream << "for(" << _size_t << " i = " << init << "; i < " << upper_bound << "; i += " << inc << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();


  math_expression::container_type const & tree = expressions.tree();
  std::vector<std::size_t> sfors = filter_nodes([](math_expression::node const & node){return node.op.type==OPERATOR_SFOR_TYPE;}, expressions, expressions.root(), true);
//  std::cout << sfors.size() << std::endl;

  for(unsigned int i = 0 ; i < sfors.size() ; ++i)
  {
    std::string info[3];
    int idx =  sfors[i];
    for(int i = 0 ; i < 2 ; ++i){
        idx = tree[idx].rhs.node_index;
        info[i] = evaluate(LHS_NODE_TYPE, {{"placeholder", "#name"}}, expressions, idx, mappings);

    }
    info[2] = evaluate(RHS_NODE_TYPE, {{"placeholder", "#name"}}, expressions, idx, mappings);
    info[0] = info[0].substr(1, info[0].size()-2);
    stream << "for(int " << info[0] << " ; " << info[1] << "; " << info[2] << ")" << std::endl;

//    stream << "int sforidx0 = 0 ;" << std::endl;
  }

  if(sfors.size()){
    stream << "{" << std::endl;
    stream.inc_tab();
  }

  size_t root = expressions.root();
  if(sfors.size())
      root = tree[sfors.back()].lhs.node_index;

  std::vector<std::size_t> assigned = filter_nodes([](math_expression::node const & node){return detail::is_assignment(node.op);}, expressions, root, true);
  std::set<std::string> processed;

  //Declares register to store results
  for(std::size_t idx: assigned)
  {
    process(stream, LHS_NODE_TYPE, {{"array1", dtype + " #namereg;"}, {"matrix_row", "#scalartype #namereg;"},
                                       {"matrix_column", "#scalartype #namereg;"},  {"matrix_diag", "#scalartype #namereg;"}}, expressions, idx, mappings, processed);
  }

  //Fetches to registers
  for(std::size_t idx: assigned)
  {
    std::string array1 = dtype + " #namereg = " + vload(p_.simd_width, "#scalartype", "i*#stride", "#pointer", "1", backend, false) + ";";
    std::string array_access = "#scalartype #namereg = #pointer[#index];";
    std::string matrix_row = dtype + " #namereg = " + vload(p_.simd_width, "#scalartype", "i", "#pointer + #row*#stride", "#ld", backend, false) + ";";
    std::string matrix_column = dtype + " #namereg = " + vload(p_.simd_width, "#scalartype", "i*#stride", "#pointer + #column*#ld", "#stride", backend, false) + ";";
    std::string matrix_diag = dtype + " #namereg = " + vload(p_.simd_width, "#scalartype", "i*(#ld + #stride)", "#pointer + ((#diag_offset<0)?-#diag_offset:(#diag_offset*#ld))", "#ld + #stride", backend, false) + ";";
    process(stream, RHS_NODE_TYPE, {{"array1", array1}, {"matrix_row", matrix_row}, {"matrix_column", matrix_column},
                                    {"matrix_diag", matrix_diag}, {"array_access", array_access}}, expressions, idx, mappings, processed);
  }


  //Compute expressions
  for(std::size_t idx: assigned)
    stream << evaluate(PARENT_NODE_TYPE, {{"array0", "#namereg"}, {"array1", "#namereg"},
                                        {"matrix_row", "#namereg"}, {"matrix_column", "#namereg"}, {"matrix_diag", "#namereg"}, {"array_access", "#namereg"},
                                        {"cast", CastPrefix(backend, dtype).get()}, {"placeholder", "#name"}, {"host_scalar", p_.simd_width==1?"#name": InitPrefix(backend, dtype).get() + "(#name)"}},
                                      expressions, idx, mappings) << ";" << std::endl;

  //Writes back to registers
  processed.clear();
  for(std::size_t idx: assigned)
  {
    std::string array1 = vstore(p_.simd_width, "#scalartype", "#namereg", "i*#stride", "#pointer", "1", backend, false) + ";";
    std::string matrix_row = vstore(p_.simd_width, "#scalartype", "#namereg", "i", "#pointer + #row*#stride", "#ld", backend, false) + ";";
    std::string matrix_column = vstore(p_.simd_width, "#scalartype", "#namereg", "i*#stride", "#pointer + #column*#ld", "#stride", backend, false) + ";";
    std::string matrix_diag = vstore(p_.simd_width, "#scalartype", "#namereg", "i*(#ld + #stride)", "#pointer + (#diag_offset<0)?-#diag_offset:(#diag_offset*#ld)", "#ld + #stride", backend, false) + ";";
    process(stream, LHS_NODE_TYPE, {{"array1", array1}, {"matrix_row", matrix_row}, {"matrix_column", matrix_column}, {"matrix_diag", matrix_diag}}, expressions, idx, mappings, processed);
  }

  if(sfors.size()){
    stream.dec_tab();
    stream << "}" << std::endl;
  }

  stream.dec_tab();
  stream << "}" << std::endl;

  processed.clear();
  if(assigned_scalar.size())
  {
    stream << "if(idx==0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    for(std::size_t idx: assigned)
      process(stream, LHS_NODE_TYPE, { {"array0", "#pointer[#start] = #namereg;"} }, expressions, idx, mappings, processed);
    stream.dec_tab();
    stream << "}" << std::endl;
  }

  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;

  return stream.str();
}

axpy::axpy(axpy_parameters const & parameters,
                               binding_policy_t binding_policy) :
    base_impl<axpy, axpy_parameters>(parameters, binding_policy)
{}

axpy::axpy(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, binding_policy_t bind):
    base_impl<axpy, axpy_parameters>(axpy_parameters(simd,ls,ng,fetch), bind)
{}


std::vector<int_t> axpy::input_sizes(math_expression const & expressions) const
{
  size4 shape = expressions.shape();
  return {std::max(shape[0], shape[1])};
}

void axpy::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, base & fallback, execution_handler const & control)
{
  math_expression const & expressions = control.x();
  //Size
  int_t size = input_sizes(expressions)[0];
  //Fallback
  if(p_.simd_width > 1 && (requires_fallback(expressions) || (size%p_.simd_width>0)))
  {
      fallback.enqueue(queue, program, "fallback", fallback, control);
      return;
  }
  //Kernel
  std::string name = "axpy";
  name += suffix;
  driver::Kernel kernel(program, name.c_str());
  //NDRange
  driver::NDRange global(p_.local_size_0*p_.num_groups);
  driver::NDRange local(p_.local_size_0);
  //Arguments
  unsigned int current_arg = 0;
  kernel.setSizeArg(current_arg++, size);
  set_arguments(expressions, kernel, current_arg, binding_policy_);
  control.execution_options().enqueue(program.context(), kernel, global, local);
}


}
}
