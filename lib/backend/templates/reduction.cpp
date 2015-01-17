#include <iostream>
#include "atidlas/backend/templates/reduction.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/tools/to_string.hpp"
#include "atidlas/tools/make_map.hpp"
#include "atidlas/tools/make_vector.hpp"

namespace atidlas
{

reduction_parameters::reduction_parameters(unsigned int _simd_width,
                     unsigned int _group_size, unsigned int _num_groups,
                     fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _group_size, 1, 2), num_groups(_num_groups), fetching_policy(_fetching_policy)
{ }

unsigned int reduction::lmem_usage(symbolic_expressions_container const & symbolic_expressions) const
{
  unsigned int res = 0;
  for(symbolic_expressions_container::data_type::const_iterator it = symbolic_expressions.data().begin() ; it != symbolic_expressions.data().end() ; ++it)
  {
    numeric_type numeric_t= lhs_most((*it)->tree(), (*it)->root()).lhs.dtype;
    res += p_.local_size_0*size_of(numeric_t);
  }
  return res;
}

int reduction::check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

inline void reduction::reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_reduction*> exprs,
                                   std::string const & buf_str, std::string const & buf_value_str) const
{
  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << size/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream << "if (lid <  stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (unsigned int k = 0; k < exprs.size(); k++)
    if (exprs[k]->is_index_reduction())
      compute_index_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]")
                              , exprs[k]->process(buf_value_str+"[lid]"), exprs[k]->process(buf_value_str+"[lid+stride]"),
                              exprs[k]->root_op());
    else
      compute_reduction(stream, exprs[k]->process(buf_str+"[lid]"), exprs[k]->process(buf_str+"[lid+stride]"), exprs[k]->root_op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
}

std::string reduction::generate_impl(unsigned int label, char type, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width) const
{
  kernel_generation_stream stream;

  std::vector<mapped_scalar_reduction*> exprs;
  for (std::vector<mapping_type>::const_iterator it = mappings.begin(); it != mappings.end(); ++it)
    for (mapping_type::const_iterator iit = it->begin(); iit != it->end(); ++iit)
      if (mapped_scalar_reduction * p = dynamic_cast<mapped_scalar_reduction*>(iit->second.get()))
        exprs.push_back(p);
  std::size_t N = exprs.size();

  std::string arguments = "unsigned int N, ";
  for (unsigned int k = 0; k < N; ++k)
  {
    std::string numeric_type = numeric_type_to_string(lhs_most(exprs[k]->symbolic_expression().tree(),
                                                                      exprs[k]->symbolic_expression().root()).lhs.dtype);
    if (exprs[k]->is_index_reduction())
    {
      arguments += exprs[k]->process("__global unsigned int* #name_temp, ");
      arguments += exprs[k]->process("__global " + tools::to_string(numeric_type) + "* #name_temp_value, ");
    }
    else
      arguments += exprs[k]->process("__global " + tools::to_string(numeric_type) + "* #name_temp, ");
  }


  /* ------------------------
   * First Kernel
   * -----------------------*/
  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
  stream << "__kernel void " << "k" << label << type << "0" << "(" << arguments << generate_arguments("#scalartype", mappings, symbolic_expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "unsigned int lid = get_local_id(0);" << std::endl;
  process(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("scalar", "#scalartype #namereg = *#pointer;")
                                                                                              ("array", "#pointer += #start1;"), symbolic_expressions, mappings);

  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_reduction())
    {
      stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(exprs[k]->root_op()) + ";") << std::endl;
      stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
    }
    else
    {
      stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(exprs[k]->root_op()) + ";") << std::endl;
    }
  }

  class loop_body : public loop_body_base
  {
  public:
    loop_body(std::vector<mapped_scalar_reduction*> const & _exprs) : exprs(_exprs){ }

    void operator()(kernel_generation_stream & stream, unsigned int simd_width) const
    {
      std::string i = (simd_width==1)?"i*#stride1":"i";
      //Fetch vector entry
      for (std::vector<mapped_scalar_reduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
        (*it)->process_recursive(stream, PARENT_NODE_TYPE, tools::make_map<std::map<std::string, std::string> >("array",  append_width("#scalartype",simd_width) + " #namereg = " + vload(simd_width,i,"#pointer")+";")
                                                                                         ("matrix_row",  "#scalartype #namereg = #pointer[$OFFSET{#row*#stride1, i*#stride2}];")
                                                                                         ("matrix_column", "#scalartype #namereg = #pointer[$OFFSET{i*#stride1,#column*#stride2}];")
                                                                                         ("matrix_diag", "#scalartype #namereg = #pointer[#diag_offset<0?$OFFSET{(i - #diag_offset)*#stride1, i*#stride2}:$OFFSET{i*#stride1, (i + #diag_offset)*#stride2}];"));


      //Update accumulators
      std::vector<std::string> str(simd_width);
      if (simd_width==1)
        str[0] = "#namereg";
      else
        for (unsigned int a = 0; a < simd_width; ++a)
          str[a] = append_simd_suffix("#namereg.s", a);

      for (unsigned int k = 0; k < exprs.size(); ++k)
      {
        for (unsigned int a = 0; a < simd_width; ++a)
        {
          std::map<std::string, std::string> accessors;
          accessors["array"] = str[a];
          accessors["matrix_row"] = str[a];
          accessors["matrix_column"] = str[a];
          accessors["matrix_diag"] = str[a];
          accessors["scalar"] = "#namereg";
          std::string value = exprs[k]->evaluate_recursive(LHS_NODE_TYPE, accessors);
          if (exprs[k]->is_index_reduction())
            compute_index_reduction(stream, exprs[k]->process("#name_acc"),  "i*" + tools::to_string(simd_width) + "+"
                                    + tools::to_string(a), exprs[k]->process("#name_acc_value"), value,exprs[k]->root_op());
          else
            compute_reduction(stream, exprs[k]->process("#name_acc"), value,exprs[k]->root_op());
        }
      }
    }

  private:
    std::vector<mapped_scalar_reduction*> exprs;
  };

  element_wise_loop_1D(stream, loop_body(exprs), p_.fetching_policy, simd_width, "i", "N", "get_global_id(0)", "get_global_size(0)");

  //Fills local memory
  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_reduction())
      stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
  }

  //Reduce local memory
  reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value");

  //Write to temporary buffers
  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_reduction())
      stream << exprs[k]->process("#name_temp_value[get_group_id(0)] = #name_buf_value[0];") << std::endl;
    stream << exprs[k]->process("#name_temp[get_group_id(0)] = #name_buf[0];") << std::endl;
  }
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  /* ------------------------
   * Second kernel
   * -----------------------*/
  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl;
  stream << "__kernel void " << "k" << label << type << "1" << "(" << arguments << generate_arguments("#scalartype", mappings, symbolic_expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "unsigned int lid = get_local_id(0);" << std::endl;

  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_reduction())
    {
      stream << exprs[k]->process("__local unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];");
      stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
      stream << exprs[k]->process("__local #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(exprs[k]->root_op()) + ";");
    }
    else
    {
      stream << exprs[k]->process("__local #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(exprs[k]->root_op()) + ";");
    }
  }

  stream << "for(unsigned int i = lid; i < " << p_.num_groups << "; i += get_local_size(0))" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (unsigned int k = 0; k < N; ++k)
    if (exprs[k]->is_index_reduction())
      compute_index_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"),
                              exprs[k]->process("#name_acc_value"),exprs[k]->process("#name_temp_value[i]"),exprs[k]->root_op());
    else
      compute_reduction(stream, exprs[k]->process("#name_acc"), exprs[k]->process("#name_temp[i]"), exprs[k]->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_reduction())
      stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
  }


  //Reduce and write final result
  reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value");

  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  std::map<std::string, std::string> accessors;
  accessors["scalar_reduction"] = "#name_buf[0]";
  accessors["scalar"] = "*#pointer";
  accessors["array"] = "#pointer[#start1]";
  evaluate(stream, PARENT_NODE_TYPE, accessors, symbolic_expressions, mappings);
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

std::vector<std::string> reduction::generate_impl(unsigned int label,  symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const
{
  std::vector<std::string> result;
  result.push_back(generate_impl(label, 'f', symbolic_expressions, mappings, 1));
  result.push_back(generate_impl(label, 'o', symbolic_expressions, mappings, p_.simd_width));
  return result;
}

reduction::reduction(reduction::parameters_type const & parameters,
                                       binding_policy_t binding) : base_impl<reduction, reduction_parameters>(parameters, binding)
{ }

reduction::reduction(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, binding_policy_t bind):
    base_impl<reduction, reduction_parameters>(reduction_parameters(simd,ls,ng,fetch), bind)
{}

std::vector<int_t> reduction::input_sizes(symbolic_expressions_container const & symbolic_expressions)
{
  std::vector<size_t> reductions_idx = filter_nodes(&is_reduction, *(symbolic_expressions.data().front()), false);
  int_t N = vector_size(lhs_most(symbolic_expressions.data().front()->tree(), reductions_idx[0]));
  return tools::make_vector<int_t>() << N;
}

void reduction::enqueue(cl::CommandQueue & queue,
             std::vector<cl::lazy_compiler> & programs,
             unsigned int label,
             symbolic_expressions_container const & symbolic_expressions)
{
  //Preprocessing
  std::vector<int_t> size = input_sizes(symbolic_expressions);
  std::vector<symbolic_expression_node const *> reductions;
  for (symbolic_expressions_container::data_type::const_iterator it = symbolic_expressions.data().begin(); it != symbolic_expressions.data().end(); ++it)
  {
    std::vector<size_t> reductions_idx = filter_nodes(&is_reduction, **it, false);
    for (std::vector<size_t>::iterator itt = reductions_idx.begin(); itt != reductions_idx.end(); ++itt)
      reductions.push_back(&(*it)->tree()[*itt]);
  }

  //Kernel
  char kfallback[2][10];
  fill_kernel_name(kfallback[0], label, "f0");
  fill_kernel_name(kfallback[1], label, "f1");
  char kopt[2][10];
  fill_kernel_name(kopt[0], label, "o0");
  fill_kernel_name(kopt[1], label, "o1");

  bool fallback = has_strided_access(symbolic_expressions) && p_.simd_width > 1;
  cl::Program & program = programs[fallback?0:1].program();
  cl::Kernel kernels[2] = { cl::Kernel(program, fallback?kfallback[0]:kopt[0]),
                            cl::Kernel(program, fallback?kfallback[1]:kopt[1]) };

  //NDRange
  cl::NDRange grange[2] = { cl::NDRange(p_.local_size_0*p_.num_groups), cl::NDRange(p_.local_size_0) };
  cl::NDRange lrange[2] = { cl::NDRange(p_.local_size_0), cl::NDRange(p_.local_size_0) };

  //Arguments
  cl::Context context = symbolic_expressions.context();
  symbolic_expression const & s = *(symbolic_expressions.data().front());
  unsigned int dtype_size = size_of(lhs_most(s.tree(), s.root()).lhs.dtype);
  for (unsigned int k = 0; k < 2; k++)
  {
    unsigned int n_arg = 0;
    kernels[k].setArg(n_arg++, cl_uint(size[0]));

    //Temporary buffers
    unsigned int i = 0;
    unsigned int j = 0;
    for (std::vector<symbolic_expression_node const *>::const_iterator it = reductions.begin(); it != reductions.end(); ++it)
    {
      if (is_index_reduction((*it)->op))
      {
        if (tmpidx_.size() <= j)
          tmpidx_.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, p_.num_groups*4));
        kernels[k].setArg(n_arg++, tmpidx_[j]);
        j++;
      }
      if (tmp_.size() <= i)
        tmp_.push_back(cl::Buffer(context, CL_MEM_READ_WRITE, p_.num_groups*dtype_size));
      kernels[k].setArg(n_arg++, tmp_[i]);
      i++;
    }
    set_arguments(symbolic_expressions, kernels[k], n_arg);
  }

  for (unsigned int k = 0; k < 2; k++)
    queue.enqueueNDRangeKernel(kernels[k], cl::NullRange, grange[k], lrange[k]);
}

template class base_impl<reduction, reduction_parameters>;

}
