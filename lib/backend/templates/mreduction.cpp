#include <iostream>
#include "atidlas/backend/stream.h"
#include "atidlas/backend/templates/mreduction.h"
#include "atidlas/tools/to_string.hpp"
#include "atidlas/tools/make_map.hpp"
#include "atidlas/tools/make_vector.hpp"

namespace atidlas
{

mreduction_parameters::mreduction_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, fetching_policy_type _fetch_policy): template_base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
num_groups_0(_num_groups_0), fetch_policy(_fetch_policy) { }


int mreduction::check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const
{
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int mreduction::lmem_usage() const
{
  return p_.local_size_0*(p_.local_size_1+1);
}

std::string mreduction::generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings, unsigned int simd_width, std::vector<mapped_mreduction*> const & exprs) const
{
  using tools::to_string;

  unsigned int lsize0 = p_.local_size_0;
  unsigned int lsize1 = p_.local_size_1+1;
  std::string lsize1str = to_string(lsize1);

  kernel_generation_stream stream;

  stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;
  stream << "__kernel void " << "k" << label << "d" << "(unsigned int M, unsigned int N, " << generate_arguments("#scalartype", mappings, symbolic_expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        tools::make_map<std::multimap<std::string, std::string> >("scalar", "#scalartype #namereg = *#pointer;")
                                                        ("array", "#pointer += #start1 + #start2*#ld;")
                                                        ("array", "#ld *= #nldstride;")
                                                        ("array", "#pointer += #start1;"), symbolic_expressions, mappings);

  for (std::vector<mapped_mreduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
    stream << (*it)->process("__local #scalartype #name_buf[" + to_string(lsize0*lsize1) + "];") << std::endl;

  stream << "unsigned int lid0 = get_local_id(0);" << std::endl;
  stream << "unsigned int lid1 = get_local_id(1);" << std::endl;
  stream << "unsigned int upper_bound_0 = ( M +" << p_.local_size_0 - 1 << ")/" << p_.local_size_0 << "*" << p_.local_size_0 << ";" << std::endl;
  stream << "for(unsigned int r = get_global_id(0); r < upper_bound_0; r += get_global_size(0)){" << std::endl;
  stream.inc_tab();

  for (std::vector<mapped_mreduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
    stream << (*it)->process("#scalartype #name_acc = " + neutral_element((*it)->root_op()) + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  class loop_body : public loop_body_base
  {
  public:
    loop_body(std::vector<mapped_mreduction*> const & _exprs, reduction_type _reduction_type) : exprs(_exprs), reduction(_reduction_type){ }

    void operator()(kernel_generation_stream & stream, unsigned int simd_width) const
    {
      std::string data_type = append_width("#scalartype",simd_width);

      for (std::vector<mapped_mreduction*>::const_iterator it = exprs.begin(); it != exprs.end(); ++it)
      {
        std::multimap<std::string, std::string> accessors;
        accessors.insert(std::make_pair("matrix_repeat", "#scalartype #namereg = #pointer[$OFFSET{(r%#tuplearg0)*#stride1, (c%#tuplearg1)*#stride2}];"));
        if(reduction==REDUCE_COLUMNS)
          accessors.insert(std::make_pair("array", data_type + " #namereg = " + vload(simd_width, "c*#stride1", "#pointer + r*#ld")+";"));
        else
          accessors.insert(std::make_pair("array","#scalartype #namereg = #pointer[r*#stride1 + c*#ld];"));
        (*it)->process_recursive(stream, PARENT_NODE_TYPE, accessors);
      }


      //Update accumulators
      std::vector<std::string> str(simd_width);
      if (simd_width==1)
        str[0] = "#namereg";
      else
        for (unsigned int a = 0; a < simd_width; ++a)
          str[a] = append_simd_suffix("#namereg.s",a);


      for (unsigned int k = 0; k < exprs.size(); ++k)
      {
        for (unsigned int a = 0; a < simd_width; ++a)
        {
          std::map<std::string, std::string> accessors;
          accessors["array"] = str[a];
          accessors["matrix_repeat"] = "#namereg";
          accessors["scalar"] = "#namereg";
          std::string value = exprs[k]->evaluate_recursive(LHS_NODE_TYPE, accessors);
          if (exprs[k]->is_index_reduction())
            compute_index_reduction(stream, exprs[k]->process("#name_acc"), "c*"+to_string(simd_width) + to_string(a), exprs[k]->process("#name_acc_value"), value,exprs[k]->root_op());
          else
            compute_reduction(stream, exprs[k]->process("#name_acc"), value,exprs[k]->root_op());
        }
      }
    }
  private:
    std::vector<mapped_mreduction*> exprs;
    reduction_type reduction;
  };

  element_wise_loop_1D(stream, loop_body(exprs, reduction_type_), p_.fetch_policy, simd_width, "c", "N", "get_local_id(1)", "get_local_size(1)");
  stream.dec_tab();
  stream << "}" << std::endl;

  for (unsigned int k = 0; k < exprs.size(); ++k)
    stream << exprs[k]->process("#name_buf[lid0*" + lsize1str + "+ lid1] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << p_.local_size_1/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream <<  "if (lid1 < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (unsigned int k = 0; k < exprs.size(); k++)
    if (exprs[k]->is_index_reduction())
      compute_index_reduction(stream, exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1 + stride]")
                              , exprs[k]->process("#name_buf_value[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf_value[lid0*" + lsize1str + " + lid1 + stride]"),
                              exprs[k]->root_op());
    else
      compute_reduction(stream,exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1]"), exprs[k]->process("#name_buf[lid0*" + lsize1str + " + lid1 + stride]"), exprs[k]->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lid1 == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();
  std::map<std::string, std::string> accessors;
  accessors["mreduction"] = "#name_buf[lid0*" + lsize1str + "]";
  accessors["array"] = "#pointer[r*#stride1]";
  evaluate(stream, PARENT_NODE_TYPE, accessors, symbolic_expressions, mappings);
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  return stream.str();
}

std::vector<std::string> mreduction::generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const
{
  std::vector<mapped_mreduction*> exprs;
  symbolic_expressions_container::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;
  for (mit = mappings.begin(), sit = symbolic_expressions.data().begin(); mit != mappings.end(); ++mit, ++sit)
  {
    symbolic_expression const & first_expression = *symbolic_expressions.data().front();
    std::vector<size_t> idx = filter_nodes(&is_reduction, first_expression, false);
    for (unsigned int j = 0; j < idx.size(); ++j)
      exprs.push_back((mapped_mreduction*)(mit->at(mapping_key(idx[j], PARENT_NODE_TYPE)).get()));
  }

  std::vector<std::string> res;
  if (reduction_type_ && p_.simd_width>1)
  {
    res.push_back(generate_impl(label, symbolic_expressions, mappings, p_.simd_width, exprs));
    res.push_back(generate_impl(label, symbolic_expressions, mappings, 1, exprs));
  }
  else
    res.push_back(generate_impl(label, symbolic_expressions, mappings, 1, exprs));
  return res;
}

mreduction::mreduction(mreduction::parameters_type const & parameters,
                                         mreduction::reduction_type rtype,
                                         binding_policy_t binding_policy) :
  template_base_impl<mreduction, mreduction_parameters>(parameters, binding_policy),
  reduction_type_(rtype){ }

std::vector<int_t> mreduction::input_sizes(symbolic_expressions_container const & symbolic_expressions)
{
  symbolic_expression const & first_expression = *symbolic_expressions.data().front();
  std::vector<std::size_t> idx = filter_nodes(&is_reduction, first_expression, false);
  std::pair<int_t, int_t> MN = matrix_size(lhs_most(first_expression.tree(), idx[0]));
  if(reduction_type_==REDUCE_COLUMNS)
    std::swap(MN.first,MN.second);
  return tools::make_vector<int_t>() << MN.first << MN.second;
}

void mreduction::enqueue(cl::CommandQueue & queue,
             std::vector<cl::lazy_compiler> & programs,
             unsigned int label,
             symbolic_expressions_container const & symbolic_expressions)
{
  char kname[10];
  fill_kernel_name(kname, label, "d");
  std::vector<int_t> MN = input_sizes(symbolic_expressions);

  //Kernel
  int idx = 0;
  if(reduction_type_==REDUCE_COLUMNS && p_.simd_width>1 && has_strided_access(symbolic_expressions))
    idx = 1;
  cl::Program & program = programs[idx].program();
  cl::Kernel kernel(program, kname);

  //NDRange
  cl::NDRange grange(p_.local_size_0*p_.num_groups_0, p_.local_size_1);
  cl::NDRange lrange(p_.local_size_0, p_.local_size_1);

  unsigned int current_arg = 0;
  kernel.setArg(current_arg++, cl_uint(MN[0]));
  kernel.setArg(current_arg++, cl_uint(MN[1]));
  set_arguments(symbolic_expressions, kernel, current_arg);

  queue.enqueueNDRangeKernel(kernel, cl::NullRange, grange, lrange);
}

mreduction_rows::mreduction_rows(mreduction_parameters  const & parameters,
                                           binding_policy_t binding_policy):
  mreduction(parameters, REDUCE_ROWS, binding_policy){}

mreduction_rows::mreduction_rows(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng, fetching_policy_type fetch, binding_policy_t bind):
  mreduction(mreduction_parameters(simd, ls1, ls2, ng, fetch), REDUCE_ROWS, bind)
{}


mreduction_cols::mreduction_cols(mreduction::parameters_type  const & parameters,
                                           binding_policy_t binding_policy):
  mreduction(parameters, REDUCE_COLUMNS, binding_policy){}

mreduction_cols::mreduction_cols(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng, fetching_policy_type fetch, binding_policy_t bind):
  mreduction(mreduction_parameters(simd, ls1, ls2, ng, fetch), REDUCE_COLUMNS, bind)
{}

template class template_base_impl<mreduction, mreduction_parameters>;


}
