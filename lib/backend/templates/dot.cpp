#include <cstring>
#include <iostream>
#include "isaac/backend/templates/dot.h"
#include "isaac/tools/to_string.hpp"
#include "isaac/tools/make_map.hpp"
#include "isaac/tools/make_vector.hpp"
#include "isaac/backend/keywords.h"
namespace isaac
{
namespace templates
{
dot_parameters::dot_parameters(unsigned int _simd_width,
                     unsigned int _group_size, unsigned int _num_groups,
                     fetching_policy_type _fetching_policy) : base::parameters_type(_simd_width, _group_size, 1, 2), num_groups(_num_groups), fetching_policy(_fetching_policy)
{ }

unsigned int dot::lmem_usage(expressions_tuple const & expressions) const
{
  unsigned int res = 0;
  for(const auto & elem : expressions.data())
  {
    numeric_type numeric_t= lhs_most((elem)->tree(), (elem)->root()).lhs.dtype;
    res += p_.local_size_0*size_of(numeric_t);
  }
  return res;
}

int dot::is_invalid_impl(driver::Device const &, expressions_tuple const &) const
{
  if (p_.fetching_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

inline void dot::reduce_1d_local_memory(kernel_generation_stream & stream, unsigned int size, std::vector<mapped_scalar_dot*> exprs,
                                   std::string const & buf_str, std::string const & buf_value_str, driver::backend_type backend) const
{
  stream << "#pragma unroll" << std::endl;
  stream << "for(unsigned int stride = " << size/2 << "; stride > 0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << LocalBarrier(backend) << ";" << std::endl;
  stream << "if (lid <  stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & expr : exprs)
    if (expr->is_index_dot())
      compute_index_dot(stream, expr->process(buf_str+"[lid]"), expr->process(buf_str+"[lid+stride]")
                              , expr->process(buf_value_str+"[lid]"), expr->process(buf_value_str+"[lid+stride]"),
                              expr->root_op());
    else
      compute_dot(stream, expr->process(buf_str+"[lid]"), expr->process(buf_str+"[lid+stride]"), expr->root_op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
}

std::string dot::generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const
{
  kernel_generation_stream stream;

  std::vector<mapped_scalar_dot*> exprs;
  for (const auto & mapping : mappings)
    for (mapping_type::const_iterator iit = mapping.begin(); iit != mapping.end(); ++iit)
      if (mapped_scalar_dot * p = dynamic_cast<mapped_scalar_dot*>(iit->second.get()))
        exprs.push_back(p);
  std::size_t N = exprs.size();
  driver::backend_type backend = device.backend();
  std::string _size_t = size_type(device);

  std::string arguments = _size_t + " N, ";
  for (unsigned int k = 0; k < N; ++k)
  {
    std::string numeric_type = numeric_type_to_string(lhs_most(exprs[k]->array_expression().tree(),  exprs[k]->array_expression().root()).lhs.dtype);
    if (exprs[k]->is_index_dot())
    {
      arguments += exprs[k]->process(Global(backend).get() + " unsigned int* #name_temp, ");
      arguments += exprs[k]->process(Global(backend).get() + " " + tools::to_string(numeric_type) + "* #name_temp_value, ");
    }
    else
      arguments += exprs[k]->process(Global(backend).get() + " " + tools::to_string(numeric_type) + "* #name_temp, ");
  }

  char name[2][16] = {{"prod"}, {"reduce"}};
  strcat(name[0], suffix);
  strcat(name[1], suffix);

  /* ------------------------
   * First Kernel
   * -----------------------*/
  switch(backend)
  {
#ifdef ISAAC_WITH_CUDA
    case driver::CUDA: stream << "#include  \"helper_math.h\"" << std::endl; break;
#endif
    case driver::OPENCL: stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << ",1,1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void " << name[0] << "(" << arguments << generate_arguments("#scalartype", device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "unsigned int lid = " <<LocalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gid = " <<GlobalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gpid = " <<GroupIdx0(backend) << ";" << std::endl;
  stream << "unsigned int gsize = " <<GlobalSize0(backend) << ";" << std::endl;

  process(stream, PARENT_NODE_TYPE, {{"array0", "#scalartype #namereg = #pointer[#start];"},
                                     {"array1", "#pointer += #start;"}},
                                    expressions, mappings);

  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_dot())
    {
      stream << exprs[k]->process(Local(backend).get() + " #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc_value = " + neutral_element(exprs[k]->root_op(), backend, "#scalartype") + ";") << std::endl;
      stream << exprs[k]->process(Local(backend).get() + " unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("unsigned int #name_acc = 0;") << std::endl;
    }
    else
    {
      stream << exprs[k]->process(Local(backend).get() + " #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << exprs[k]->process("#scalartype #name_acc = " + neutral_element(exprs[k]->root_op(), backend, "#scalartype") + ";") << std::endl;
    }
  }


  element_wise_loop_1D(stream, p_.fetching_policy, p_.simd_width, "i", "N", GlobalIdx0(backend).get(), GlobalSize0(backend).get(), device, [&](unsigned int simd_width)
  {
    std::string i = (simd_width==1)?"i*#stride":"i";
    //Fetch vector entry
    for (const auto & elem : exprs)
      (elem)->process_recursive(stream, PARENT_NODE_TYPE, {{"array1",  append_width("#scalartype",simd_width) + " #namereg = " + vload(simd_width,"#scalartype",i,"#pointer",backend)+";"},
                                                           {"matrix_row",  "#scalartype #namereg = #pointer[$OFFSET{#row*#stride, i*#stride2}];"},
                                                           {"matrix_column", "#scalartype #namereg = #pointer[$OFFSET{i*#stride,#column*#stride2}];"},
                                                           {"matrix_diag", "#scalartype #namereg = #pointer[#diag_offset<0?$OFFSET{(i - #diag_offset)*#stride, i*#stride2}:$OFFSET{i*#stride, (i + #diag_offset)*#stride2}];"}});

    //Update accumulators
    std::vector<std::string> str(simd_width);
    if (simd_width==1)
      str[0] = "#namereg";
    else
      for (unsigned int a = 0; a < simd_width; ++a)
        str[a] = append_simd_suffix("#namereg.s", a);

    for (auto & elem : exprs)
    {
      for (unsigned int a = 0; a < simd_width; ++a)
      {
        std::map<std::string, std::string> accessors;
        accessors["array1"] = str[a];
        accessors["matrix_row"] = str[a];
        accessors["matrix_column"] = str[a];
        accessors["matrix_diag"] = str[a];
        accessors["array0"] = "#namereg";
        std::string value = elem->evaluate_recursive(LHS_NODE_TYPE, accessors);
        if (elem->is_index_dot())
          compute_index_dot(stream, elem->process("#name_acc"),  "i*" + tools::to_string(simd_width) + "+"
                                  + tools::to_string(a), elem->process("#name_acc_value"), value,elem->root_op());
        else
          compute_dot(stream, elem->process("#name_acc"), value,elem->root_op());
      }
    }
  });

  //Fills local memory
  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_dot())
      stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
  }

  //Reduce local memory
  reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value", backend);

  //Write to temporary buffers
  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_dot())
      stream << exprs[k]->process("#name_temp_value[gpid] = #name_buf_value[0];") << std::endl;
    stream << exprs[k]->process("#name_temp[gpid] = #name_buf[0];") << std::endl;
  }
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  /* ------------------------
   * Second kernel
   * -----------------------*/



  stream << KernelPrefix(backend) << " void " << name[1] << "(" << arguments << generate_arguments("#scalartype", device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "unsigned int lid = " <<LocalIdx0(backend) << ";" << std::endl;
  stream << "unsigned int lsize = " <<LocalSize0(backend) << ";" << std::endl;

  for (mapped_scalar_dot* e: exprs)
  {
    if (e->is_index_dot())
    {
      stream << e->process(Local(backend).get() + " unsigned int #name_buf[" + tools::to_string(p_.local_size_0) + "];");
      stream << e->process("unsigned int #name_acc = 0;") << std::endl;
      stream << e->process(Local(backend).get() + " #scalartype #name_buf_value[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << e->process("#scalartype #name_acc_value = " + neutral_element(e->root_op(), backend, "#scalartype") + ";");
    }
    else
    {
      stream << e->process(Local(backend).get() + " #scalartype #name_buf[" + tools::to_string(p_.local_size_0) + "];") << std::endl;
      stream << e->process("#scalartype #name_acc = " + neutral_element(e->root_op(), backend, "#scalartype") + ";");
    }
  }

  stream << "for(unsigned int i = lid; i < " << p_.num_groups << "; i += lsize)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (mapped_scalar_dot* e: exprs)
    if (e->is_index_dot())
      compute_index_dot(stream, e->process("#name_acc"), e->process("#name_temp[i]"), e->process("#name_acc_value"),e->process("#name_temp_value[i]"),e->root_op());
    else
      compute_dot(stream, e->process("#name_acc"), e->process("#name_temp[i]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  for (unsigned int k = 0; k < N; ++k)
  {
    if (exprs[k]->is_index_dot())
      stream << exprs[k]->process("#name_buf_value[lid] = #name_acc_value;") << std::endl;
    stream << exprs[k]->process("#name_buf[lid] = #name_acc;") << std::endl;
  }


  //Reduce and write final result
  reduce_1d_local_memory(stream, p_.local_size_0, exprs, "#name_buf", "#name_buf_value", backend);

  stream << "if (lid==0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  std::map<std::string, std::string> accessors;
  accessors["scalar_dot"] = "#name_buf[0]";
  accessors["array0"] = "#pointer[#start]";
  evaluate(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);
  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

//  std::cout << stream.str() << std::endl;

  return stream.str();
}

dot::dot(dot::parameters_type const & parameters,
                                       binding_policy_t binding) : base_impl<dot, dot_parameters>(parameters, binding)
{ }

dot::dot(unsigned int simd, unsigned int ls, unsigned int ng,
                               fetching_policy_type fetch, binding_policy_t bind):
    base_impl<dot, dot_parameters>(dot_parameters(simd,ls,ng,fetch), bind)
{}

std::vector<int_t> dot::input_sizes(expressions_tuple const & expressions) const
{
  std::vector<size_t> dots_idx = filter_nodes(&is_dot, *(expressions.data().front()), false);
  int_t N = vector_size(lhs_most(expressions.data().front()->tree(), dots_idx[0]));
  return tools::make_vector<int_t>() << N;
}

void dot::enqueue(driver::CommandQueue & queue, driver::Program const & program, const char * suffix, base & fallback, controller<expressions_tuple> const & controller)
{
  expressions_tuple const & expressions = controller.x();

  //Preprocessing
  int_t size = input_sizes(expressions)[0];

  //fallback
  if(p_.simd_width > 1 && (requires_fallback(expressions) || (size%p_.simd_width>0)))
  {
      fallback.enqueue(queue, program, "fallback", fallback, controller);
      return;
  }

  std::vector<array_expression::node const *> dots;
  for (const auto & elem : expressions.data())
  {
    std::vector<size_t> dots_idx = filter_nodes(&is_dot, *elem, false);
    for (auto & dots_idx_itt : dots_idx)
      dots.push_back(&(elem)->tree()[dots_idx_itt]);
  }

  //Kernel
  char name[2][32] = {{"prod"}, {"reduce"}};
  strcat(name[0], suffix);
  strcat(name[1], suffix);

  driver::Kernel kernels[2] = { driver::Kernel(program,name[0]), driver::Kernel(program,name[1]) };

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.local_size_0*p_.num_groups), driver::NDRange(p_.local_size_0) };
  driver::NDRange local[2] = { driver::NDRange(p_.local_size_0), driver::NDRange(p_.local_size_0) };

  //Arguments
  driver::Context const & context = expressions.context();
  array_expression const & s = *(expressions.data().front());
  unsigned int dtype_size = size_of(lhs_most(s.tree(), s.root()).lhs.dtype);
  for (auto & kernel : kernels)
  {
    unsigned int n_arg = 0;
    kernel.setSizeArg(n_arg++, size);

    //Temporary buffers
    unsigned int i = 0;
    unsigned int j = 0;
    for (std::vector<array_expression::node const *>::const_iterator it = dots.begin(); it != dots.end(); ++it)
    {
      if (is_index_dot((*it)->op))
      {
        if (tmpidx_.size() <= j)
          tmpidx_.push_back(driver::Buffer(context, p_.num_groups*4));
        kernel.setArg(n_arg++, tmpidx_[j]);
        j++;
      }
      if (tmp_.size() <= i)
        tmp_.push_back(driver::Buffer(context, p_.num_groups*dtype_size));
      kernel.setArg(n_arg++, tmp_[i]);
      i++;
    }

    set_arguments(expressions, kernel, n_arg);
  }

  for (unsigned int k = 0; k < 2; k++)
    controller.execution_options().enqueue(program.context(), kernels[k], global[k], local[k]);
}

}
}
