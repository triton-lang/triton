#include <cstring>
#include <iostream>
#include "isaac/kernels/stream.h"
#include "isaac/kernels/keywords.h"
#include "isaac/kernels/templates/gemv.h"
#include "isaac/tools/to_string.hpp"
#include "isaac/tools/make_map.hpp"
#include "isaac/tools/make_vector.hpp"

namespace isaac
{
namespace templates
{

gemv_parameters::gemv_parameters(unsigned int _simd_width,
                              unsigned int _local_size_0, unsigned int _local_size_1,
                              unsigned int _num_groups_0, unsigned int _num_groups_1, fetching_policy_type _fetch_policy): base::parameters_type(_simd_width, _local_size_0, _local_size_1, 1),
num_groups_0(_num_groups_0), num_groups_1(_num_groups_1), fetch_policy(_fetch_policy) { }


int gemv::is_invalid_impl(driver::Device const &, expressions_tuple const &) const
{
  if(dot_type_==REDUCE_ROWS && p_.simd_width>1)
    return TEMPLATE_INVALID_SIMD_WIDTH;
  if (p_.fetch_policy==FETCH_FROM_LOCAL)
    return TEMPLATE_INVALID_FETCHING_POLICY_TYPE;
  return TEMPLATE_VALID;
}

unsigned int gemv::lmem_usage(const expressions_tuple &) const
{
  return (p_.local_size_0+1)*p_.local_size_1;
}

std::string gemv::generate_impl(const char * suffix, expressions_tuple const & expressions, driver::Device const & device, std::vector<mapping_type> const & mappings) const
{
  using tools::to_string;


  std::vector<mapped_gemv*> dots;
  expressions_tuple::data_type::const_iterator sit;
  std::vector<mapping_type>::const_iterator mit;
  for (mit = mappings.begin(), sit = expressions.data().begin(); mit != mappings.end(); ++mit, ++sit)
  {
    array_expression const & first_expression = *expressions.data().front();
    std::vector<size_t> idx = filter_nodes(&is_dot, first_expression, false);
    for (auto & elem : idx)
      dots.push_back((mapped_gemv*)(mit->at(mapping_key(elem, PARENT_NODE_TYPE)).get()));
  }

  kernel_generation_stream stream;
  driver::backend_type backend = device.backend();
  std::string _size_t = size_type(device);

  char name[2][16] = {{"prod"}, {"reduce"}};
  strcat(name[0], suffix);
  strcat(name[1], suffix);

  std::string arguments = _size_t + " M, " + _size_t + " N, " ;
  for (const auto & e : dots)
  {
    std::string numeric_type = numeric_type_to_string(lhs_most(e->array_expression().tree(), e->array_expression().root()).lhs.dtype);
    if (e->is_index_dot())
    {
      arguments += e->process(Global(backend).get() + " unsigned int* #name_temp, ");
      arguments += e->process(Global(backend).get() + " " + to_string(numeric_type) + "* #name_temp_value,");
    }
    else
      arguments += e->process(Global(backend).get() + " " + to_string(numeric_type) + "* #name_temp, ");
  }

  switch(backend)
  {
#ifdef ISAAC_WITH_CUDA
    case driver::CUDA: stream << "#include  \"helper_math.h\"" << std::endl; break;
#endif
    case driver::OPENCL: stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl; break;
  }

  stream << KernelPrefix(backend) << " void " << name[0] << "(" << arguments << generate_arguments("#scalartype", device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array0", "#scalartype #namereg = #pointer[#start];"},
                         {"array1", "#pointer += #start;"},
                         {"array2", "#pointer += #start1 + #start2*#ld; "
                                    "#ld *= #nldstride; "}}, expressions, mappings);

  unsigned int local_size_0_ld = p_.local_size_0;
  std::string local_size_0_ld_str = to_string(local_size_0_ld);

  for (const auto & e : dots)
    stream << e->process(Local(backend).get() + " #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << "" << _size_t << " lid0 = " << LocalIdx0(backend) << ";" << std::endl;
  stream << "" << _size_t << " gid0 = " << GlobalIdx0(backend) << ";" << std::endl;
  stream << "" << _size_t << " gpid0 = " << GroupIdx0(backend) << ";" << std::endl;
  stream << "" << _size_t << " gsize0 = " << GlobalSize0(backend) << ";" << std::endl;

  stream << "" << _size_t << " lid1 = " << LocalIdx1(backend) <<";" << std::endl;
  stream << "" << _size_t << " gid1 = " << GlobalIdx1(backend) <<";" << std::endl;
  stream << "" << _size_t << " gpid1 = " << GroupIdx1(backend) << ";" << std::endl;
  stream << "" << _size_t << " gsize1 = " << GlobalSize1(backend) <<";" << std::endl;

  stream << "" << _size_t << " upper_bound_1 = ( M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << ";" << std::endl;
  stream << "for(" << _size_t << " r = gid1; r < upper_bound_1; r += gsize1){" << std::endl;
  stream.inc_tab();

  for (const auto & e : dots)
    stream << e->process("#scalartype #name_acc = " + neutral_element((e)->root_op(), backend, "#scalartype") + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  element_wise_loop_1D(stream, p_.fetch_policy, p_.simd_width, "c", "N", "gid0", "gsize0", device, [&](unsigned int simd_width)
  {
    std::string data_type = append_width("#scalartype",simd_width);


    for (const auto & e : dots)
    {
      std::map<std::string, std::string> accessors;
      if(dot_type_==REDUCE_COLUMNS)
      {
        accessors["array2"] = data_type + " #namereg = " + vload(simd_width, "#scalartype", "c*#stride1", "#pointer + r*#ld", backend)+";";
        accessors["repeat"] = data_type + " #namereg = " + vload(simd_width, "#scalartype", "(c%#tuplearg0)*#stride", "#pointer + (r%#tuplearg1)*#stride ", backend)+";";
      }
      else
      {
        accessors["array2"] = "#scalartype #namereg = #pointer[r*#stride1 + c*#ld];";
        accessors["repeat"] = "#scalartype #namereg = $VALUE{(r%#tuplearg0)*#stride, (c%#tuplearg1)*#stride};";
      }
      e->process_recursive(stream, PARENT_NODE_TYPE, accessors);
    }

    //Update accumulators
    std::vector<std::string> str(simd_width);
    if (simd_width==1)
      str[0] = "#namereg";
    else
      for (unsigned int a = 0; a < simd_width; ++a)
        str[a] = access_vector_type("#namereg",a);


    for (auto & elem : dots)
      for (unsigned int a = 0; a < simd_width; ++a)
      {
        std::string value = elem->evaluate_recursive(LHS_NODE_TYPE, {{"array2", str[a]}, {"repeat", str[a]}, {"array0", "#namereg"}});
        if (elem->is_index_dot())
          compute_index_dot(stream, elem->process("#name_acc"), "c*"+to_string(simd_width) + to_string(a), elem->process("#name_acc_value"), value, elem->root_op());
        else
          compute_dot(stream, elem->process("#name_acc"), value,elem->root_op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : dots)
    stream << expr->process("#name_buf[lid1*" + local_size_0_ld_str + "+ lid0] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(" << _size_t << " stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << LocalBarrier(backend) << ";" << std::endl;
  stream <<  "if (lid0 < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : dots)
    if (e->is_index_dot())
      compute_index_dot(stream, e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->root_op());
    else
      compute_dot(stream,e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lid0 == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();
  if(p_.num_groups_0==1)
  {
    std::map<std::string, std::string> accessors;
    accessors["gemv"] = "#name_buf[lid1*" + local_size_0_ld_str + "]";
    accessors["array1"] = "#pointer[r*#stride]";
    evaluate(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);
  }
  else
  {
    for (mapped_dot const * e : dots)
    {
      if (e->is_index_dot())
        stream << e->process("#name_temp_value[r + M*gpid0] = #name_buf_value[lid1*" + local_size_0_ld_str + "];") << std::endl;
      stream << e->process("#name_temp[r + M*gpid0] = #name_buf[lid1*" + local_size_0_ld_str + "];") << std::endl;
    }
  }
  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;

  if(p_.num_groups_0>1)
  {
  /////////////////////////////////////////
  ////////////// Kernel 2
  ////////////////////////////////////////

  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << p_.local_size_0 << "," << p_.local_size_1 << ",1)))" << std::endl;

  stream << KernelPrefix(backend) << " void " << name[1] << "(" << arguments << generate_arguments("#scalartype", device, mappings, expressions) << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  process(stream, PARENT_NODE_TYPE,
                        {{"array0", "#scalartype #namereg = #pointer[#start];"},
                         {"array1", "#pointer += #start;"},
                         {"array2", "#pointer += #start1 + #start2*#ld; "
                                    "#ld *= #nldstride; "}}, expressions, mappings);

  for (const auto & e : dots)
    stream << e->process(Local(backend).get() + " #scalartype #name_buf[" + to_string(p_.local_size_1*local_size_0_ld) + "];") << std::endl;

  stream << _size_t << " lid0 = " << LocalIdx0(backend) << ";" << std::endl;
  stream << _size_t << " lsize0 = " << LocalSize0(backend) << ";" << std::endl;

  stream << _size_t << " lid1 = " << LocalIdx1(backend) <<";" << std::endl;
  stream << _size_t << " gid1 = " << GlobalIdx1(backend) <<";" << std::endl;
  stream << _size_t << " gsize1 = " << GlobalSize1(backend) <<";" << std::endl;



  stream << _size_t << " upper_bound_1 = ( M +" << p_.local_size_1 - 1 << ")/" << p_.local_size_1 << "*" << p_.local_size_1 << ";" << std::endl;
  stream << "for(" << _size_t << " r = gid1; r < upper_bound_1; r += gsize1){" << std::endl;
  stream.inc_tab();

  for (const auto & e : dots)
    stream << e->process("#scalartype #name_acc = " + neutral_element((e)->root_op(), backend, "#scalartype") + ";") << std::endl;

  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << "for(" << _size_t << " c = lid0; c < " << p_.num_groups_0 << "; c += lsize0){" << std::endl;
  stream.inc_tab();

  for (mapped_dot* e: dots)
    compute_dot(stream, e->process("#name_acc"), e->process("#name_temp[r + M*c]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  for (auto & expr : dots)
    stream << expr->process("#name_buf[lid1*" + local_size_0_ld_str + "+ lid0] = #name_acc;") << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(" << _size_t << " stride = " << p_.local_size_0/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  stream << LocalBarrier(backend) << ";" << std::endl;
  stream <<  "if (lid0 < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  for (auto & e : dots)
    if (e->is_index_dot())
      compute_index_dot(stream, e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf_value[lid1*" + local_size_0_ld_str + " + lid0 + stride]")
                                    , e->root_op());
    else
      compute_dot(stream,e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0]"), e->process("#name_buf[lid1*" + local_size_0_ld_str + " + lid0 + stride]"), e->root_op());

  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;


  stream <<  "if (lid0 == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();

  std::map<std::string, std::string> accessors;
  accessors["gemv"] = "#name_buf[lid1*" + local_size_0_ld_str + "]";
  accessors["array1"] = "#pointer[r*#stride]";
  evaluate(stream, PARENT_NODE_TYPE, accessors, expressions, mappings);

  stream.dec_tab();
  stream << "}" << std::endl;


  stream.dec_tab();
  stream << "}" << std::endl;

  stream.dec_tab();
  stream << "}" << std::endl;
  }

  return stream.str();
}

gemv::gemv(gemv::parameters_type const & parameters,
                                         gemv::dot_type rtype,
                                         binding_policy_t binding_policy) :
  base_impl<gemv, gemv_parameters>(parameters, binding_policy),
  dot_type_(rtype){ }

std::vector<int_t> gemv::input_sizes(expressions_tuple const & expressions) const
{
  array_expression const & first_expression = *expressions.data().front();
  std::vector<std::size_t> idx = filter_nodes(&is_dot, first_expression, false);
  std::pair<int_t, int_t> MN = matrix_size(lhs_most(first_expression.tree(), idx[0]));
  if(dot_type_==REDUCE_COLUMNS)
    std::swap(MN.first,MN.second);
  return tools::make_vector<int_t>() << MN.first << MN.second;
}

void gemv::enqueue(driver::CommandQueue & queue, driver::Program const & program, const char * suffix, base & fallback, controller<expressions_tuple> const & controller)
{
  expressions_tuple const & expressions = controller.x();
  driver::Context const & context = expressions.context();

  std::vector<int_t> MN = input_sizes(expressions);
  std::vector<array_expression::node const *> dots;
  for (const auto & e : expressions.data())
  {
    std::vector<size_t> dots_idx = filter_nodes(&is_dot, *e, false);
    for (auto & r : dots_idx)
      dots.push_back(&(e)->tree()[r]);
  }

  //Fallback
  if(dot_type_==REDUCE_COLUMNS && p_.simd_width>1 && requires_fallback(expressions))
  {
      fallback.enqueue(queue, program, "fallback", fallback, controller);
      return;
  }

  //Kernel
  std::vector< driver::Buffer > tmp;
  std::vector< driver::Buffer > tmpidx;
  unsigned int dtype_size = size_of(lhs_most(expressions.data().front()->tree(), expressions.data().front()->root()).lhs.dtype);

  char name[2][32] = {{"prod"}, {"reduce"}};
  strcat(name[0], suffix);
  strcat(name[1], suffix);

  unsigned int nk = (p_.num_groups_0==1)?1:2;

  std::vector<driver::Kernel> kernels;
  for(unsigned int k = 0 ; k < nk ; ++k)
    kernels.push_back(driver::Kernel(program, name[k]));

  for(unsigned int k = 0 ; k < nk ; ++k)
  {
    driver::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setSizeArg(n_arg++, M);
    kernel.setSizeArg(n_arg++, N);

    //Temporary buffers
    unsigned int i = 0;
    unsigned int j = 0;
    for (auto const & r : dots)
    {
      if (is_index_dot(r->op))
      {
        if (tmpidx.size() <= j)
          tmpidx.push_back(driver::Buffer(context, p_.num_groups_0*M*4));
        kernel.setArg(n_arg++, tmpidx[j]);
        j++;
      }
      if (tmp.size() <= i)
        tmp.push_back(driver::Buffer(context, p_.num_groups_0*M*dtype_size));
      kernel.setArg(n_arg++, tmp[i]);
      i++;
    }
    set_arguments(expressions, kernel, n_arg);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(p_.local_size_0*p_.num_groups_0, p_.local_size_1*p_.num_groups_1), driver::NDRange(p_.local_size_0, p_.local_size_1*p_.num_groups_1) };
  driver::NDRange local[2] = { driver::NDRange(p_.local_size_0, p_.local_size_1), driver::NDRange(p_.local_size_0, p_.local_size_1) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    controller.execution_options().enqueue(program.context(), kernels[i], global[i], local[i]);
}

gemv_n::gemv_n(gemv_parameters  const & parameters,
                                           binding_policy_t binding_policy):
  gemv(parameters, REDUCE_ROWS, binding_policy){}

gemv_n::gemv_n(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind):
  gemv(gemv_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_ROWS, bind)
{}


gemv_t::gemv_t(gemv::parameters_type  const & parameters,
                                           binding_policy_t binding_policy):
  gemv(parameters, REDUCE_COLUMNS, binding_policy){}

gemv_t::gemv_t(unsigned int simd, unsigned int ls1, unsigned int ls2,
                                           unsigned int ng1, unsigned int ng2, fetching_policy_type fetch, binding_policy_t bind):
  gemv(gemv_parameters(simd, ls1, ls2, ng1, ng2, fetch), REDUCE_COLUMNS, bind)
{}


}
}
