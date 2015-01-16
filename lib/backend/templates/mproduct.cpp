#include "atidlas/array.h"
#include "atidlas/backend/templates/mproduct.h"
#include "atidlas/cl/lazy_compiler.h"
#include "atidlas/tools/make_vector.hpp"
#include "atidlas/tools/to_string.hpp"

namespace atidlas
{

mproduct_parameters::mproduct_parameters(unsigned int simd_width
                          , int_t local_size_0, int_t KL, int_t local_size_1
                          , int_t ms, int_t ks, int_t ns
                          , fetching_policy_type A_fetching_policy, fetching_policy_type B_fetching_policy
                          , int_t local_fetch_0, int_t local_fetch_1): template_base::parameters_type(simd_width, local_size_0, local_size_1, 1),
  kL(KL), mS(ms), kS(ks), nS(ns), A_fetching_policy(A_fetching_policy), B_fetching_policy(B_fetching_policy),
  local_fetch_0(local_fetch_0), local_fetch_1(local_fetch_1),
  mL(ms*local_size_0), nL(ns*local_size_1){}


  unsigned int mproduct::lmem_usage(symbolic_expressions_container const & symbolic_expressions) const
  {
    atidlas::symbolic_expression const & symbolic_expression = (*symbolic_expressions.data().front());
    numeric_type numeric_t = lhs_most(symbolic_expression.tree(), symbolic_expression.root()).lhs.dtype;

    unsigned int N = 0;
    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
      N += p_.kL * (p_.mL+1);
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
      N += p_.nL * (p_.kL+1);
    return N*size_of(numeric_t);
  }

  unsigned int mproduct::registers_usage(symbolic_expressions_container const & symbolic_expressions) const
  {
    atidlas::symbolic_expression const & symbolic_expression = (*symbolic_expressions.data().front());
    numeric_type numeric_t = lhs_most(symbolic_expression.tree(), symbolic_expression.root()).lhs.dtype;

    unsigned int N = p_.mS * p_.nS + p_.mS * p_.kS + p_.kS * p_.nS;
    return N*size_of(numeric_t);
  }

  int mproduct::check_invalid_impl(cl::Device const &, symbolic_expressions_container const &) const
  {
    if (p_.A_fetching_policy!=FETCH_FROM_LOCAL && p_.B_fetching_policy!=FETCH_FROM_LOCAL&& (p_.local_fetch_0!=0 || p_.local_fetch_1!=0))
      return TEMPLATE_GLOBAL_MEMORY_REQUIRES_ZERO_LOCAL_FETCH;

    if ((p_.mS % p_.simd_width) > 0 || (p_.nS % p_.simd_width) > 0)
      return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

    if (p_.kS > p_.kL)
      return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

    if (!(A_trans_=='N' && B_trans_=='T') && p_.simd_width>1)
      return TEMPLATE_SIMD_WIDTH_MUST_BE_ONE;

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL || p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      if ((p_.local_fetch_0*p_.local_fetch_1) !=(p_.local_size_0*p_.local_size_1))
        return TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT;
    }

    if (p_.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      unsigned int bound1 = (A_trans_=='N')?p_.kL:p_.mL;
      unsigned int bound0 = (A_trans_=='N')?p_.mL:p_.kL;

      if (p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0)
        return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

      if (p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0)
        return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE:TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE;

    }
    if (p_.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      unsigned int bound1 = (B_trans_=='T')?p_.kL:p_.nL;
      unsigned int bound0 = (B_trans_=='T')?p_.nL:p_.kL;

      if (p_.local_fetch_1>0 && (bound1 % p_.local_fetch_1)> 0)
        return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

      if (p_.local_fetch_0>0 && (bound0 % (p_.local_fetch_0*p_.simd_width)) > 0)
        return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    }

    return TEMPLATE_VALID;
  }

  std::string mproduct::generate_impl(unsigned int label, char id, const symbolic_expressions_container &symbolic_expressions, const std::vector<mapping_type> &, bool fallback) const
  {
    using std::string;
    using tools::to_string;

    parameters_type pfallback(1, p_.local_size_0, p_.kL, p_.local_size_1, p_.mS, 1, p_.nS, p_.A_fetching_policy, p_.B_fetching_policy, p_.local_fetch_0, p_.local_fetch_1);
    parameters_type const & p = fallback?pfallback:p_;

#define MUL_STRIDE1 string(fallback?"*#stride1":"")
#define HANDLE_BOUNDS(in_bounds, to_load) (!fallback?string(to_load):string( string(in_bounds) + "?" + string(to_load) + ":0"))
#define VLOAD(offset, ptr) vload(p.simd_width, offset, ptr)
#define VSTORE(value, offset, ptr) vstore(p.simd_width, value, offset, ptr)

    string widthstr = to_string(p.simd_width);

    //////////////////
    /// INIT
    /// //////////////
    kernel_generation_stream stream;
    symbolic_expression const & st = (*symbolic_expressions.data().front());
    numeric_type dtype = lhs_most(st.tree(), st.root()).lhs.dtype;
    std::string dtypestr = numeric_type_to_string(dtype);

    mapped_array C(dtypestr, 0, 'm');
    mapped_host_scalar alpha(dtypestr, 1);
    mapped_array A(dtypestr, 2, 'm');
    mapped_array B(dtypestr, 3, 'm');
    mapped_host_scalar beta(dtypestr, 4);

    //////////////////
    /// DECLARATIONS
    /// //////////////
    std::string widthdtype = append_width("#scalartype", p.simd_width);
    stream << " __attribute__((reqd_work_group_size(" << p.local_size_0 << "," << p.local_size_1 << ",1)))" << std::endl;
    stream << "__kernel void " << "k" << label << id << "(unsigned int M, unsigned int N,  unsigned int K, "
                               << C.process("__global #scalartype* #pointer, uint #ld, uint #start1, uint #start2, uint #stride1, uint #stride2,")
                               << alpha.process("#scalartype #name,")
                               << A.process("__global " + widthdtype + "* #pointer, uint #ld, uint #start1, uint #start2, uint #stride1, uint #stride2,")
                               << B.process("__global " + widthdtype + "* #pointer, uint #ld, uint #start1, uint #start2, uint #stride1, uint #stride2,")
                               << beta.process("#scalartype #name") << ")"
                               << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(!fallback)
    {
      stream << A.process("#start1 /= " + to_string(p.simd_width) + ";") << std::endl;
      stream << A.process("#ld /= " + to_string(p.simd_width) + ";") << std::endl;
      stream << B.process("#start1/= "  + to_string(p.simd_width) + ";") << std::endl;
      stream << B.process("#ld /= " + to_string(p.simd_width) + ";") << std::endl;
    }
    stream << A.process("#pointer += $OFFSET{#start1, #start2};") << std::endl
           << A.process("#ld *= #nldstride;") << std::endl
           << B.process("#pointer += $OFFSET{#start1, #start2};") << std::endl
           << B.process("#ld *= #nldstride;") << std::endl
           << C.process("#pointer += $OFFSET{#start1, #start2};") << std::endl
           << C.process("#ld *= #nldstride;") << std::endl;

    ///Result Values
    stream << C.process("#scalartype rC[" + to_string(p.mS) + "][" + to_string(p.nS) + "] = {{(#scalartype)0}};") << std::endl;
    if (p.A_fetching_policy==FETCH_FROM_LOCAL)
      stream << A.process("#scalartype rA[" + to_string(p.kS) + "][" + to_string(p.mS) + "];") << std::endl;
    else
      stream << A.process(append_width("#scalartype",p.simd_width) + " rA[" + to_string(p.kS) + "][" + to_string(p.mS/p.simd_width) + "];") << std::endl;
    if (p.B_fetching_policy==FETCH_FROM_LOCAL)
      stream << B.process("#scalartype rB[" + to_string(p.kS) + "][" + to_string(p.nS) + "];");
    else
      stream << B.process(append_width("#scalartype",p.simd_width) + " rB[" + to_string(p.kS) + "][" + to_string(p.nS/p.simd_width) + "];") << std::endl;


    if (p.A_fetching_policy==FETCH_FROM_LOCAL)
      stream << A.process("__local #scalartype lA[" + to_string(p.kL*(p.mL+1)) + "];");
    if (p.B_fetching_policy==FETCH_FROM_LOCAL)
      stream << B.process("__local #scalartype lB[" + to_string(p.kL*(p.nL+1)) + "];");
    stream << std::endl;

    stream << "uint gidx = get_group_id(0);" << std::endl;
    stream << "uint gidy = get_group_id(1);" << std::endl;
    stream << "uint idx = get_local_id(0);" << std::endl;
    stream << "uint idy = get_local_id(1);" << std::endl;

    if (p.A_fetching_policy==FETCH_FROM_LOCAL || p.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      stream << std::endl;
      stream << "uint idt = " << p.local_size_0 << "*idy + idx;" << std::endl;
      stream << "uint idxT = idt % " << p.local_fetch_0 << ";" << std::endl;
      stream << "uint idyT = idt / " << p.local_fetch_0 << ";" << std::endl;
    }
    stream << std::endl;

    if (fallback)
    {
      //Bounds checking for M (in A, C)
      stream << "bool in_bounds_m[" << p.mS << "];" << std::endl;
      stream << "for(unsigned int m = 0; m < " << p.mS << "; m++)" << std::endl;
      stream.inc_tab();
      switch (p.A_fetching_policy)
      {
      case FETCH_FROM_GLOBAL_CONTIGUOUS:
        stream << "in_bounds_m[m] = gidx*" << p.mL << " + idx*" << p.mS << " + m < M;" << std::endl;
        break;
      default:
        stream << "in_bounds_m[m] = gidx*" << p.mL << " + idx + m*" << p.local_size_0 << " < M;" << std::endl;
        break;
      }
      stream.dec_tab();

      //Bounds checking for A if Local
      if (p.A_fetching_policy==FETCH_FROM_LOCAL)
      {
        unsigned int fetch_size = (A_trans_=='N'?p.local_fetch_0*p.simd_width:p.local_fetch_1);
        stream << "bool in_bounds_m_local[" << p.mL/fetch_size << "];" << std::endl;
        stream << "for(unsigned int m = 0; m < " << p.mL/fetch_size << "; m++)" << std::endl;
        stream.inc_tab();
        stream << "in_bounds_m_local[m] = gidx*" << p.mL << " + " << (A_trans_=='N'?"idxT":"idyT") << " + m*" << fetch_size << " < M;" << std::endl;
        stream.dec_tab();
      }

      //Bounds checking for N (in B, C)
      stream << "bool in_bounds_n[" << p.nS << "];" << std::endl;
      stream << "for(unsigned int n = 0; n < " << p.nS << "; n++)" << std::endl;
      stream.inc_tab();
      switch (p.B_fetching_policy)
      {
      case FETCH_FROM_GLOBAL_CONTIGUOUS:
        stream << "in_bounds_n[n] = gidy*" << p.nL << " + idy*" << p.nS << " + n < N;" << std::endl;
        break;
      default:
        stream << "in_bounds_n[n] = gidy*" << p.nL << " + idy + n*" << p.local_size_1 << " < N;" << std::endl;
        break;
      }
      stream.dec_tab();

      //Bounds checking for B if Local
      if (p.B_fetching_policy==FETCH_FROM_LOCAL)
      {
        unsigned int fetch_size = (B_trans_=='T'?p.local_fetch_0*p.simd_width:p.local_fetch_1);
        stream << "bool in_bounds_n_local[" << p.nL/fetch_size << "];" << std::endl;
        stream << "for(unsigned int n = 0; n < " <<  p.nL/fetch_size << "; n++)" << std::endl;
        stream.inc_tab();
        stream << "in_bounds_n_local[n] = gidy*" << p.nL << " + " << (B_trans_=='T'?"idxT":"idyT") << " + n*" << fetch_size << " < N;" << std::endl;
        stream.dec_tab();
      }
    }

    switch (p.A_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      if (A_trans_=='N')
        stream << A.process("#pointer += (gidx*" + to_string(p.mL/p.simd_width) + " + idxT)" + MUL_STRIDE1 + " + idyT*#ld;") << std::endl;
      else
        stream << A.process("#pointer += idxT" + MUL_STRIDE1 + " + gidx*" + to_string(p.mL/p.simd_width) + "*#ld + idyT*#ld;") << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
      if (A_trans_=='N')
        stream << A.process("#pointer += (gidx*" + to_string(p.mL/p.simd_width) + "+ idx*" + to_string(p.mS/p.simd_width) + ")" + MUL_STRIDE1 + ";") << std::endl;
      else
        stream << A.process("#pointer += (gidx*" + to_string(p.mL/p.simd_width) + "+ idx*" + to_string(p.mS/p.simd_width) + ")*#ld;") << std::endl;
      break;

    case FETCH_FROM_GLOBAL_STRIDED:
      if (A_trans_=='N')
        stream << A.process("#pointer += (gidx*" + to_string(p.mL/p.simd_width) + "+ idx" + ")" + MUL_STRIDE1 + ";") << std::endl;
      else
        stream << A.process("#pointer += (gidx*" + to_string(p.mL/p.simd_width) + "+ idx)*#ld;") << std::endl;
      break;

    default: break;
    }

    switch (p.B_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      if (B_trans_=='T')
        stream << B.process("#pointer += (gidy*" + to_string(p.nL/p.simd_width) + " + idxT" + ")" + MUL_STRIDE1 + " + idyT*#ld;") << std::endl;
      else
        stream << B.process("#pointer += idxT" + MUL_STRIDE1 + " + gidy*" + to_string(p.nL/p.simd_width) + "*#ld + idyT*#ld;") << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
      if (B_trans_=='T')
        stream << B.process("#pointer += (gidy*" + to_string(p.nL/p.simd_width) + "+ idy*" + to_string(p.nS/p.simd_width) + ")" + MUL_STRIDE1 + ";") << std::endl;
      else
        stream << B.process("#pointer += (gidy*" + to_string(p.nL/p.simd_width) + "+ idy*" + to_string(p.nS/p.simd_width) + ")*#ld;") << std::endl;
      break;

    case FETCH_FROM_GLOBAL_STRIDED:
      if (B_trans_=='T')
        stream << B.process("#pointer += (gidy*" + to_string(p.nL/p.simd_width) + "+ idy" + ")" + MUL_STRIDE1 + ";") << std::endl;
      else
        stream << B.process("#pointer += (gidy*" + to_string(p.nL/p.simd_width) + "+ idy)*#ld;") << std::endl;
      break;

    default: break;
    }

    stream << std::endl;
    stream << "for(unsigned int block_k=0; block_k < K; block_k+=" << p.kL << "){" << std::endl;
    stream.inc_tab();

    if (p.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (A_trans_=='N')
        stream << A.process("__local #scalartype* plA = lA + idyT*" + to_string(p.mL + 1) + " + " + to_string(p.simd_width) + "*idxT;") << std::endl;
      else
        stream << A.process("__local #scalartype* plA = lA + idxT*" + to_string(p.mL + 1) + " + idyT;") << std::endl;
    }


    if (p.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (B_trans_=='T')
        stream  << B.process("__local #scalartype* plB = lB + idyT*" + to_string(p.nL+1) + " + " + to_string(p.simd_width) + "*idxT;") << std::endl;
      else
        stream << B.process("__local #scalartype* plB = lB + idxT*" + to_string(p.nL+1) + "+ idyT;") <<std::endl;
    }


    if (p.A_fetching_policy==FETCH_FROM_LOCAL || p.B_fetching_policy==FETCH_FROM_LOCAL)
      stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;

    ///Fetch LHS to Local Memory
    if (p.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='N')
      for(int_t k = 0; k < p.kL; k += p.local_fetch_1)
        for(int_t m = 0; m < p.mL; m += p.local_fetch_0*p.simd_width)
        {
          string in_bounds = "in_bounds_m_local[" + to_string(m/(p.local_fetch_0*p.simd_width)) + "]";
          string to_load = "#pointer[" + to_string(k) + "*#ld + " + to_string(m/p.simd_width) + MUL_STRIDE1 + "]";
          stream << A.process(VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "plA + " + to_string(k*(p.mL+1)+m))) << ";" << std::endl;
        }
    else if (p.A_fetching_policy==FETCH_FROM_LOCAL && A_trans_=='T')
      for(int_t k = 0; k < p.mL; k += p.local_fetch_1)
        for(int_t m = 0; m < p.kL; m += p.local_fetch_0*p.simd_width)
        {
          string in_bounds = "in_bounds_m_local[" + to_string(k/p.local_fetch_1) + "]";
          string to_load = "#pointer[" + to_string(k) + "*#ld + " + to_string(m/p.simd_width) + MUL_STRIDE1 + "]";
          stream << A.process(VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "plA + " + to_string(m*(p.mL+1)+k))) << ";" << std::endl;
        }

    if (p.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='T')
      for(int_t k = 0; k < p.kL; k += p.local_fetch_1)
        for(int_t n = 0; n < p.nL; n += p.local_fetch_0*p.simd_width)
        {
          string in_bounds = "in_bounds_n_local[" + to_string(n/(p.local_fetch_0*p.simd_width)) + "]";
          string to_load = "#pointer[" + to_string(k) + "*#ld + " + to_string(n/p.simd_width) + MUL_STRIDE1 + "]";
          stream << B.process(VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "plB + " + to_string(k*(p.nL+1)+n))) << ";" << std::endl;
        }
    else if (p.B_fetching_policy==FETCH_FROM_LOCAL && B_trans_=='N')
      for(int_t k = 0; k < p.nL; k += p.local_fetch_1)
        for(int_t n = 0; n < p.kL; n += p.local_fetch_0*p.simd_width)
        {
          string in_bounds = "in_bounds_n_local[" + to_string(k/p.local_fetch_1) + "]";
          string to_load = "#pointer[" + to_string(k) + "*#ld + " + to_string(n/p.simd_width) + MUL_STRIDE1 + "]";
          stream << B.process(VSTORE(HANDLE_BOUNDS(in_bounds, to_load), "0", "plB + " + to_string(n*(p.nL+1)+k))) << ";" << std::endl;
        }

    if (p.A_fetching_policy==FETCH_FROM_LOCAL || p.B_fetching_policy == FETCH_FROM_LOCAL)
    {
      stream << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      stream << "uint offA = " << p.simd_width << "*idx;" << std::endl;
      stream << "uint offB = " << p.simd_width << "*idy;" << std::endl;
    }

    if (fallback)
      stream << "for(unsigned int k = 0; k < " << p.kL << " && (block_k + k < K); k+=" << p.kS << "){" << std::endl;
    else
      stream << "for(unsigned int k = 0; k < " << p.kL << "; k+=" << p.kS << "){" << std::endl;
    stream.inc_tab();

    ///Fetch LHS to registers
    stream << "#pragma unroll " << p.kS << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << p.kS << "; kk++)" << std::endl;
    stream << "#pragma unroll " << p.mS/p.simd_width << std::endl;
    stream << "for(unsigned int mm = 0; mm < " << p.mS/p.simd_width << "; mm++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    switch (p.A_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      for (unsigned int ss = 0; ss < p.simd_width; ++ss)
        stream << "rA[kk][mm*" << p.simd_width << "+" << ss << "] = lA[offA + mm*" << p.local_size_0*p.simd_width << "+" << ss << "+ kk*" << (p.mL+1) << "];" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
    {
      if (A_trans_=='N')
        stream << "rA[kk][mm] = " << A.process(HANDLE_BOUNDS("in_bounds_m[mm]", "#pointer[kk*#ld + mm" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      else
        stream << "rA[kk][mm] = " << A.process(HANDLE_BOUNDS("in_bounds_m[mm]", "#pointer[mm*#ld + kk" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      break;
    }

    case FETCH_FROM_GLOBAL_STRIDED:
    {
      if (A_trans_=='N')
        stream << "rA[kk][mm] = " << A.process(HANDLE_BOUNDS("in_bounds_m[mm]", "#pointer[kk*#ld + mm*" + to_string(p.local_size_0) + MUL_STRIDE1 + "]")) << ";" << std::endl;
      else
        stream << "rA[kk][mm] = " << A.process(HANDLE_BOUNDS("in_bounds_m[mm]", "#pointer[mm*#ld*" + to_string(p.local_size_0) + " + kk" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      break;
    }

    default:
      break;

    }
    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "#pragma unroll " << p.kS << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << p.kS << "; kk++)" << std::endl;
    stream << "#pragma unroll " << p.nS/p.simd_width << std::endl;
    stream << "for(unsigned int nn = 0; nn < " << p.nS/p.simd_width << "; nn++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    switch (p.B_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      for (unsigned int ss = 0; ss < p.simd_width; ++ss)
        stream << "rB[kk][nn*" << p.simd_width << "+" << ss << "] = lB[offB + nn*" << p.local_size_1*p.simd_width << "+" << ss  << "+ kk*" << (p.nL+1) << "];" << std::endl;
      break;

    case FETCH_FROM_GLOBAL_CONTIGUOUS:
    {
      if (B_trans_=='T')
        stream << "rB[kk][nn] = " << B.process(HANDLE_BOUNDS("in_bounds_n[nn]", "#pointer[kk*#ld + nn" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      else
        stream << "rB[kk][nn] = " << B.process(HANDLE_BOUNDS("in_bounds_n[nn]", "#pointer[nn*#ld + kk" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      break;
    }

    case FETCH_FROM_GLOBAL_STRIDED:
    {
      if (B_trans_=='T')
        stream << "rB[kk][nn] = " << B.process(HANDLE_BOUNDS("in_bounds_n[nn]", "#pointer[kk*#ld + nn*" + to_string(p.local_size_1) + MUL_STRIDE1 + "]")) << ";" << std::endl;
      else
        stream << "rB[kk][nn] = " << B.process(HANDLE_BOUNDS("in_bounds_n[nn]", "#pointer[nn*#ld*" + to_string(p.local_size_1) + " + kk" + MUL_STRIDE1 + "]")) << ";" << std::endl;
      break;
    }

    default: break;
    }
    stream.dec_tab();
    stream << "}" << std::endl;


    ///Increment pointers
    switch (p.A_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      stream << "offA += " << p.kS*(p.mL+1) << ";" << std::endl;
      break;

    default:
      if (A_trans_=='N')
        stream << A.process("#pointer += " + to_string(p.kS) + "*#ld;") << std::endl;
      else
        stream << A.process("#pointer += " + to_string(p.kS) + "" + MUL_STRIDE1 + ";") << std::endl;
      break;
    }


    switch (p.B_fetching_policy)
    {
    case FETCH_FROM_LOCAL:
      stream << "offB += " << p.kS*(p.nL+1) << ";" << std::endl;
      break;

    default:
      if (B_trans_=='T')
        stream << B.process("#pointer += " + to_string(p.kS) + "*#ld;") << std::endl;
      else
        stream << B.process("#pointer += " + to_string(p.kS) + "" + MUL_STRIDE1 + ";") << std::endl;
      break;
    }


    stream << "#pragma unroll " << p.kS << std::endl;
    stream << "for(unsigned int kk = 0; kk <" << p.kS << "; ++kk)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    for(int_t nn=0; nn < p.nS; ++nn)
      for(int_t mm=0; mm < p.mS; ++mm)
      {
        string res_str, lhs_str, rhs_str;
        res_str = "rC[" + tools::to_string(mm) + "][" + tools::to_string(nn) + "]";
        if (p.A_fetching_policy==FETCH_FROM_LOCAL || p.simd_width==1)
          lhs_str = "rA[kk][" + tools::to_string(mm) + "]";
        else
          lhs_str = "rA[kk][" + tools::to_string(mm/p.simd_width) + "].s" + tools::to_string(mm%p.simd_width);
        if (p.B_fetching_policy==FETCH_FROM_LOCAL || p.simd_width==1)
          rhs_str = "rB[kk]["+tools::to_string(nn)+"]";
        else
          rhs_str = "rB[kk]["+tools::to_string(nn/p.simd_width)+"].s"+tools::to_string(nn%p.simd_width);
        stream << res_str << "=" << "fma(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
      }
    stream.dec_tab();
    stream << "}" << std::endl;




    stream.dec_tab();
    stream << "}" << std::endl;

    //Increment global pointer if local memory is used
    //Else, it's incremented directly when fetching
    if (p.A_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (A_trans_=='N')
        stream << A.process("#pointer += " + to_string(p.kL) + "*#ld;") << std::endl;
      else
        stream << A.process("#pointer += " + to_string(p.kL) + "" + MUL_STRIDE1 + ";") << std::endl;
    }

    if (p.B_fetching_policy==FETCH_FROM_LOCAL)
    {
      if (B_trans_=='T')
        stream << B.process("#pointer += " + to_string(p.kL) + "*#ld;") << std::endl;
      else
        stream << B.process("#pointer += " + to_string(p.kL) + "" + MUL_STRIDE1 + ";") << std::endl;
    }

    stream.dec_tab();
    stream << "}" << std::endl;

    unsigned int ministartstride0 = p.A_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?p.mS:p.simd_width;
    unsigned int ministartstride1 = p.B_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?p.nS:p.simd_width;

    stream << C.process("#pointer += gidx*" + to_string(p.mL) + "*#stride1;") << std::endl;
    stream << C.process("#pointer += idx*" + to_string(ministartstride0) + "*#stride1;") << std::endl;
    stream << C.process("#pointer += gidy*" + to_string(p.nL) + "*#ld;") << std::endl;
    stream << C.process("#pointer += idy*" + to_string(ministartstride1) + "*#ld;") << std::endl;

    for(int_t m=0; m < p.mS; ++m)
    {
      for(int_t n=0; n < p.nS; ++n)
      {
        unsigned int ministride1 = p.B_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS?1:p.local_size_1;
        string Cj = to_string((n/p.simd_width)*(ministride1*p.simd_width) + n%p.simd_width);
        if (fallback)
        {
          stream << "if (in_bounds_m[" + to_string(m) + "] && in_bounds_n[" + to_string(n) + "])" << std::endl;
          stream.inc_tab();
        }
        stream << C.process("#pointer[" + Cj + "*#ld] = rC[" + to_string(m) + "][" + to_string(n) + "]*" + alpha.name() + " + ((" + beta.name() + "==0)?0:(#pointer[" + Cj + "*#ld]*" + beta.name() + "));") << std::endl;
        if (fallback)
          stream.dec_tab();
      }

      if ((m+1)%p.simd_width>0 || p.A_fetching_policy==FETCH_FROM_GLOBAL_CONTIGUOUS)
        stream << C.process("#pointer += #stride1;") << std::endl;
      else
        stream << C.process("#pointer += " + to_string((p.local_size_0*p.simd_width) - (p.simd_width-1)) + "*#stride1;") << std::endl;
    }

    stream.dec_tab();
    stream << "}" << std::endl;

    return stream.str();

#undef MUL_STRIDE1
#undef HANDLE_BOUNDS
#undef VLOAD
#undef VST0RE
  }

  std::vector<std::string> mproduct::generate_impl(unsigned int label, symbolic_expressions_container const & symbolic_expressions, std::vector<mapping_type> const & mappings) const
  {
    std::vector<std::string> res;
    res.push_back(generate_impl(label, 'o', symbolic_expressions, mappings, false));
    res.push_back(generate_impl(label, 'f', symbolic_expressions, mappings, true));
    return res;
  }

  void mproduct::enqueue_block(cl::CommandQueue & queue, int_t M, int_t N, int_t K,
                     array const & A, array const & B, array const & C,
                     value_scalar const & alpha, value_scalar const & beta,
                     std::vector<cl::lazy_compiler> & programs, unsigned int label, int id)
  {
    if (min(A.shape())==0 || min(B.shape())==0 || min(C.shape())==0)
      return;

    char kname[10];
    fill_kernel_name(kname, label, id==1?"f":"o");

    cl::Program & program = programs[id].program();
    cl::Kernel kernel(program, kname);
    cl::NDRange lrange(p_.local_size_0, p_.local_size_1);
    cl::NDRange grange = (id==1)?cl::NDRange(align(align(M,p_.mS)/p_.mS, p_.local_size_0), align(align(N,p_.nS)/p_.nS, p_.local_size_1)):
                                 cl::NDRange(M/p_.mS, N/p_.nS);

    unsigned int current_arg = 0;
    kernel.setArg(current_arg++, cl_uint(M));
    kernel.setArg(current_arg++, cl_uint(N));
    kernel.setArg(current_arg++, cl_uint(K));

    tools::shared_ptr<symbolic_binder> binder = make_binder();
    set_arguments_functor fun(*binder, current_arg, kernel);
    fun.set_arguments(C);
    fun.set_arguments(alpha.dtype(), alpha.values());
    fun.set_arguments(A);
    fun.set_arguments(B);
    fun.set_arguments(beta.dtype(), beta.values());

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, grange, lrange);
  }

  array mproduct::create_slice(array & M, int_t s0_0, int_t s0_1, int_t s1_0, int_t s1_1, bool swap)
  {
    slice s0(s0_0, s0_1);
    slice s1(s1_0, s1_1);
    if (swap)
      std::swap(s0, s1);
    return array(M, s0, s1);
  }

  std::vector<int_t> mproduct::infos(symbolic_expressions_container const & symbolic_expressions,
                                   lhs_rhs_element & C, lhs_rhs_element & A, lhs_rhs_element & B)
  {
    atidlas::symbolic_expression const & symbolic_expression = (*symbolic_expressions.data().front());
    symbolic_expression::container_type const & array = symbolic_expression.tree();
    std::size_t root = symbolic_expression.root();

    C = array[root].lhs;

    int_t A_idx = array[root].rhs.node_index;
    A = array[A_idx].lhs;

    int_t B_idx = array[root].rhs.node_index;
    B = array[B_idx].rhs;

    int_t M = C.array->shape()._1;
    int_t N = C.array->shape()._2;
    int_t K = (A_trans_=='T')?A.array->shape()._1:A.array->shape()._2;

    return tools::make_vector<int_t>() << M << N << K;
  }

  mproduct::mproduct(mproduct_parameters const & parameters, char A_trans, char B_trans) : template_base_impl<mproduct, mproduct_parameters>(parameters, BIND_ALL_UNIQUE), A_trans_(A_trans), B_trans_(B_trans){ }

  std::vector<int_t> mproduct::input_sizes(symbolic_expressions_container const & symbolic_expressions)
  {
    lhs_rhs_element d0, d1, d2;
    return infos(symbolic_expressions, d0, d1, d2);
  }

  void mproduct::enqueue(cl::CommandQueue & queue,
               std::vector<cl::lazy_compiler> & programs,
               unsigned int label,
               symbolic_expressions_container const & symbolic_expressions)
  {
    using namespace tools;

    lhs_rhs_element C, A, B;
    std::vector<int_t> MNK = infos(symbolic_expressions, C, A, B);

    int_t M = MNK[0];
    int_t N = MNK[1];
    int_t K = MNK[2];


    array* pA = A.array;
    array* pB = B.array;
    array* pC = C.array;

    int_t ldstrideA = pA->stride()._1;
    int_t ldstrideB = pB->stride()._1;
    int_t ldstrideC = pC->stride()._1;
    int_t ldstartA = pA->start()._1;
    int_t ldstartB = pB->start()._1;

    bool swap_A = (A_trans_=='T');
    bool swap_B = (B_trans_=='T');

    value_scalar _0f(cl_float(0));
    value_scalar _0d(cl_double(0));
    value_scalar* _0 = C.dtype==FLOAT_TYPE?&_0f:&_0d;

    value_scalar _1f(cl_float(1));
    value_scalar _1d(cl_double(1));
    value_scalar* _1 = C.dtype==FLOAT_TYPE?&_1f:&_1d;

    if (M < p_.mL || N < p_.nL || K < p_.kL ||
        ldstrideA> 1 || ldstrideB > 1 || ldstrideC > 1 ||
        (p_.simd_width>1 && (ldstartA % p_.simd_width > 0 || ldstartB % p_.simd_width > 0)))
    {
      enqueue_block(queue, M, N, K, create_slice(*pA, 0, M, 0, K, swap_A),
                    create_slice(*pB, 0, K, 0, N,  swap_B),
                    create_slice(*pC, 0, M, 0, N, false), *_1, *_0, programs, label, 1);
      return;
    }

    int_t lM = M / p_.mL * p_.mL;
    int_t lN = N / p_.nL * p_.nL;
    int_t lK = K / p_.kL * p_.kL;

    enqueue_block(queue,  lM, lN, lK, create_slice(*pA, 0, lM, 0, lK, swap_A), create_slice(*pB, 0, lK, 0, lN, swap_B), create_slice(*pC, 0, lM, 0, lN, false), *_1, *_0, programs, label, 0);
    enqueue_block(queue,  lM, lN, K - lK, create_slice(*pA, 0, lM, lK, K, swap_A), create_slice(*pB, lK, K, 0, lN, swap_B), create_slice(*pC, 0, lM, 0, lN, false), *_1, *_1, programs, label, 1);

    enqueue_block(queue,  lM, N - lN, lK, create_slice(*pA, 0, lM, 0, lK, swap_A), create_slice(*pB, 0, lK, lN, N, swap_B), create_slice(*pC, 0, lM, lN, N, false), *_1, *_0, programs, label, 1);
    enqueue_block(queue,  lM, N - lN, K - lK, create_slice(*pA, 0, lM, lK, K, swap_A), create_slice(*pB, lK, K, lN, N, swap_B), create_slice(*pC, 0, lM, lN, N, false), *_1, *_1, programs, label, 1);

    enqueue_block(queue,  M - lM, lN, lK, create_slice(*pA, lM, M, 0, lK, swap_A), create_slice(*pB, 0, lK, 0, lN, swap_B), create_slice(*pC, lM, M, 0, lN, false), *_1, *_0, programs, label, 1);
    enqueue_block(queue,  M - lM, lN, K - lK, create_slice(*pA, lM, M, lK, K, swap_A), create_slice(*pB, lK, K, 0, lN, swap_B), create_slice(*pC, lM, M, 0, lN, false), *_1, *_1, programs, label, 1);

    enqueue_block(queue,  M - lM, N - lN, lK, create_slice(*pA, lM, M, 0, lK, swap_A), create_slice(*pB, 0, lK, lN, N, swap_B), create_slice(*pC, lM, M, lN, N, false), *_1, *_0, programs, label, 1);
    enqueue_block(queue,  M - lM, N - lN, K - lK, create_slice(*pA, lM, M, lK, K, swap_A), create_slice(*pB, lK, K, lN, N, swap_B), create_slice(*pC, lM, M, lN, N, false), *_1, *_1, programs, label, 1);
  }

  //

  mproduct_nn::mproduct_nn(mproduct_parameters  const & p) :
    mproduct(p, 'N', 'N')
  { }

  mproduct_nn::mproduct_nn(unsigned int simd
                                           , int_t ls0, int_t KL, int_t ls1
                                           , int_t ms, int_t ks, int_t ns
                                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                                           , int_t lfetch0, int_t lfetch1) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), 'N', 'N')
  { }

  //

  mproduct_tn::mproduct_tn(mproduct_parameters  const & p) :
    mproduct(p, 'T', 'N')
  { }

  mproduct_tn::mproduct_tn(unsigned int simd
                                           , int_t ls0, int_t KL, int_t ls1
                                           , int_t ms, int_t ks, int_t ns
                                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                                           , int_t lfetch0, int_t lfetch1) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), 'T', 'N')
  { }

  //

  mproduct_nt::mproduct_nt(mproduct_parameters  const & p) :
    mproduct(p, 'N', 'T')
  { }

  mproduct_nt::mproduct_nt(unsigned int simd
                                           , int_t ls0, int_t KL, int_t ls1
                                           , int_t ms, int_t ks, int_t ns
                                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                                           , int_t lfetch0, int_t lfetch1) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), 'N', 'T')
  { }

  //

  mproduct_tt::mproduct_tt(mproduct_parameters  const & p) :
    mproduct(p, 'T', 'T')
  { }

  mproduct_tt::mproduct_tt(unsigned int simd
                                           , int_t ls0, int_t KL, int_t ls1
                                           , int_t ms, int_t ks, int_t ns
                                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                                           , int_t lfetch0, int_t lfetch1) :
    mproduct(mproduct_parameters(simd, ls0, KL, ls1, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1), 'T', 'T')
  { }
}


