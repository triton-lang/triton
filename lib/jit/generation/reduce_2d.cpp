/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include <cstring>
#include <iostream>

#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/jit/generation/engine/stream.h"
#include "isaac/jit/generation/reduce_2d.h"
#include "tools/arguments.hpp"
#include "tools/loop.hpp"
#include "tools/reductions.hpp"
#include "tools/vector_types.hpp"

#include <string>

namespace isaac
{
namespace templates
{
void infos(expression_tree const & tree, symbolic::preset::gemv::args& arguments)
{
  expression_tree::data_type const & array = tree.data();
  std::size_t root = tree.root();
  arguments = symbolic::preset::gemv::check(array, root);
}

/* ------------------ INTELBLAS ------------------ */
intelblas_gemv::intelblas_gemv()
{

}

std::string intelblas_gemv::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  using tools::to_string;


  std::vector<symbolic::reduce_2d*> reductions = symbolic::extract<symbolic::reduce_2d>(tree, symbols);
  std::vector<std::size_t> assignments = symbolic::assignments(tree);
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  stream << "void matvec_mul_trail_rows(unsigned int M," << std::endl;
  stream << "                           unsigned int N," << std::endl;
  stream << "                           unsigned int row_gid," << std::endl;
  stream << "                           unsigned int lid," << std::endl;
  stream << "                           const __global float* src0_read," << std::endl;
  stream << "                           int lda," << std::endl;
  stream << "                           const __global float* src1_read," << std::endl;
  stream << "                           int incv," << std::endl;
  stream << "                           __local float4* work," << std::endl;
  stream << "                           float alpha," << std::endl;
  stream << "                           float beta," << std::endl;
  stream << "                           __global float* result," << std::endl;
  stream << "                           int incr)" << std::endl;
  stream << "{" << std::endl;
  stream << "  __local float* work_each = (__local float*)work;" << std::endl;
  stream << "" << std::endl;
  stream << "  unsigned int rows = M - row_gid * 4;" << std::endl;
  stream << "" << std::endl;
  stream << "  float4 dot[3] = {(float4)(0.f), (float4)(0.f), (float4)(0.f)};" << std::endl;
  stream << "" << std::endl;
  stream << "  unsigned int i = lid;" << std::endl;
  stream << "  while( i < N / 4) {" << std::endl;
  stream << "    // const float4 b0 = vload4(i, src1_read);" << std::endl;
  stream << "    const float4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "    for(unsigned int j = 0; j < rows; ++j) {" << std::endl;
  stream << "      dot[j] += b0 * vload4(i, src0_read + j * lda);" << std::endl;
  stream << "    }" << std::endl;
  stream << "" << std::endl;
  stream << "    i += get_local_size(0);" << std::endl;
  stream << "  }" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "  for(unsigned int j = 0; j < rows; ++j) {" << std::endl;
  stream << "    work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;" << std::endl;
  stream << "  }" << std::endl;
  stream << "" << std::endl;
  stream << "  if(i == N / 4) {" << std::endl;
  stream << "    unsigned short trail_item = N % 4;" << std::endl;
  stream << "" << std::endl;
  stream << "    if(trail_item != 0) {" << std::endl;
  stream << "      const __global float *src0_trail = src0_read + i * 4;" << std::endl;
  stream << "      const __global float *src1_trail = src1_read + i * 4 * incv;" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "      for(unsigned short i = 0; i < trail_item; ++i) {" << std::endl;
  stream << "        const float bt = src1_trail[i*incv];" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "        for(unsigned int j = 0; j < rows; ++j) {" << std::endl;
  stream << "          work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];" << std::endl;
  stream << "        }" << std::endl;
  stream << "      }" << std::endl;
  stream << "    }" << std::endl;
  stream << "  }" << std::endl;
  stream << "" << std::endl;
  stream << "  for(unsigned int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {" << std::endl;
  stream << "    barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  stream << "    if(lid < stride)" << std::endl;
  stream << "      work[lid] += work[lid+stride];" << std::endl;
  stream << "  }" << std::endl;
  stream << "" << std::endl;
  stream << "  if(lid == 0) {" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "    for(unsigned int j = 0; j < rows; ++j) {" << std::endl;
  stream << "      result[(row_gid * 4  + j) * incr] = alpha * work_each[j] + beta * result[(row_gid * 4 + j) * incr];" << std::endl;
  stream << "    }" << std::endl;
  stream << "  }" << std::endl;
  stream << "" << std::endl;
  stream << "}" << std::endl;
  stream << "" << std::endl;
  stream << "__kernel void matvec_mul(" << std::endl;
  stream << "          unsigned int M," << std::endl;
  stream << "          unsigned int N," << std::endl;
  stream << "          __global const float * A," << std::endl;
  stream << "          int offA," << std::endl;
  stream << "          int lda," << std::endl;
  stream << "          __global const float * v," << std::endl;
  stream << "          int offv," << std::endl;
  stream << "          int incv," << std::endl;
  stream << "          float alpha," << std::endl;
  stream << "          float beta," << std::endl;
  stream << "          __global float * result," << std::endl;
  stream << "          int offr," << std::endl;
  stream << "          int incr)" << std::endl;
  stream << "{" << std::endl;
  stream << "  unsigned int row_gid = get_group_id(0);" << std::endl;
  stream << "  unsigned int lid = get_local_id(0);" << std::endl;
  stream << "  const __global float *src0_read = A + row_gid * 4 * lda + offA;" << std::endl;
  stream << "  const __global float *src1_read = v + offv;" << std::endl;
  stream << "  result = result + offr;" << std::endl;
  stream << "" << std::endl;
  stream << "  src1_read += incv > 0 ? 0 : (1 - N) * incv;" << std::endl;
  stream << "  result += incr > 0 ? 0 : (1 - M) * incr;" << std::endl;
  stream << "  __local float4 work[128];" << std::endl;
  stream << "  __local float* work_each = (__local float*)work;" << std::endl;
  stream << "" << std::endl;
  stream << "  if(row_gid == M / 4)" << std::endl;
  stream << "    matvec_mul_trail_rows(M, N, row_gid, lid, src0_read, lda, src1_read, incv, work, alpha, beta, result, incr);" << std::endl;
  stream << "  else" << std::endl;
  stream << "  {" << std::endl;
  stream << "    float4 dot[4] = {(float4)(0.f), (float4)(0.f), (float4)(0.f), (float4)(0.f)};" << std::endl;
  stream << "    unsigned int i = lid;" << std::endl;
  stream << "    while( i < N / 4) {" << std::endl;
  stream << "      // const float4 b0 = vload4(i, src1_read);" << std::endl;
  stream << "      const float4 b0 = {src1_read[i*4*incv], src1_read[(i*4+1)*incv], src1_read[(i*4+2)*incv], src1_read[(i*4+3)*incv]};" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "      for(unsigned int j = 0; j < 4; ++j) {" << std::endl;
  stream << "        dot[j] += b0 * vload4(i, src0_read + j * lda);" << std::endl;
  stream << "      }" << std::endl;
  stream << "      i += get_local_size(0);" << std::endl;
  stream << "    }" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "    for(unsigned int j = 0; j < 4; ++j) {" << std::endl;
  stream << "      work_each[lid * 4 + j] = dot[j].x + dot[j].y + dot[j].z + dot[j].w;" << std::endl;
  stream << "    }" << std::endl;
  stream << "" << std::endl;
  stream << "    if(i == N / 4) {" << std::endl;
  stream << "      unsigned short trail_item = N % 4;" << std::endl;
  stream << "      if(trail_item != 0) {" << std::endl;
  stream << "        const __global float *src0_trail = src0_read + i * 4;" << std::endl;
  stream << "        const __global float *src1_trail = src1_read + i * 4 * incv;" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "        for(unsigned short i = 0; i < trail_item; ++i) {" << std::endl;
  stream << "          const float bt = src1_trail[i * incv];" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "          for(unsigned int j = 0; j < 4; ++j) {" << std::endl;
  stream << "            work_each[lid * 4 + j] += bt * src0_trail[i + j * lda];" << std::endl;
  stream << "          }" << std::endl;
  stream << "        }" << std::endl;
  stream << "      }" << std::endl;
  stream << "    }" << std::endl;
  stream << "" << std::endl;
  stream << "    for(unsigned int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {" << std::endl;
  stream << "      barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
  stream << "      if(lid < stride)" << std::endl;
  stream << "        work[lid] += work[lid+stride];" << std::endl;
  stream << "    }" << std::endl;
  stream << "" << std::endl;
  stream << "    if(lid == 0) {" << std::endl;
  stream << "      // vstore4(alpha * work[0] + beta * vload4(row_gid, result), row_gid, result);" << std::endl;
  stream << "      result[row_gid*4*incr] = alpha * work[0].s0 + beta * result[row_gid*4*incr];" << std::endl;
  stream << "      result[(row_gid*4+1)*incr] = alpha * work[0].s1 + beta * result[(row_gid*4+1)*incr];" << std::endl;
  stream << "      result[(row_gid*4+2)*incr] = alpha * work[0].s2 + beta * result[(row_gid*4+2)*incr];" << std::endl;
  stream << "      result[(row_gid*4+3)*incr] = alpha * work[0].s3 + beta * result[(row_gid*4+3)*incr];" << std::endl;
  stream << "    }" << std::endl;
  stream << "  }" << std::endl;
  stream << "}" << std::endl;
  stream << "" << std::endl;
  stream << "__kernel void trans_matvec_mul(" << std::endl;
  stream << "          unsigned int M," << std::endl;
  stream << "          unsigned int N," << std::endl;
  stream << "          __global const float * A," << std::endl;
  stream << "          int offA," << std::endl;
  stream << "          int lda," << std::endl;
  stream << "          __global const float * v," << std::endl;
  stream << "          int offv," << std::endl;
  stream << "          int incv," << std::endl;
  stream << "          float alpha," << std::endl;
  stream << "          float beta," << std::endl;
  stream << "          __global float * result," << std::endl;
  stream << "          int offr, " << std::endl;
  stream << "          int incr)" << std::endl;
  stream << "{" << std::endl;
  stream << "  unsigned int col_gid = get_global_id(0);" << std::endl;
  stream << "  A += offA + col_gid;" << std::endl;
  stream << "  v += offv;" << std::endl;
  stream << "  result += offr;" << std::endl;
  stream << "" << std::endl;
  stream << "  v += incv > 0 ? 0 : (1 - M) * incv;" << std::endl;
  stream << "  result += incr > 0 ? 0 : (1 - N) * incr;" << std::endl;
  stream << "" << std::endl;
  stream << "  float dot_prod = 0;" << std::endl;
  stream << "  unsigned int row_id = 0;" << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "  for(unsigned int row = 0; row < M; ++row) {" << std::endl;
  stream << "    dot_prod += A[row_id] * v[row * incv];" << std::endl;
  stream << "    row_id += lda;" << std::endl;
  stream << "  }" << std::endl;
  stream << "  result[col_gid * incr] = beta * result[col_gid * incr];" << std::endl;
  stream << "  result[col_gid * incr] += alpha * dot_prod;" << std::endl;
  stream << "}" << std::endl;
  // std::cout << stream.str() << std::endl;
  return stream.str();
}

void intelblas_gemv::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
    namespace drv = driver;
    //Get GEMV info
    symbolic::preset::gemv::args args;
    infos(control.x(), args);
    std::vector<int_t> MN = args.A->shape;
    int_t M = MN[0], N = MN[1];
    if(args.type == REDUCE_2D_COLS)
    {
      M = MN[1], N = MN[0];
    }

    //Kernel
    std::string name[2] = {"matvec_mul", "trans_matvec_mul"};


    std::vector<driver::Kernel> kernels;
    if(args.type == REDUCE_2D_COLS)
    kernels.push_back(driver::Kernel(program, name[0].c_str()));
    else
    kernels.push_back(driver::Kernel(program, name[1].c_str()));

    driver::Kernel & kernel = kernels[0];
    unsigned int n_arg = 0;
    int_t rows = M;
    int_t cols = N;
    if (args.type == REDUCE_2D_ROWS) {
      rows = N;
      cols = M;
    }

    kernel.setSizeArg(n_arg++, rows);
    kernel.setSizeArg(n_arg++, cols);
    kernel.setArg(n_arg++, args.A->array.handle.cl);
    kernel.setSizeArg(n_arg++, args.A->array.start);
    kernel.setSizeArg(n_arg++, args.A->ld[1]);
    kernel.setArg(n_arg++, args.X->array.handle.cl);
    kernel.setSizeArg(n_arg++, args.X->array.start);
    kernel.setSizeArg(n_arg++, args.X->ld[0]);
    kernel.setArg(n_arg++, args.alpha);
    kernel.setArg(n_arg++, args.beta);
    kernel.setArg(n_arg++, args.Y->array.handle.cl);
    kernel.setSizeArg(n_arg++, args.Y->array.start);
    kernel.setSizeArg(n_arg++, args.Y->ld[0]);

    //NDRange
    driver::NDRange local[2] = { driver::NDRange(128), driver::NDRange(1) };
    driver::NDRange global[2] = { driver::NDRange((rows + 3) / 4 * 128), driver::NDRange(cols) };


    if(args.type == REDUCE_2D_COLS)
    control.execution_options().enqueue(program.context(), kernels[0], global[0], local[0]);
    else
    control.execution_options().enqueue(program.context(), kernels[0], global[1], local[1]);
}

unsigned int reduce_2d::lmem_usage(const expression_tree&) const
{
  return (ls0_+1)*ls1_;
}

unsigned int reduce_2d::temporary_workspace(expression_tree const & expressions) const
{
    std::vector<int_t> MN = input_sizes(expressions);
    int_t M = MN[0];
    if(ng0_ > 1)
      return M*ng0_;
    return 0;
}

std::string reduce_2d::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  using tools::to_string;


  std::vector<symbolic::reduce_2d*> reductions = symbolic::extract<symbolic::reduce_2d>(tree, symbols);
  std::vector<std::size_t> assignments = symbolic::assignments(tree);
  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  unsigned int ldls = ls0_;
  std::string ls0ldstr = to_string(ldls);

  auto unroll_tmp = [&]()
  {
      unsigned int offset = 0;
      for (symbolic::reduce_2d* rd : reductions)
      {
        numeric_type dtype = tree.dtype();
        std::string sdtype = to_string(dtype);
        if (is_indexing(rd->op().type))
        {
          stream << rd->process("$GLOBAL uint* #name_temp = ($GLOBAL uint*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += 4*ng0_;
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp_value = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*ng0_;
        }
        else{
          stream << rd->process("$GLOBAL " + sdtype + "* #name_temp = ($GLOBAL " + sdtype + "*)(tmp + " + tools::to_string(offset) + "*M);");
          offset += size_of(dtype)*ng0_;
        }
      }
  };

  /* ------------------------
   * Kernel 1
   * -----------------------*/
  switch(backend)
  {
    case driver::CUDA:
      stream << "#include  \"vector.h\"" << std::endl;
      break;
    case driver::OPENCL:
      stream << " __attribute__((reqd_work_group_size(" << ls0_ << "," << ls1_ << ",1)))" << std::endl;
      break;
  }
  stream << "$KERNEL void " << name[0] << "($SIZE_T M, $SIZE_T N, $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  //Unroll
  unroll_tmp();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  //Loop r
  std::ostringstream upper;
  upper << "(M +" << ls1_ - 1 << ")/" << ls1_ << "*" << ls1_;

  element_wise_loop_1D(stream, (reduction_type_==REDUCE_ROWS)?1:1, "r", upper.str(), "$GLOBAL_IDX_1", "$GLOBAL_SIZE_1", [&](unsigned int cwidth)
  {
  //Declare Buffers
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL " + append_width("#scalartype", cwidth) + " #name_buf[" + to_string(ls1_*ldls) + "];") << std::endl;

  //Accumulators
  for (symbolic::reduce_2d* rd : reductions){
    std::string data_type = append_width("#scalartype",cwidth);
    stream << rd->process(data_type + " #name_acc = " + InitPrefix(backend, data_type).get()  + "(" + neutral_element((rd)->op(), backend, "#scalartype") + ");") << std::endl;
  }
  //Loop c
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  element_wise_loop_1D(stream, (reduction_type_==REDUCE_COLUMNS)?vwidth_:1, "c", "N", "$GLOBAL_IDX_0", "$GLOBAL_SIZE_0", [&](unsigned int rwidth)
  {
    std::string rdtype = append_width("#scalartype", rwidth);
    std::string cdtype = append_width("#scalartype", cwidth);
    //Fetch
    std::set<std::string> fetched;
    for (symbolic::reduce_2d* rd : reductions)
      for(symbolic::leaf* sym: symbolic::extract<symbolic::leaf>(tree, symbols, rd->root(), false))
        if(fetched.insert(sym->process("#name")).second){
          if(reduction_type_==REDUCE_COLUMNS)
            stream << sym->process(rdtype + " #name = " + append_width("loadv",rwidth) + "(c,r);") << std::endl;
          else
            stream << sym->process(cdtype + " #name = " + append_width("loadv",cwidth) + "(r,c);") << std::endl;
        }
    //Compute
    for (symbolic::reduce_2d* rd : reductions)
      for (unsigned int s = 0; s < rwidth; ++s){
        std::string value = rd->lhs()->evaluate({{"leaf", access_vector_type("#name", s, rwidth)}});
        if (is_indexing(rd->op().type))
          compute_index_reduce_1d(stream, rd->process("#name_acc"), "c*"+to_string(rwidth) + to_string(s), rd->process("#name_acc_value"), value, rd->op());
        else
          compute_reduce_1d(stream, rd->process("#name_acc"), value,rd->op());
      }
  });
  stream.dec_tab();
  stream << "}" << std::endl;
  //Copy to local memory
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#name_buf[lidy*" + ls0ldstr + "+ lidx] = #name_acc;") << std::endl;
  //Reduce local memory
  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << ls0_/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd : reductions)
    if (is_indexing(rd->op().type))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  //Write result/temporary
  stream <<  "if (r < M && lidx == 0)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  if(ng0_==1)
    for(size_t idx: assignments)
      for(size_t s = 0 ; s < cwidth ; ++s)
          stream << symbols.at(idx)->evaluate({{"leaf", "at(r+" + to_string(s) + ")"},
                                                {"reduce_2d", access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s, cwidth)}}) << ";" << std::endl;
  else
    for (symbolic::reduction const * rd : reductions)
      for(size_t s = 0 ; s < cwidth ; ++s){
          if (is_indexing(rd->op().type))
              stream << "if(r + " << s << "< M) " << rd->process("#name_temp_value[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf_value[lidy*" + ls0ldstr + "]", s, cwidth)) << ";" << std::endl;
          stream << "if(r + " << s << "< M) " << rd->process("#name_temp[r + " + to_string(s) + " + M*$GROUP_IDX_0] = " + access_vector_type("#name_buf[lidy*" + ls0ldstr + "]", s, cwidth)) << ";" << std::endl;
      }
  stream.dec_tab();
  stream << "}" << std::endl;
  });
  stream.dec_tab();
  stream << "}" << std::endl;


  /* ------------------------
   * Kernel 2
   * -----------------------*/
  if(ng0_>1)
  {
  if(backend==driver::OPENCL)
    stream << " __attribute__((reqd_work_group_size(" << ls0_ << "," << ls1_ << ",1)))" << std::endl;
  stream << "$KERNEL void " << name[1] << "($SIZE_T M, $SIZE_T N , $GLOBAL char* tmp, " << tools::join(kernel_arguments(device, symbols, tree), ", ") << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  unroll_tmp();
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("$LOCAL #scalartype #name_buf[" + to_string(ls1_*ldls) + "];") << std::endl;
  stream << "for($SIZE_T r = $GLOBAL_IDX_1; r < (M +" << ls1_ - 1 << ")/" << ls1_ << "*" << ls1_ << "; r += " << GlobalSize1(backend) << "){" << std::endl;
  stream.inc_tab();
  stream << "$SIZE_T lidx = $LOCAL_IDX_0;" << std::endl;
  stream << "$SIZE_T lidy = $LOCAL_IDX_1;" << std::endl;
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#scalartype #name_acc = " + neutral_element((rd)->op(), backend, "#scalartype") + ";") << std::endl;
  stream << "if (r < M)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "for($SIZE_T c = lidx; c < " << ng0_ << "; c += $LOCAL_SIZE_0){" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd: reductions)
    compute_reduce_1d(stream, rd->process("#name_acc"), rd->process("#name_temp[r + M*c]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  for (symbolic::reduce_2d* rd : reductions)
    stream << rd->process("#name_buf[lidy*" + ls0ldstr + "+ lidx] = #name_acc;") << std::endl;
  stream << "#pragma unroll" << std::endl;
  stream << "for($SIZE_T stride = " << ls0_/2 << "; stride >0; stride /=2)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  stream << "$LOCAL_BARRIER;" << std::endl;
  stream <<  "if (lidx < stride)" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  for (symbolic::reduce_2d* rd : reductions)
    if (is_indexing(rd->op().type))
      compute_index_reduce_1d(stream, rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf_value[lidy*" + ls0ldstr + " + lidx + stride]")
                                    , rd->op());
    else
      compute_reduce_1d(stream,rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx]"), rd->process("#name_buf[lidy*" + ls0ldstr + " + lidx + stride]"), rd->op());
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream <<  "if (lidx == 0 && r < M)";
  stream << "{" << std::endl;
  stream.inc_tab();
  for(size_t idx: assignments)
    stream << symbols.at(idx)->evaluate({{"leaf", "at(r)"}, {"reduce_2d", "#name_buf[lidy*" + ls0ldstr + "]"}}) << ";" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  }
  // std::cout << stream.str() << std::endl;
  return stream.str();
}

reduce_2d::reduce_2d(unsigned int vwidth, unsigned int ls0, unsigned int ls1, unsigned int ng0, unsigned int ng1,
                    operation_type_family rtype) :
  parameterized_base(vwidth, ls0, ls1), ng0_(ng0), ng1_(ng1),
  reduction_type_(rtype){ }

std::vector<int_t> reduce_2d::input_sizes(expression_tree const & tree) const
{
  std::vector<size_t> idx = symbolic::find(tree, [this](expression_tree::node const & x){return x.type==COMPOSITE_OPERATOR_TYPE && x.binary_operator.op.type_family==reduction_type_;});
  std::vector<int_t> shape = tree[tree[idx[0]].binary_operator.lhs].shape;
  if(reduction_type_==REDUCE_COLUMNS)
    return {shape[1], shape[0]};
  return {shape[0], shape[1]};
}

expression_type reduce_2d::type() const
{
  if(reduction_type_==REDUCE_ROWS)
    return REDUCE_2D_ROWS;
  else
    return REDUCE_2D_COLS;
}

void reduce_2d::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const & tree = control.x();
  std::vector<int_t> MN = input_sizes(tree);

  //Kernel
  std::string name[2] = {"prod", "reduce"};
  name[0] += suffix;
  name[1] += suffix;

  unsigned int nk = (ng0_==1)?1:2;

  std::vector<driver::Kernel> kernels;
  for(unsigned int k = 0 ; k < nk ; ++k)
    kernels.push_back(driver::Kernel(program, name[k].c_str()));

  for(unsigned int k = 0 ; k < nk ; ++k)
  {
    driver::Kernel & kernel = kernels[k];
    unsigned int n_arg = 0;
    int_t M = MN[0];
    int_t N = MN[1];
    kernel.setSizeArg(n_arg++, M);
    kernel.setSizeArg(n_arg++, N);
    kernel.setArg(n_arg++, driver::backend::workspaces::get(queue)); //Temporary buffers
    symbolic::set_arguments(tree, kernel, n_arg);
  }

  //NDRange
  driver::NDRange global[2] = { driver::NDRange(ls0_*ng0_, ls1_*ng1_), driver::NDRange(ls0_, ls1_*ng1_) };
  driver::NDRange local[2] = { driver::NDRange(ls0_, ls1_), driver::NDRange(ls0_, ls1_) };
  for(unsigned int i = 0 ; i < nk ; ++i)
    control.execution_options().enqueue(program.context(), kernels[i], global[i], local[i]);
}

reduce_2d_rows::reduce_2d_rows(unsigned int vwidth, unsigned int ls0, unsigned int ls1,  unsigned int ng0, unsigned int ng1): reduce_2d(vwidth, ls0, ls1, ng0, ng1, REDUCE_ROWS) {}

reduce_2d_cols::reduce_2d_cols(unsigned int vwidth, unsigned int ls0, unsigned int ls1, unsigned int ng0, unsigned int ng1): reduce_2d(vwidth, ls0, ls1, ng0, ng1, REDUCE_COLUMNS) {}


}
}
