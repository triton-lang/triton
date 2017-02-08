/* Copyright 2015-2017 Philippe Tillet
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "isaac/array.h"
#include "isaac/driver/dispatch.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/gemm.h"
#include "isaac/jit/generation/engine/keywords.h"
#include "isaac/exception/api.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"
#include <sys/time.h>

#include <string>
#include <cstring>
#include "isaac/tools/cpp/align.hpp"

namespace isaac
{
namespace templates
{

std::vector<int_t> infos(expression_tree const & tree, symbolic::preset::gemm::args& arguments, char A_trans)
{
  expression_tree::data_type const & array = tree.data();
  std::size_t root = tree.root();
  arguments = symbolic::preset::gemm::check(array, root);
  int_t M = arguments.C->shape[0];
  int_t N = arguments.C->shape[1];
  int_t K = (A_trans=='T')?arguments.A->shape[0]:arguments.A->shape[1];
  return {M, N, K};
}

/* ------------------ CUBLAS ------------------ */
cublas_gemm::cublas_gemm(char A_trans, char B_trans): A_trans_(A_trans), B_trans_(B_trans), init_(true)
{ }

int cublas_gemm::is_invalid(expression_tree const  &, driver::Device const & device) const
{ return (init_ && device.backend()==driver::CUDA)?0:-1; }

std::vector<int_t> cublas_gemm::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

expression_type cublas_gemm::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}

void cublas_gemm::enqueue(driver::CommandQueue & queue, driver::Program const &, std::string const &, runtime::execution_handler const & control)
{
  namespace drv = driver;
  //Get GEMM info
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(control.x(), args, A_trans_);
  int_t M = MNK[0], N = MNK[1], K = MNK[2];
  CUdeviceptr cuA = args.A->array.handle.cu;
  CUdeviceptr cuB = args.B->array.handle.cu;
  CUdeviceptr cuC = args.C->array.handle.cu;
  runtime::execution_options_type const & opt = control.execution_options();
  auto cuT = [](char xt) { return (xt=='N')?CUBLAS_OP_N:CUBLAS_OP_T; };
  int offA = args.A->array.start, offB = args.B->array.start, offC = args.C->array.start;
  cublasHandle_t h = drv::dispatch::cublasHandle(queue.context());
  //Set new stream
  cudaStream_t bkp;
  drv::Event event(drv::CUDA);
  drv::dispatch::cublasGetStream_v2(h,&bkp);
  drv::dispatch::cublasSetStream_v2(h,(cudaStream_t)queue.handle().cu());
  values_holder alpha = args.alpha.values();
  values_holder beta = args.beta.values();
  if(opt.events)
    drv::check(drv::dispatch::cuEventRecord(event.handle().cu().first, queue.handle().cu()));
  if(args.C->dtype==FLOAT_TYPE)
    drv::dispatch::cublasSgemm_v2(h,cuT(A_trans_), cuT(B_trans_), M, N, K, &alpha.float32, (float*)cuA + offA , args.A->ld[1], (float*)cuB + offB, args.B->ld[1], &beta.float32, (float*)cuC + offC, args.C->ld[1]);
  else
    drv::dispatch::cublasDgemm_v2(h,cuT(A_trans_), cuT(B_trans_), M, N, K, &alpha.float64, (double*)cuA + offA, args.A->ld[1], (double*)cuB + offB, args.B->ld[1], &beta.float64, (double*)cuC + offC, args.C->ld[1]);
  if(opt.events){
    drv::check(drv::dispatch::cuEventRecord(event.handle().cu().second, queue.handle().cu()));
    opt.events->push_back(event);
  }
  //Revert old stream
  drv::dispatch::cublasSetStream_v2(h,bkp);
}

/* -------------------------------------------- */
  
/* ------------------ INTELBLAS_image ------------------ */
intelblas_gemm_image::intelblas_gemm_image(char A_trans, char B_trans): A_trans_(A_trans), B_trans_(B_trans), init_(true)
{ }

int intelblas_gemm_image::is_invalid(expression_tree const  &, driver::Device const & device) const
{ return (init_ && device.backend()==driver::OPENCL)?0:-1; }

std::vector<int_t> intelblas_gemm_image::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

expression_type intelblas_gemm_image::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}

std::string intelblas_gemm_image::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  (void)tree;
  (void)suffix;
  (void)symbols;
  using std::string;
  using tools::to_string;

  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  stream << "#define TILE_M 32 " << std::endl;
  stream << "#define TILE_K 8 " << std::endl;
  stream << "#define TILE_N 8 " << std::endl;
  stream << "#ifdef USE_IMAGE_C " << std::endl;
  stream << "#define BLOCKC_READ8( _C, _coordC ) as_float8( intel_sub_group_block_read8( _C, _coordC ) ) " << std::endl;
  stream << "#define BLOCKC_WRITE8( _C, _coordC, _val ) intel_sub_group_block_write8( _C, _coordC, as_uint8( _val ) ) " << std::endl;
  stream << "#define MATC_PARAMETER __read_only image2d_t C, __write_only image2d_t dst " << std::endl;
  stream << "#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, C, dst, sizeof(uint)) " << std::endl;
  stream << "#else " << std::endl;
  stream << "#define BLOCKC_READ8( _C, _coordC ) \\ " << std::endl;
  stream << "          (float8) ( (_coordC.x + get_local_id(0) < N && _coordC.y < M) ? _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 1 < M) ? _C[ ( _coordC.y + 1 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 2 < M) ? _C[ ( _coordC.y + 2 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 3 < M) ? _C[ ( _coordC.y + 3 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 4 < M) ? _C[ ( _coordC.y + 4 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 5 < M) ? _C[ ( _coordC.y + 5 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 6 < M) ? _C[ ( _coordC.y + 6 ) * ldc + _coordC.x + get_local_id(0) ] : 0, \\ " << std::endl;
  stream << "                     (_coordC.x + get_local_id(0) < N && _coordC.y + 7 < M) ? _C[ ( _coordC.y + 7 ) * ldc + _coordC.x + get_local_id(0) ] : 0) " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKC_WRITE8( _C, _coordC, _val) do {\\ " << std::endl;
  stream << "                     if (_coordC.x + get_local_id(0) < N) { \\ " << std::endl;
  stream << "                       if (_coordC.y < M) \\ " << std::endl;
  stream << "                         _C[ _coordC.y * ldc + _coordC.x + get_local_id(0) ] = _val.s0; \\ " << std::endl;
  stream << "                       if (_coordC.y + 1 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 1 )* ldc + _coordC.x + get_local_id(0) ] = _val.s1; \\ " << std::endl;
  stream << "                       if (_coordC.y + 2 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 2 )* ldc + _coordC.x + get_local_id(0) ] = _val.s2; \\ " << std::endl;
  stream << "                       if (_coordC.y + 3 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 3 )* ldc + _coordC.x + get_local_id(0) ] = _val.s3; \\ " << std::endl;
  stream << "                       if (_coordC.y + 4 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 4 )* ldc + _coordC.x + get_local_id(0) ] = _val.s4; \\ " << std::endl;
  stream << "                       if (_coordC.y + 5 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 5 )* ldc + _coordC.x + get_local_id(0) ] = _val.s5; \\ " << std::endl;
  stream << "                       if (_coordC.y + 6 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 6 )* ldc + _coordC.x + get_local_id(0) ] = _val.s6; \\ " << std::endl;
  stream << "                       if (_coordC.y + 7 < M) \\ " << std::endl;
  stream << "                         _C[ ( _coordC.y + 7 )* ldc + _coordC.x + get_local_id(0) ] = _val.s7; \\ " << std::endl;
  stream << "                     }} while(0) " << std::endl;
  stream << "#define MATC_PARAMETER __global float * C, const int offC, const int M, const int N, const int ldc " << std::endl;
  stream << "#define GEMM_OUTPUT(ALPHA1, BETA_NOT0) GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, (C + offC), (C + offC), 1) " << std::endl;
  stream << "#endif " << std::endl;
  stream << " " << std::endl;
  stream << "#define GEMM_OUTPUT_EXT(ALPHA1, BETA_NOT0, _C, _dst, _C_step) \\ " << std::endl;
  stream << "    int2    coordDst = (int2)( ( group_x * TILE_N ) * _C_step, ( group_y * TILE_M ) ); \\ " << std::endl;
  stream << "    int2    coordC = coordDst; \\ " << std::endl;
  stream << "    float8 blockC00; \\ " << std::endl;
  stream << "    float8 blockC01; \\ " << std::endl;
  stream << "    float8 blockC02; \\ " << std::endl;
  stream << "    float8 blockC03; \\ " << std::endl;
  stream << "    if (BETA_NOT0) { \\ " << std::endl;
  stream << "        blockC00 = (index == 0) ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC01 = (index == 0) ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC02 = (index == 0) ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC03 = (index == 0) ? BLOCKC_READ8( _C, coordC ) * beta : BLOCKC_READ8( _C, coordC ); \\ " << std::endl;
  stream << "        if (!ALPHA1) { \\ " << std::endl;
  stream << "            blockC00 = mad(blockAxB00, (float8)alpha, blockC00); \\ " << std::endl;
  stream << "            blockC01 = mad(blockAxB01, (float8)alpha, blockC01); \\ " << std::endl;
  stream << "            blockC02 = mad(blockAxB02, (float8)alpha, blockC02); \\ " << std::endl;
  stream << "            blockC03 = mad(blockAxB03, (float8)alpha, blockC03); \\ " << std::endl;
  stream << "        } else { \\ " << std::endl;
  stream << "            blockC00 += blockAxB00; \\ " << std::endl;
  stream << "            blockC01 += blockAxB01; \\ " << std::endl;
  stream << "            blockC02 += blockAxB02; \\ " << std::endl;
  stream << "            blockC03 += blockAxB03; \\ " << std::endl;
  stream << "        } \\ " << std::endl;
  stream << "    } else { \\ " << std::endl;
  stream << "        blockC00 = (index == 0) ? 0 : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC01 = (index == 0) ? 0 : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC02 = (index == 0) ? 0 : BLOCKC_READ8( _C, coordC );    coordC.y += 8; \\ " << std::endl;
  stream << "        blockC03 = (index == 0) ? 0 : BLOCKC_READ8( _C, coordC ); \\ " << std::endl;
  stream << "        if (!ALPHA1) { \\ " << std::endl;
  stream << "          blockC00 = mad(blockAxB00, alpha, blockC00); \\ " << std::endl;
  stream << "          blockC01 = mad(blockAxB01, alpha, blockC01); \\ " << std::endl;
  stream << "          blockC02 = mad(blockAxB02, alpha, blockC02); \\ " << std::endl;
  stream << "          blockC03 = mad(blockAxB03, alpha, blockC03); \\ " << std::endl;
  stream << "        } else { \\ " << std::endl;
  stream << "          blockC00 += blockAxB00; \\ " << std::endl;
  stream << "          blockC01 += blockAxB01; \\ " << std::endl;
  stream << "          blockC02 += blockAxB02; \\ " << std::endl;
  stream << "          blockC03 += blockAxB03; \\ " << std::endl;
  stream << "        } \\ " << std::endl;
  stream << "    } \\ " << std::endl;
  stream << "    BLOCKC_WRITE8( _dst, coordDst, blockC00 );    coordDst.y += 8; \\ " << std::endl;
  stream << "    BLOCKC_WRITE8( _dst, coordDst, blockC01 );    coordDst.y += 8; \\ " << std::endl;
  stream << "    BLOCKC_WRITE8( _dst, coordDst, blockC02 );    coordDst.y += 8; \\ " << std::endl;
  stream << "    BLOCKC_WRITE8( _dst, coordDst, blockC03 ); " << std::endl;
  stream << " " << std::endl;
  stream << "#define TRANSPOSE_BLOCK_8( _block, _col )   \\ " << std::endl;
  stream << "        (float8)( intel_sub_group_shuffle( _block.s0, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s1, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s2, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s3, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s4, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s5, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s6, _col ),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s7, _col ) ); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \\ " << std::endl;
  stream << "        {   \\ " << std::endl;
  stream << "            const float8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \\ " << std::endl;
  stream << "            const float8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \\ " << std::endl;
  stream << "            const float8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \\ " << std::endl;
  stream << "            const float8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \\ " << std::endl;
  stream << "            const float8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \\ " << std::endl;
  stream << "            const float8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \\ " << std::endl;
  stream << "            const float8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \\ " << std::endl;
  stream << "            const float8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s0), acol0, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s1), acol1, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s2), acol2, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s3), acol3, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s4), acol4, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s5), acol5, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s6), acol6, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s7), acol7, _result );      \\ " << std::endl;
  stream << "        } " << std::endl;
  stream << " " << std::endl;
  stream << "#define GEMM_NN(ALPHA1, BETA_NOT0) \\ " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, 1, 1))) \\ " << std::endl;
  stream << "__kernel void intelblas_gemm_image_32_1_NN_ ##ALPHA1 ##_ ##BETA_NOT0 ##_float( \\ " << std::endl;
  stream << "    __read_only image2d_t A, \\ " << std::endl;
  stream << "    __read_only image2d_t B, \\ " << std::endl;
  stream << "    MATC_PARAMETER, \\ " << std::endl;
  stream << "    float alpha, \\ " << std::endl;
  stream << "    float beta, \\ " << std::endl;
  stream << "    int width0, \\ " << std::endl;
  stream << "    int index) \\ " << std::endl;
  stream << "{ \\ " << std::endl;
  stream << "    const int group_x = get_group_id(0); \\ " << std::endl;
  stream << "    const int group_y = get_group_id(1); \\ " << std::endl;
  stream << "    float8 blockAxB00 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB01 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB02 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB03 = 0.0f; \\ " << std::endl;
  stream << "    int2    coordA = (int2)( 0, group_y * TILE_M ); \\ " << std::endl;
  stream << "    int2    coordB = (int2)( ( group_x * TILE_N ) * sizeof(uint), 0 ); \\ " << std::endl;
  stream << "    do \\ " << std::endl;
  stream << "    {  \\ " << std::endl;
  stream << "        int2    coordBTemp = coordB; \\ " << std::endl;
  stream << "        float8  blockB00 = as_float8( intel_sub_group_block_read8( B, coordBTemp ) );    coordB.y += TILE_K; \\ " << std::endl;
  stream << "        int2    coordATemp = coordA; \\ " << std::endl;
  stream << "        float8  blockA00 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8  blockA01 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8  blockA02 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8  blockA03 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordA.x += TILE_K * sizeof(uint); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 ); \\ " << std::endl;
  stream << "    } \\ " << std::endl;
  stream << "    while( coordB.y < width0 ); \\ " << std::endl;
  stream << "    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \\ " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_NN(1, 0)  " << std::endl;
  stream << "GEMM_NN(1, 1)  " << std::endl;
  stream << "GEMM_NN(0, 0)  " << std::endl;
  stream << "GEMM_NN(0, 1)  " << std::endl;
  stream << " " << std::endl;
  stream << "#undef TRANSPOSE_BLOCK_8 " << std::endl;
  stream << "#undef MULTIPLY_BLOCKS_8x8 " << std::endl;
  stream << " " << std::endl;
  stream << "#define TRANSPOSE_BLOCK_8(_vec) \\ " << std::endl;
  stream << "        (float8)( intel_sub_group_shuffle(_vec, 0), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 1), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 2), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 3), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 4), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 5), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 6), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 7) ) " << std::endl;
  stream << " " << std::endl;
  stream << "#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \\ " << std::endl;
  stream << "        {   \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s0), TRANSPOSE_BLOCK_8(_blockA.s0), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s1), TRANSPOSE_BLOCK_8(_blockA.s1), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s2), TRANSPOSE_BLOCK_8(_blockA.s2), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s3), TRANSPOSE_BLOCK_8(_blockA.s3), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s4), TRANSPOSE_BLOCK_8(_blockA.s4), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s5), TRANSPOSE_BLOCK_8(_blockA.s5), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s6), TRANSPOSE_BLOCK_8(_blockA.s6), _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)(_blockB.s7), TRANSPOSE_BLOCK_8(_blockA.s7), _result );      \\ " << std::endl;
  stream << "        } " << std::endl;
  stream << " " << std::endl;
  stream << "#define GEMM_TN(ALPHA1, BETA_NOT0) \\ " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, 1, 1))) \\ " << std::endl;
  stream << "__kernel void intelblas_gemm_image_32_1_TN_ ##ALPHA1 ##_ ##BETA_NOT0 ##_float( \\ " << std::endl;
  stream << "    __read_only image2d_t A, \\ " << std::endl;
  stream << "    __read_only image2d_t B, \\ " << std::endl;
  stream << "    MATC_PARAMETER, \\ " << std::endl;
  stream << "    float alpha, \\ " << std::endl;
  stream << "    float beta, \\ " << std::endl;
  stream << "    int width0, \\ " << std::endl;
  stream << "    int index) \\ " << std::endl;
  stream << "{ \\ " << std::endl;
  stream << "    const int group_x = get_group_id(0);\\ " << std::endl;
  stream << "    const int group_y = get_group_id(1);\\ " << std::endl;
  stream << "    float8 blockAxB00 = 0.0f;\\ " << std::endl;
  stream << "    float8 blockAxB01 = 0.0f;\\ " << std::endl;
  stream << "    float8 blockAxB02 = 0.0f;\\ " << std::endl;
  stream << "    float8 blockAxB03 = 0.0f;\\ " << std::endl;
  stream << "    int2    coordA = (int2)( group_y * TILE_M * sizeof(uint), 0 );\\ " << std::endl;
  stream << "    int2    coordB = (int2)( ( group_x * TILE_N ) * sizeof(uint), 0 );\\ " << std::endl;
  stream << "    do\\ " << std::endl;
  stream << "    {\\ " << std::endl;
  stream << "        int2    coordBTemp = coordB;\\ " << std::endl;
  stream << "        float8 blockB00 = as_float8( intel_sub_group_block_read8( B, coordBTemp ) );    coordB.y += TILE_K;\\ " << std::endl;
  stream << "        int2    coordATemp = coordA;\\ " << std::endl;
  stream << "        float8 blockA00 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint);\\ " << std::endl;
  stream << "        float8 blockA01 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint);\\ " << std::endl;
  stream << "        float8 blockA02 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint);\\ " << std::endl;
  stream << "        float8 blockA03 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordA.y += TILE_K;\\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 ); \\ " << std::endl;
  stream << "    } \\ " << std::endl;
  stream << "    while( coordB.y < width0 ); \\ " << std::endl;
  stream << "    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \\ " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_TN(1, 0) " << std::endl;
  stream << "GEMM_TN(1, 1)  " << std::endl;
  stream << "GEMM_TN(0, 0)  " << std::endl;
  stream << "GEMM_TN(0, 1)  " << std::endl;
  stream << " " << std::endl;
  stream << "#undef MULTIPLY_BLOCKS_8x8 " << std::endl;
  stream << "#undef TRANSPOSE_BLOCK_8 " << std::endl;
  stream << " " << std::endl;
  stream << "#define TRANSPOSE_BLOCK_8( _block, _col )   \\ " << std::endl;
  stream << "        (float8)( intel_sub_group_shuffle( _block.s0, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s1, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s2, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s3, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s4, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s5, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s6, _col),   \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle( _block.s7, _col) ) " << std::endl;
  stream << " " << std::endl;
  stream << "#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \\ " << std::endl;
  stream << "        {   \\ " << std::endl;
  stream << "            const float8    acol0 = TRANSPOSE_BLOCK_8( _blockA, 0 );    \\ " << std::endl;
  stream << "            const float8    acol1 = TRANSPOSE_BLOCK_8( _blockA, 1 );    \\ " << std::endl;
  stream << "            const float8    acol2 = TRANSPOSE_BLOCK_8( _blockA, 2 );    \\ " << std::endl;
  stream << "            const float8    acol3 = TRANSPOSE_BLOCK_8( _blockA, 3 );    \\ " << std::endl;
  stream << "            const float8    acol4 = TRANSPOSE_BLOCK_8( _blockA, 4 );    \\ " << std::endl;
  stream << "            const float8    acol5 = TRANSPOSE_BLOCK_8( _blockA, 5 );    \\ " << std::endl;
  stream << "            const float8    acol6 = TRANSPOSE_BLOCK_8( _blockA, 6 );    \\ " << std::endl;
  stream << "            const float8    acol7 = TRANSPOSE_BLOCK_8( _blockA, 7 );    \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s0, acol0, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s1, acol1, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s2, acol2, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s3, acol3, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s4, acol4, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s5, acol5, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s6, acol6, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s7, acol7, _result );      \\ " << std::endl;
  stream << "        } " << std::endl;
  stream << " " << std::endl;
  stream << "#define GEMM_NT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) \\ " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, 1, 1))) \\ " << std::endl;
  stream << "__kernel void intelblas_gemm_image_32_1_NT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0 ##_float( \\ " << std::endl;
  stream << "    __read_only image2d_t A, \\ " << std::endl;
  stream << "    MATB_PARAMETER, \\ " << std::endl;
  stream << "    MATC_PARAMETER, \\ " << std::endl;
  stream << "    float alpha, \\ " << std::endl;
  stream << "    float beta, \\ " << std::endl;
  stream << "    int padded_k, \\ " << std::endl;
  stream << "    int k, \\ " << std::endl;
  stream << "    int index) \\ " << std::endl;
  stream << "{ \\ " << std::endl;
  stream << "    const int group_x = get_group_id(0); \\ " << std::endl;
  stream << "    const int group_y = get_group_id(1); \\ " << std::endl;
  stream << "    float8 blockAxB00 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB01 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB02 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB03 = 0.0f; \\ " << std::endl;
  stream << "    int2    coordA = (int2)( 0, group_y * TILE_M ); \\ " << std::endl;
  stream << "    int2    coordB = (int2)( 0, ( group_x * TILE_N )); \\ " << std::endl;
  stream << "    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \\ " << std::endl;
  stream << "    do \\ " << std::endl;
  stream << "    { \\ " << std::endl;
  stream << "        float8 blockB00;             \\ " << std::endl;
  stream << "        BLOCKB_READ8(blockB00, B, coordB); \\ " << std::endl;
  stream << "        int2    coordATemp = coordA; \\ " << std::endl;
  stream << "        float8 blockA00 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8 blockA01 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8 blockA02 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.y += 8; \\ " << std::endl;
  stream << "        float8 blockA03 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordA.x += TILE_K * sizeof(uint); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02, blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03, blockB00 ); \\ " << std::endl;
  stream << "    } \\ " << std::endl;
  stream << "    while( coordB.x < padded_k / VECSIZE ); \\ " << std::endl;
  stream << "    GEMM_OUTPUT(ALPHA1, BETA_NOT0); \\ " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        _blockb.s0123 = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s4567 = read_imagef(_B, _coordBTemp); _coordB.x += 2; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __read_only image2d_t B " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_NT(1, 0, VEC4, 4) " << std::endl;
  stream << "GEMM_NT(1, 1, VEC4, 4)  " << std::endl;
  stream << "GEMM_NT(0, 0, VEC4, 4) " << std::endl;
  stream << "GEMM_NT(0, 1, VEC4, 4) " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        _blockb = *(__global float8*)&_B[_coordBTemp.y * ldb + _coordBTemp.x + offB];\\ " << std::endl;
  stream << "        _coordB.x += TILE_K; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __global float *B, int offB, int ldb " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_NT(1, 0, BUFFER, 1)  " << std::endl;
  stream << "GEMM_NT(1, 1, BUFFER, 1)  " << std::endl;
  stream << "GEMM_NT(0, 0, BUFFER, 1)  " << std::endl;
  stream << "GEMM_NT(0, 1, BUFFER, 1)  " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        float4 temp; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s0 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s1 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s2 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s3 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s4 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s5 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s6 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(_B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s7 = temp.s0; \\ " << std::endl;
  stream << "        _coordB.x += 8; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __read_only image2d_t B " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_NT(1, 0, SCALAR, 1) " << std::endl;
  stream << "GEMM_NT(1, 1, SCALAR, 1) " << std::endl;
  stream << "GEMM_NT(0, 0, SCALAR, 1) " << std::endl;
  stream << "GEMM_NT(0, 1, SCALAR, 1) " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << "#undef MULTIPLY_BLOCKS_8x8 " << std::endl;
  stream << "#undef TRANSPOSE_BLOCK_8 " << std::endl;
  stream << " " << std::endl;
  stream << "#define TRANSPOSE_BLOCK_8(_vec) \\ " << std::endl;
  stream << "        (float8)( intel_sub_group_shuffle(_vec, 0), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 1), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 2), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 3), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 4), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 5), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 6), \\ " << std::endl;
  stream << "                  intel_sub_group_shuffle(_vec, 7) ); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MULTIPLY_BLOCKS_8x8( _result, _blockA, _blockB )    \\ " << std::endl;
  stream << "        {   \\ " << std::endl;
  stream << "            const float8    acol0 = TRANSPOSE_BLOCK_8( _blockA.s0 );    \\ " << std::endl;
  stream << "            const float8    acol1 = TRANSPOSE_BLOCK_8( _blockA.s1 );    \\ " << std::endl;
  stream << "            const float8    acol2 = TRANSPOSE_BLOCK_8( _blockA.s2 );    \\ " << std::endl;
  stream << "            const float8    acol3 = TRANSPOSE_BLOCK_8( _blockA.s3 );    \\ " << std::endl;
  stream << "            const float8    acol4 = TRANSPOSE_BLOCK_8( _blockA.s4 );    \\ " << std::endl;
  stream << "            const float8    acol5 = TRANSPOSE_BLOCK_8( _blockA.s5 );    \\ " << std::endl;
  stream << "            const float8    acol6 = TRANSPOSE_BLOCK_8( _blockA.s6 );    \\ " << std::endl;
  stream << "            const float8    acol7 = TRANSPOSE_BLOCK_8( _blockA.s7 );    \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s0, acol0, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s1, acol1, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s2, acol2, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s3, acol3, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s4, acol4, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s5, acol5, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s6, acol6, _result );      \\ " << std::endl;
  stream << "            _result = mad( (float8)_blockB.s7, acol7, _result );      \\ " << std::endl;
  stream << "        } " << std::endl;
  stream << " " << std::endl;
  stream << "#define GEMM_TT(ALPHA1, BETA_NOT0, VECSCALAR, VECSIZE) \\ " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, 1, 1))) \\ " << std::endl;
  stream << "__kernel void intelblas_gemm_image_32_1_TT_ ##VECSCALAR ##_ ##ALPHA1 ##_ ##BETA_NOT0 ##_float( \\ " << std::endl;
  stream << "    __read_only image2d_t A, \\ " << std::endl;
  stream << "    MATB_PARAMETER, \\ " << std::endl;
  stream << "    MATC_PARAMETER, \\ " << std::endl;
  stream << "    float alpha, \\ " << std::endl;
  stream << "    float beta, \\ " << std::endl;
  stream << "    int padded_k, \\ " << std::endl;
  stream << "    int k, \\ " << std::endl;
  stream << "    int index) \\ " << std::endl;
  stream << "{ \\ " << std::endl;
  stream << "    const int group_x = get_group_id(0); \\ " << std::endl;
  stream << "    const int group_y = get_group_id(1); \\ " << std::endl;
  stream << "    float8 blockAxB00 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB01 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB02 = 0.0f; \\ " << std::endl;
  stream << "    float8 blockAxB03 = 0.0f; \\ " << std::endl;
  stream << "    int2    coordA = (int2)( group_y * TILE_M * sizeof(uint), 0 ); \\ " << std::endl;
  stream << "    int2    coordB = (int2)( 0, ( group_x * TILE_N )); \\ " << std::endl;
  stream << "    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST; \\ " << std::endl;
  stream << "    do \\ " << std::endl;
  stream << "    { \\ " << std::endl;
  stream << "        float8 blockB00;             \\ " << std::endl;
  stream << "        BLOCKB_READ8(blockB00, B, coordB); \\ " << std::endl;
  stream << "        int2    coordATemp = coordA; \\ " << std::endl;
  stream << "        float8 blockA00 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint); \\ " << std::endl;
  stream << "        float8 blockA01 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint); \\ " << std::endl;
  stream << "        float8 blockA02 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordATemp.x += 8 * sizeof(uint); \\ " << std::endl;
  stream << "        float8 blockA03 = as_float8( intel_sub_group_block_read8( A, coordATemp ) );    coordA.y += TILE_K; \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB00, blockA00 , blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB01, blockA01 , blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB02, blockA02 , blockB00 ); \\ " << std::endl;
  stream << "        MULTIPLY_BLOCKS_8x8( blockAxB03, blockA03 , blockB00 ); \\ " << std::endl;
  stream << "    } \\ " << std::endl;
  stream << "    while( coordB.x < padded_k / VECSIZE ); \\ " << std::endl;
  stream << "    GEMM_OUTPUT(ALPHA1, BETA_NOT0);\\ " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        blockB00.s0123 = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        blockB00.s4567 = read_imagef(B, _coordBTemp); _coordB.x += 2; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __read_only image2d_t B " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_TT(1, 0, VEC4, 4) " << std::endl;
  stream << "GEMM_TT(1, 1, VEC4, 4) " << std::endl;
  stream << "GEMM_TT(0, 0, VEC4, 4) " << std::endl;
  stream << "GEMM_TT(0, 1, VEC4, 4) " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        _blockb = *(__global float8*)&_B[_coordBTemp.y * ldb + _coordBTemp.x + offB];\\ " << std::endl;
  stream << "        _coordB.x += TILE_K; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __global float *B, int offB, int ldb " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_TT(1, 0, BUFFER, 1) " << std::endl;
  stream << "GEMM_TT(1, 1, BUFFER, 1) " << std::endl;
  stream << "GEMM_TT(0, 0, BUFFER, 1) " << std::endl;
  stream << "GEMM_TT(0, 1, BUFFER, 1) " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << "#define BLOCKB_READ8(_blockb, _B, _coordB) \\ " << std::endl;
  stream << "        int2 _coordBTemp = _coordB; \\ " << std::endl;
  stream << "        _coordBTemp.y += get_local_id(0); \\ " << std::endl;
  stream << "        float4 temp; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s0 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s1 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s2 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s3 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s4 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s5 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s6 = temp.s0; \\ " << std::endl;
  stream << "        temp = read_imagef(B, _coordBTemp); _coordBTemp.x += 1; \\ " << std::endl;
  stream << "        _blockb.s7 = temp.s0; \\ " << std::endl;
  stream << "        _coordB.x += 8; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MATB_PARAMETER __read_only image2d_t B " << std::endl;
  stream << " " << std::endl;
  stream << "GEMM_TT(1, 0, SCALAR, 1) " << std::endl;
  stream << "GEMM_TT(1, 1, SCALAR, 1) " << std::endl;
  stream << "GEMM_TT(0, 0, SCALAR, 1) " << std::endl;
  stream << "GEMM_TT(0, 1, SCALAR, 1) " << std::endl;
  stream << "#undef BLOCKB_READ8 " << std::endl;
  stream << "#undef MATB_PARAMETER " << std::endl;
  stream << " " << std::endl;
  stream << "#undef MULTIPLY_BLOCKS_8x8 " << std::endl;
  stream << "#undef TRANSPOSE_BLOCK_8 " << std::endl;
  stream << " " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;

  stream << "__kernel void gemm_buffer_copy_image_float( " << std::endl;
  stream << "    __global float* A, " << std::endl;
  stream << "    __write_only image2d_t ImA, " << std::endl;
  stream << "    int offA, " << std::endl;
  stream << "    int width, " << std::endl;
  stream << "    int height, " << std::endl;
  stream << "    int ldA) " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int gidx = get_global_id(0); " << std::endl;
  stream << "    const int gidy = get_global_id(1); " << std::endl;
  stream << "    int2 coord_dst = (int2)(gidx, gidy); " << std::endl;
  stream << "    __global float* A_off = A + offA; " << std::endl;
  stream << "    float srcA = A_off[gidy * ldA + gidx]; " << std::endl;
  stream << "    write_imagef(ImA, coord_dst, (float4)srcA); " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;

  stream << "__kernel void gemm_buffer_copy_image_ui( " << std::endl;
  stream << "    __global float* A, " << std::endl;
  stream << "    __write_only image2d_t ImA, " << std::endl;
  stream << "    int offA, " << std::endl;
  stream << "    int width, " << std::endl;
  stream << "    int height, " << std::endl;
  stream << "    int ldA) " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int gidx = get_global_id(0); " << std::endl;
  stream << "    const int gidy = get_global_id(1); " << std::endl;
  stream << "    int2 coord_dst = (int2)(gidx, gidy); " << std::endl;
  stream << "    if (gidx >= width || gidy >= height) { " << std::endl;
  stream << "      write_imageui(ImA, coord_dst, (uint4)0); " << std::endl;
  stream << "      return; " << std::endl;
  stream << "    } " << std::endl;
  stream << "    __global float* A_off = A + offA; " << std::endl;
  stream << "    uint4 srcA = convert_uint4(as_uchar4(A_off[gidy * ldA + gidx])); " << std::endl;
  stream << "    write_imageui(ImA, coord_dst, srcA); " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;

  return stream.str();
}

void gpu_gemm_copy_buffer_to_image(driver::CommandQueue & queue, driver::Program const & program, runtime::execution_handler const & control, cl_mem *image, cl_mem buffer, int offset,
                                   bool is_matrix_a, bool transpose, bool padding, int padded_height, int padded_width, int height, int width, int ld) {
  namespace drv = driver;
  (void) control;
  if (!is_matrix_a && transpose) {
    if(ld == width) {
      size_t origin[] = {0, 0, 0};
      size_t region[] = {(size_t)width, (size_t)height, 1};
      if(drv::dispatch::clEnqueueCopyBufferToImage(queue.handle().cl(),
                                 buffer, *image, sizeof(float) * offset,
                                 origin, region, 0,
                                 NULL, NULL))
        exit(0);
      return;
    }

    std::string kernel_name("gemm_buffer_copy_image_float");
    std::vector<driver::Kernel> kernels;
    kernels.push_back(driver::Kernel(program, kernel_name.c_str()));

    driver::Kernel & kernel = kernels[0];

    unsigned int n_arg = 0;

    kernel.setArg(n_arg++, buffer);
    kernel.setArg(n_arg++, *image);
    kernel.setSizeArg(n_arg++, offset);
    kernel.setSizeArg(n_arg++, width);
    kernel.setSizeArg(n_arg++, height);
    kernel.setSizeArg(n_arg++, ld);

    size_t global_copy[2];
    global_copy[0] = width;
    global_copy[1] = height;
    if(drv::dispatch::clEnqueueNDRangeKernel(queue.handle().cl(), kernel.handle().cl(), 2, NULL, global_copy, NULL, 0, NULL, NULL) != CL_SUCCESS)
      exit(0);
    return;
  }

  if (!padding && ld == width) {
    size_t origin[] = {0, 0, 0};
    size_t region[] = {(size_t)width, (size_t)height, 1};
    if(drv::dispatch::clEnqueueCopyBufferToImage(queue.handle().cl(),
                               buffer, *image, sizeof(float) * offset,
                               origin, region, 0, NULL, NULL))
      exit(0);
    return;
  }

  std::string kernel_name("gemm_buffer_copy_image_ui");
  std::vector<driver::Kernel> kernels;
  kernels.push_back(driver::Kernel(program, kernel_name.c_str()));

  driver::Kernel & kernel = kernels[0];

  unsigned int n_arg = 0;

  kernel.setArg(n_arg++, buffer);
  kernel.setArg(n_arg++, *image);
  kernel.setSizeArg(n_arg++, offset);
  kernel.setSizeArg(n_arg++, width);
  kernel.setSizeArg(n_arg++, height);
  kernel.setSizeArg(n_arg++, ld);

  size_t global_copy[2];
  global_copy[0] = padding ? padded_width : width;
  global_copy[1] = padding ? padded_height : height;

  if(drv::dispatch::clEnqueueNDRangeKernel(queue.handle().cl(), kernel.handle().cl(), 2, NULL, global_copy, NULL, 0, NULL, NULL) != CL_SUCCESS)
    exit(0);
}

void intelblas_gemm_image::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  (void) suffix;
  namespace drv = driver;
  //Get GEMM info
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(control.x(), args, A_trans_);
  int_t M = MNK[0], N = MNK[1], K = MNK[2];

  int offA = args.A->array.start, offB = args.B->array.start, offC = args.C->array.start;
  int ldA = args.A->ld[1];
  int ldB = args.B->ld[1];
  int ldC = args.C->ld[1];
  //Default order in isaac is column major.
  //This kernel is implemented in row major.
  //Need to swap matrix A and B each time.
  std::swap(args.A, args.B);
  std::swap(offA, offB);
  std::swap(ldA, ldB);
  std::swap(M, N);

  bool transA, transB;

  if(args.type == GEMM_NN) {
    transA = false;
    transB = false;
  } else if(args.type == GEMM_NT) {
    transA = true;
    transB = false;
  } else if(args.type == GEMM_TN) {
    transA = false;
    transB = true;
  } else {
    transA = true;
    transB = true;
  }

  int widthA = (transA == false) ? K : M;
  int heightA = (transA == false) ? M : K;
  int widthB = (transB == false) ? N : K;
  int heightB = (transB == false) ? K : N;

  int A_start_x = 0, A_start_y = 0, B_start_x = 0, B_start_y = 0, C_start_x = 0, C_start_y = 0;
  int blocksize = 1024;
  int blockA_width = blocksize;
  int blockA_height = blocksize;
  int blockB_width = blocksize;
  int blockB_height = blocksize;
  int blockC_width = blocksize;
  int blockC_height = blocksize;
  
  cl_int err;
  cl_mem ImA = NULL, ImB = NULL;
  cl_image_desc desc;
  cl_image_format format;
  memset(&desc, 0, sizeof(desc));

  desc.image_type = CL_MEM_OBJECT_IMAGE2D;  
  format.image_channel_data_type = CL_UNSIGNED_INT8;
  format.image_channel_order = CL_RGBA;
  desc.image_width = blocksize;
  desc.image_height = blocksize;
  
  ImA = drv::dispatch::clCreateImage(program.context().handle().cl(), CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
  if(err != CL_SUCCESS)
    exit(0);
  // if B is not transposed, use image object of B.
  // if B is transposed and ldB == widthB, use buffer object of B.
  // if B is transposed and ldB > widthB, use image object of B since kernel may access uninitialized 
  // element of B when using buffer object of B and it will cause incorrect results.
  if(transB == false) {
    ImB = drv::dispatch::clCreateImage(program.context().handle().cl(), CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
    if(err != CL_SUCCESS)
      exit(0);
  } else if(ldB > widthB) {
    format.image_channel_data_type = CL_FLOAT;
    format.image_channel_order = CL_R;
    ImB = drv::dispatch::clCreateImage(program.context().handle().cl(), CL_MEM_READ_WRITE, &format, &desc, NULL, &err);
    if(err != CL_SUCCESS)
      exit(0);
  }
  std::string kernel_name("intelblas_gemm_image_32_1_");

  if(transA == false)
    kernel_name += "N";
  else
    kernel_name += "T";

  if(transB == false)
    kernel_name += "N_";
  else {
    kernel_name += "T_";
    if(ldB > widthB)
      kernel_name += "SCALAR_";
    else
      kernel_name += "BUFFER_";
  }

  if(args.alpha.values().float32 == 1.0f)
    kernel_name += "1_";
  else
    kernel_name += "0_";

  if(args.beta.values().float32 == 0.0f)
    kernel_name += "0";
  else
    kernel_name += "1";
  kernel_name += "_float";

  driver::Kernel kernel(program, kernel_name.c_str());

  while(C_start_y < M) { 
    blockC_width = std::min((int)N - C_start_x, blocksize);
    blockC_height = std::min((int)M - C_start_y, blocksize);

    int index = 0;
    for(int k = 0; k < K; k += blocksize) {
      blockA_width = std::min(widthA - A_start_x, blocksize);
      blockA_height = std::min(heightA - A_start_y, blocksize);
      blockB_width = std::min(widthB - B_start_x, blocksize);
      blockB_height = std::min(heightB - B_start_y, blocksize);
      int block_K = std::min((int)K - k, blocksize);

      int padded_k = block_K + ((block_K & 7) ? (8 - (block_K & 7)) : 0);
      int imageA_w = (transA == false) ? padded_k : blockA_width;
      int imageA_h = (transA == false) ? blockA_height : padded_k;
      int imageB_w = (transB == false) ? blockB_width : padded_k;
      int imageB_h = (transB == false) ? padded_k : blockB_height;

      int offsetA = offA + A_start_y * ldA + A_start_x;
      int offsetB = offB + B_start_y * ldB + B_start_x;
      int offsetC = offC + C_start_y * ldC + C_start_x;

      if(transB == false) {
        bool padding_A = false;
        bool padding_B = false;

        if (M * K < N * K)
          padding_B = true;
        else
          padding_A = true;
        gpu_gemm_copy_buffer_to_image(queue, program, control, &ImA, args.A->array.handle.cl, offsetA, true, transA != false, padding_A, imageA_h, imageA_w, blockA_height, blockA_width, ldA);
        gpu_gemm_copy_buffer_to_image(queue, program, control, &ImB, args.B->array.handle.cl, offsetB, false, false, padding_B, imageB_h, imageB_w, blockB_height, blockB_width, ldB);
      } else {
        gpu_gemm_copy_buffer_to_image(queue, program, control, &ImA, args.A->array.handle.cl, offsetA, true, transA != false, true, imageA_h, imageA_w, blockA_height, blockA_width, ldA);
        if(ldB > widthB)
          gpu_gemm_copy_buffer_to_image(queue, program, control, &ImB, args.B->array.handle.cl, offsetB, false, true, false, imageB_h, imageB_w, blockB_height, blockB_width, ldB);
      }

      unsigned int n_arg = 0;

      kernel.setArg(n_arg++, ImA);
      if(transB == false || ldB > widthB)
        kernel.setArg(n_arg++, ImB);
      else {
        kernel.setArg(n_arg++, args.B->array.handle.cl);
        kernel.setSizeArg(n_arg++, offsetB);
        kernel.setSizeArg(n_arg++, ldB);
      }
      int sub_M = (transA == false) ? blockA_height : blockA_width;
      int sub_N = (transB == false) ? blockB_width : blockB_height;
      kernel.setArg(n_arg++, args.C->array.handle.cl);
      kernel.setSizeArg(n_arg++, offsetC);
      kernel.setSizeArg(n_arg++, sub_M);
      kernel.setSizeArg(n_arg++, sub_N);
      kernel.setSizeArg(n_arg++, ldC);
      kernel.setArg(n_arg++, args.alpha);
      kernel.setArg(n_arg++, args.beta);
      kernel.setSizeArg(n_arg++, padded_k);
      if(transB != false)
        kernel.setSizeArg(n_arg++, block_K);
      kernel.setSizeArg(n_arg++, index);

      driver::NDRange local(8, 1);
      driver::NDRange global((size_t)( blockC_width + 7 ) & ~7, (size_t)(blockC_height + 31) / 32);

      control.execution_options().enqueue(program.context(), kernel, global, local);

      if(transA == false)
        A_start_x += blockA_width;
      else
        A_start_y += blockA_height;

      if(transB == false)
        B_start_y += blockB_height;
      else
        B_start_x += blockB_width;

      index = 1;
    }
    C_start_x += blockC_width;
    if(transA == false)
      A_start_x = 0;
    else
      A_start_y = 0;
    if(transB == false) {
      B_start_x += blockB_width;
      B_start_y = 0;
    } else {
      B_start_y += blockB_height;
      B_start_x = 0;
    }
    if(C_start_x >= N) {
      C_start_x = 0;
      B_start_x = 0;
      B_start_y = 0;
      C_start_y += blockC_height;
      if(transA == false)
        A_start_y += blockA_height;
      else
        A_start_x += blockA_width;
    }
  }

  if (ImA)
    drv::dispatch::clReleaseMemObject(ImA);
  if (ImB)
    drv::dispatch::clReleaseMemObject(ImB);
}

/* -------------------------------------------- */

/* ------------------ INTELBLAS_buffer ------------------ */
intelblas_gemm::intelblas_gemm(char A_trans, char B_trans): A_trans_(A_trans), B_trans_(B_trans), init_(true)
{ }

int intelblas_gemm::is_invalid(expression_tree const  &, driver::Device const & device) const
{ return (init_ && device.backend()==driver::OPENCL)?0:-1; }

std::vector<int_t> intelblas_gemm::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

expression_type intelblas_gemm::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}

std::string intelblas_gemm::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const & symbols) const
{
  (void) suffix;
  (void) symbols;

  using std::string;
  using tools::to_string;

  driver::backend_type backend = device.backend();
  kernel_generation_stream stream(backend);

  numeric_type dtype = tree.dtype();
  std::string sdtype = to_string(dtype);
  std::string sdtype2 = sdtype + "2";
  std::string sdtype4 = sdtype + "4";
  std::string sdtype8 = sdtype + "8";
  std::string sdtype16 = sdtype + "16";

  stream << "#define VEC_SIZE        4 " << std::endl;
  stream << "#define LWG_HEIGHT      4 " << std::endl;
  stream << "#define TILE_M          8 " << std::endl;
  stream << "#define TILE_K          16 " << std::endl;
  stream << "#define TILE_N          32 " << std::endl;
  stream << " " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1))) " << std::endl;
  stream << "__kernel void intelblas_gemm_buffer_NN_sp_" << suffix << "( " << std::endl;
  stream << "    const __global " << sdtype << " *src0, int off0, " << std::endl;
  stream << "    const __global " << sdtype << " *src1, int off1, " << std::endl;
  stream << "    __global " << sdtype << " *dst, int offd, " << std::endl;
  stream << "    int M, " << std::endl;
  stream << "    int N, " << std::endl;
  stream << "    int K, " << std::endl;
  stream << "    float alpha, " << std::endl;
  stream << "    float beta, " << std::endl;
  stream << "    int ldA, " << std::endl;
  stream << "    int ldB, " << std::endl;
  stream << "    int ldC, " << std::endl;
  stream << "    int start_index, " << std::endl;
  stream << "    int stride) " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int group_x = get_group_id(0); " << std::endl;
  stream << "    const int group_y = get_group_id(1); " << std::endl;
  stream << "    const int local_x = get_local_id(0); " << std::endl;
  stream << "    const int local_y = get_local_id(1); " << std::endl;
  stream << "    const int global_x = get_global_id(0); " << std::endl;
  stream << "    const int global_y = get_global_id(1); " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " brow; " << std::endl;
  stream << "    " << sdtype2 << " arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7; " << std::endl;
  stream << " " << std::endl;
  stream << "    __global " << sdtype << " *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + start_index + off0; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1; " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " dot00 = (start_index != 0) ? vload4(0, dst_write0) : (" << sdtype << ")beta * vload4(0, dst_write0); " << std::endl;
  stream << "    " << sdtype4 << " dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 1 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 2 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 3 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 4 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 5 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 6 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 7 * ldC); " << std::endl;
  stream << "    " << std::endl;
  stream << "    int end_index = min(start_index + stride, K); " << std::endl;
  stream << "    int w = start_index; " << std::endl;
  stream << "    while( w + TILE_K <= end_index ) { " << std::endl;
  stream << "        arow0 = (" << sdtype << ")alpha * vload2(0, src0_read + 0 * ldA); " << std::endl;
  stream << "        arow1 = (" << sdtype << ")alpha * vload2(0, src0_read + 1 * ldA); " << std::endl;
  stream << "        arow2 = (" << sdtype << ")alpha * vload2(0, src0_read + 2 * ldA); " << std::endl;
  stream << "        arow3 = (" << sdtype << ")alpha * vload2(0, src0_read + 3 * ldA); " << std::endl;
  stream << "        arow4 = (" << sdtype << ")alpha * vload2(0, src0_read + 4 * ldA); " << std::endl;
  stream << "        arow5 = (" << sdtype << ")alpha * vload2(0, src0_read + 5 * ldA); " << std::endl;
  stream << "        arow6 = (" << sdtype << ")alpha * vload2(0, src0_read + 6 * ldA); " << std::endl;
  stream << "        arow7 = (" << sdtype << ")alpha * vload2(0, src0_read + 7 * ldA); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( index, suffix )   \\ " << std::endl;
  stream << "        brow = vload4(0, src1_read0);  src1_read0 += ldB; \\ " << std::endl;
  stream << "        dot00 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow0.s##suffix, index )), brow, dot00 ); \\ " << std::endl;
  stream << "        dot01 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow1.s##suffix, index )), brow, dot01 ); \\ " << std::endl;
  stream << "        dot02 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow2.s##suffix, index )), brow, dot02 ); \\ " << std::endl;
  stream << "        dot03 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow3.s##suffix, index )), brow, dot03 ); \\ " << std::endl;
  stream << "        dot04 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow4.s##suffix, index )), brow, dot04 ); \\ " << std::endl;
  stream << "        dot05 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow5.s##suffix, index )), brow, dot05 ); \\ " << std::endl;
  stream << "        dot06 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow6.s##suffix, index )), brow, dot06 ); \\ " << std::endl;
  stream << "        dot07 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow7.s##suffix, index )), brow, dot07 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 1); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << " " << std::endl;
  stream << "        src0_read += TILE_K; " << std::endl;
  stream << "        w += TILE_K; " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    vstore4(dot00, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot01, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot02, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot03, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot04, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot05, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot06, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "    vstore4(dot07, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#undef VEC_SIZE " << std::endl;
  stream << "#undef LWG_HEIGHT " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;

  stream << "#define VEC_SIZE        4 " << std::endl;
  stream << "#define LWG_HEIGHT      4 " << std::endl;
  stream << "#define TILE_M          8 " << std::endl;
  stream << "#define TILE_K          16 " << std::endl;
  stream << "#define TILE_N          32 " << std::endl;
  stream << " " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1))) " << std::endl;
  stream << "__kernel void intelblas_gemm_buffer_NN_" << suffix << "( " << std::endl;
  stream << "    const __global " << sdtype << " *src0, int off0, " << std::endl;
  stream << "    const __global " << sdtype << " *src1, int off1, " << std::endl;
  stream << "    __global " << sdtype << " *dst, int offd, " << std::endl;
  stream << "    int M, " << std::endl;
  stream << "    int N, " << std::endl;
  stream << "    int K, " << std::endl;
  stream << "    float alpha, " << std::endl;
  stream << "    float beta, " << std::endl;
  stream << "    int ldA, " << std::endl;
  stream << "    int ldB, " << std::endl;
  stream << "    int ldC, " << std::endl;
  stream << "    int start_index, " << std::endl;
  stream << "    int stride) " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int group_x = get_group_id(0); " << std::endl;
  stream << "    const int group_y = get_group_id(1); " << std::endl;
  stream << "    const int local_x = get_local_id(0); " << std::endl;
  stream << "    const int local_y = get_local_id(1); " << std::endl;
  stream << "    const int global_x = get_global_id(0); " << std::endl;
  stream << "    const int global_y = get_global_id(1); " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " brow; " << std::endl;
  stream << "    " << sdtype2 << " arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7; " << std::endl;
  stream << " " << std::endl;
  stream << "    __global " << sdtype << " *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + start_index + off0; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1; " << std::endl;
  stream << " " << std::endl;
  stream << "    int border = -(group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M); " << std::endl;
  stream << " " << std::endl;
  stream << "    int row0 = mad24(global_y, TILE_M, 0) < M ? 0 : border; " << std::endl;
  stream << "    int row1 = mad24(global_y, TILE_M, 1) < M ? 1 : border; " << std::endl;
  stream << "    int row2 = mad24(global_y, TILE_M, 2) < M ? 2 : border; " << std::endl;
  stream << "    int row3 = mad24(global_y, TILE_M, 3) < M ? 3 : border; " << std::endl;
  stream << "    int row4 = mad24(global_y, TILE_M, 4) < M ? 4 : border; " << std::endl;
  stream << "    int row5 = mad24(global_y, TILE_M, 5) < M ? 5 : border; " << std::endl;
  stream << "    int row6 = mad24(global_y, TILE_M, 6) < M ? 6 : border; " << std::endl;
  stream << "    int row7 = mad24(global_y, TILE_M, 7) < M ? 7 : border; " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " dot00 = (start_index != 0) ? vload4(0, dst_write0) : (" << sdtype << ")beta * vload4(0, dst_write0); " << std::endl;
  stream << "    " << sdtype4 << " dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 1 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 2 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 3 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 4 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 5 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 6 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 7 * ldC); " << std::endl;
  stream << "    " << std::endl;
  stream << "    int end_index = min(start_index + stride, K); " << std::endl;
  stream << "    int w = start_index; " << std::endl;
  stream << "    while( w + TILE_K <= end_index ) { " << std::endl;
  stream << "        arow0 = (" << sdtype << ")alpha * vload2(0, src0_read + row0 * ldA); " << std::endl;
  stream << "        arow1 = (" << sdtype << ")alpha * vload2(0, src0_read + row1 * ldA); " << std::endl;
  stream << "        arow2 = (" << sdtype << ")alpha * vload2(0, src0_read + row2 * ldA); " << std::endl;
  stream << "        arow3 = (" << sdtype << ")alpha * vload2(0, src0_read + row3 * ldA); " << std::endl;
  stream << "        arow4 = (" << sdtype << ")alpha * vload2(0, src0_read + row4 * ldA); " << std::endl;
  stream << "        arow5 = (" << sdtype << ")alpha * vload2(0, src0_read + row5 * ldA); " << std::endl;
  stream << "        arow6 = (" << sdtype << ")alpha * vload2(0, src0_read + row6 * ldA); " << std::endl;
  stream << "        arow7 = (" << sdtype << ")alpha * vload2(0, src0_read + row7 * ldA); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( index, suffix )   \\ " << std::endl;
  stream << "        brow = vload4(0, src1_read0);  src1_read0 += ldB; \\ " << std::endl;
  stream << "        dot00 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow0.s##suffix, index )), brow, dot00 ); \\ " << std::endl;
  stream << "        dot01 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow1.s##suffix, index )), brow, dot01 ); \\ " << std::endl;
  stream << "        dot02 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow2.s##suffix, index )), brow, dot02 ); \\ " << std::endl;
  stream << "        dot03 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow3.s##suffix, index )), brow, dot03 ); \\ " << std::endl;
  stream << "        dot04 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow4.s##suffix, index )), brow, dot04 ); \\ " << std::endl;
  stream << "        dot05 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow5.s##suffix, index )), brow, dot05 ); \\ " << std::endl;
  stream << "        dot06 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow6.s##suffix, index )), brow, dot06 ); \\ " << std::endl;
  stream << "        dot07 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow7.s##suffix, index )), brow, dot07 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 1); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << " " << std::endl;
  stream << "        src0_read += TILE_K; " << std::endl;
  stream << "        w += TILE_K; " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(w < end_index) { " << std::endl;
  stream << "        arow0.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row0 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow0.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row0 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow1.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row1 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow1.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row1 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow2.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row2 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow2.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row2 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow3.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row3 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow3.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row3 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow4.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row4 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow4.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row4 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow5.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row5 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow5.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row5 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow6.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row6 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow6.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row6 * ldA)[1] : 0.0f; " << std::endl;
  stream << "        arow7.x = ((w + local_x * 2) < K) ? (" << sdtype << ")alpha * (src0_read + row7 * ldA)[0] : 0.0f; " << std::endl;
  stream << "        arow7.y = ((w + local_x * 2 + 1) < K) ? (" << sdtype << ")alpha * (src0_read + row7 * ldA)[1] : 0.0f; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( index, suffix )   \\ " << std::endl;
  stream << "        brow = (w < K) ? vload4(0, src1_read0) : (" << sdtype << ")0.0f;  src1_read0 += ldB; w++; \\ " << std::endl;
  stream << "        dot00 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow0.s##suffix, index )), brow, dot00 ); \\ " << std::endl;
  stream << "        dot01 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow1.s##suffix, index )), brow, dot01 ); \\ " << std::endl;
  stream << "        dot02 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow2.s##suffix, index )), brow, dot02 ); \\ " << std::endl;
  stream << "        dot03 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow3.s##suffix, index )), brow, dot03 ); \\ " << std::endl;
  stream << "        dot04 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow4.s##suffix, index )), brow, dot04 ); \\ " << std::endl;
  stream << "        dot05 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow5.s##suffix, index )), brow, dot05 ); \\ " << std::endl;
  stream << "        dot06 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow6.s##suffix, index )), brow, dot06 ); \\ " << std::endl;
  stream << "        dot07 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle( arow7.s##suffix, index )), brow, dot07 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(0, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(1, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(2, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(3, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(4, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(5, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(6, 1); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 0); " << std::endl;
  stream << "        MM_DOT_PRODUCT(7, 1); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(global_x * 4 < N && global_y * 8 < M) { " << std::endl;
  stream << "        if(mad24(global_x, 4, 3) < N) { " << std::endl;
  stream << "            vstore4(dot00, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 2) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); " << std::endl;
  stream << "            dst_write0[2] = dot00.z; " << std::endl;
  stream << "            dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { " << std::endl;
  stream << "                vstore2(dot01.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot01.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { " << std::endl;
  stream << "                vstore2(dot02.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot02.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { " << std::endl;
  stream << "                vstore2(dot03.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot03.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { " << std::endl;
  stream << "                vstore2(dot04.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot04.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { " << std::endl;
  stream << "                vstore2(dot05.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot05.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { " << std::endl;
  stream << "                vstore2(dot06.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot06.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { " << std::endl;
  stream << "                vstore2(dot07.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot07.z; " << std::endl;
  stream << "            } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 1) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); } " << std::endl;
  stream << "        } else { " << std::endl;
  stream << "            dst_write0[0] = dot00.x; dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; } " << std::endl;
  stream << "        } " << std::endl;
  stream << "    } " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#undef VEC_SIZE " << std::endl;
  stream << "#undef LWG_HEIGHT " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;
  stream << " " << std::endl;
  stream << "#define VEC_SIZE        1 " << std::endl;
  stream << "#define LWG_HEIGHT      16 " << std::endl;
  stream << "#define TILE_M          8 " << std::endl;
  stream << "#define TILE_K          32 " << std::endl;
  stream << "#define TILE_N          8 " << std::endl;
  stream << "#define SLM_BLOCK       512 " << std::endl;
  stream << " " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1))) " << std::endl;
  stream << "__kernel void intelblas_gemm_buffer_NT_" << suffix << "( " << std::endl;
  stream << "    const __global " << sdtype << " *src0, int off0, " << std::endl;
  stream << "    const __global " << sdtype << " *src1, int off1, " << std::endl;
  stream << "    __global " << sdtype << " *dst, int offd, " << std::endl;
  stream << "    int M, " << std::endl;
  stream << "    int N, " << std::endl;
  stream << "    int K, " << std::endl;
  stream << "    float alpha, " << std::endl;
  stream << "    float beta, " << std::endl;
  stream << "    int ldA, " << std::endl;
  stream << "    int ldB, " << std::endl;
  stream << "    int ldC) " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int group_x = get_group_id(0); " << std::endl;
  stream << "    const int group_y = get_group_id(1); " << std::endl;
  stream << "    const int local_x = get_local_id(0); " << std::endl;
  stream << "    const int local_y = get_local_id(1); " << std::endl;
  stream << "    const int global_x = get_global_id(0); " << std::endl;
  stream << "    const int global_y = get_global_id(1); " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype8 << " dot00 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot01 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot02 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot03 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot04 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot05 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot06 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot07 = 0.f; " << std::endl;
  stream << "     " << std::endl;
  stream << "    " << sdtype4 << " brow0; " << std::endl;
  stream << "    " << sdtype4 << " brow1; " << std::endl;
  stream << "    " << sdtype4 << " brow2; " << std::endl;
  stream << "    " << sdtype4 << " brow3; " << std::endl;
  stream << "    " << sdtype4 << " brow4; " << std::endl;
  stream << "    " << sdtype4 << " brow5; " << std::endl;
  stream << "    " << sdtype4 << " brow6; " << std::endl;
  stream << "    " << sdtype4 << " brow7; " << std::endl;
  stream << "     " << std::endl;
  stream << "    __global " << sdtype << " *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + off0; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src1_read0 = src1 + ( group_x * TILE_N ) * ldB + off1; " << std::endl;
  stream << " " << std::endl;
  stream << "    __local " << sdtype << " slm_brow[8 * SLM_BLOCK]; " << std::endl;
  stream << "    __local " << sdtype << "* slm_brow0; " << std::endl;
  stream << " " << std::endl;
  stream << "    int local_index = mad24(local_y, 8, local_x) * 4; " << std::endl;
  stream << "    int w; " << std::endl;
  stream << "    for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) { " << std::endl;
  stream << "        barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(0, ldB, local_index)), 0, slm_brow + mad24(0, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(1, ldB, local_index)), 0, slm_brow + mad24(1, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(2, ldB, local_index)), 0, slm_brow + mad24(2, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(3, ldB, local_index)), 0, slm_brow + mad24(3, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(4, ldB, local_index)), 0, slm_brow + mad24(4, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(5, ldB, local_index)), 0, slm_brow + mad24(5, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(6, ldB, local_index)), 0, slm_brow + mad24(6, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        vstore4(vload4(0, src1_read0 + mad24(7, ldB, local_index)), 0, slm_brow + mad24(7, SLM_BLOCK, local_index)); " << std::endl;
  stream << "        barrier(CLK_LOCAL_MEM_FENCE); " << std::endl;
  stream << " " << std::endl;
  stream << "        slm_brow0 = slm_brow + local_x * (TILE_K / 8); " << std::endl;
  stream << "        w = b_tile; " << std::endl;
  stream << "        int end_w = min(b_tile + SLM_BLOCK, K); " << std::endl;
  stream << "        while( w + TILE_K <= end_w ) { " << std::endl;
  stream << "            " << sdtype4 << " arow; " << std::endl;
  stream << "                             " << std::endl;
  stream << "            brow0 = vload4(0, slm_brow0 + 0 * SLM_BLOCK); " << std::endl;
  stream << "            brow1 = vload4(0, slm_brow0 + 1 * SLM_BLOCK); " << std::endl;
  stream << "            brow2 = vload4(0, slm_brow0 + 2 * SLM_BLOCK); " << std::endl;
  stream << "            brow3 = vload4(0, slm_brow0 + 3 * SLM_BLOCK); " << std::endl;
  stream << "            brow4 = vload4(0, slm_brow0 + 4 * SLM_BLOCK); " << std::endl;
  stream << "            brow5 = vload4(0, slm_brow0 + 5 * SLM_BLOCK); " << std::endl;
  stream << "            brow6 = vload4(0, slm_brow0 + 6 * SLM_BLOCK); " << std::endl;
  stream << "            brow7 = vload4(0, slm_brow0 + 7 * SLM_BLOCK); " << std::endl;
  stream << "              " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _row, _dot )   \\ " << std::endl;
  stream << "            arow = vload4(0, src0_read + _row * ldA);                           \\ " << std::endl;
  stream << "            _dot = mad( (" << sdtype8 << ")(arow.x), (" << sdtype8 << ")(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \\ " << std::endl;
  stream << "            _dot = mad( (" << sdtype8 << ")(arow.y), (" << sdtype8 << ")(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \\ " << std::endl;
  stream << "            _dot = mad( (" << sdtype8 << ")(arow.z), (" << sdtype8 << ")(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \\ " << std::endl;
  stream << "            _dot = mad( (" << sdtype8 << ")(arow.w), (" << sdtype8 << ")(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot ); \\ " << std::endl;
  stream << "                         " << std::endl;
  stream << "            MM_DOT_PRODUCT( 0, dot00 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 1, dot01 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 2, dot02 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 3, dot03 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 4, dot04 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 5, dot05 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 6, dot06 ); " << std::endl;
  stream << "            MM_DOT_PRODUCT( 7, dot07 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "        " << std::endl;
  stream << "            src0_read += TILE_K; " << std::endl;
  stream << "            slm_brow0 += TILE_K; " << std::endl;
  stream << "            w += TILE_K; " << std::endl;
  stream << "        } " << std::endl;
  stream << "        src1_read0 += SLM_BLOCK; " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(w < K) { " << std::endl;
  stream << "        " << sdtype4 << " arow; " << std::endl;
  stream << " " << std::endl;
  stream << "#define READ_BROW(_brow, _row) \\ " << std::endl;
  stream << "        _brow = vload4(0, slm_brow0 + _row * SLM_BLOCK); \\ " << std::endl;
  stream << "        _brow.x = (mad24(local_x, 4, w) < K) ? _brow.x : 0.0f; \\ " << std::endl;
  stream << "        _brow.y = (mad24(local_x, 4, w + 1) < K) ? _brow.y : 0.0f; \\ " << std::endl;
  stream << "        _brow.z = (mad24(local_x, 4, w + 2) < K) ? _brow.z : 0.0f; \\ " << std::endl;
  stream << "        _brow.w = (mad24(local_x, 4, w + 3) < K) ? _brow.w : 0.0f; \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        READ_BROW(brow0, 0); " << std::endl;
  stream << "        READ_BROW(brow1, 1); " << std::endl;
  stream << "        READ_BROW(brow2, 2); " << std::endl;
  stream << "        READ_BROW(brow3, 3); " << std::endl;
  stream << "        READ_BROW(brow4, 4); " << std::endl;
  stream << "        READ_BROW(brow5, 5); " << std::endl;
  stream << "        READ_BROW(brow6, 6); " << std::endl;
  stream << "        READ_BROW(brow7, 7); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _row, _dot )   \\ " << std::endl;
  stream << "        arow = vload4(0, src0_read + _row * ldA);                           \\ " << std::endl;
  stream << "        arow.x = (mad24(local_x, 4, w) < K) ? arow.x : 0.0f; \\ " << std::endl;
  stream << "        arow.y = (mad24(local_x, 4, w + 1) < K) ? arow.y : 0.0f; \\ " << std::endl;
  stream << "        arow.z = (mad24(local_x, 4, w + 2) < K) ? arow.z : 0.0f; \\ " << std::endl;
  stream << "        arow.w = (mad24(local_x, 4, w + 3) < K) ? arow.w : 0.0f; \\ " << std::endl;
  stream << "        _dot = mad( (" << sdtype8 << ")(arow.x), (" << sdtype8 << ")(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \\ " << std::endl;
  stream << "        _dot = mad( (" << sdtype8 << ")(arow.y), (" << sdtype8 << ")(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \\ " << std::endl;
  stream << "        _dot = mad( (" << sdtype8 << ")(arow.z), (" << sdtype8 << ")(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \\ " << std::endl;
  stream << "        _dot = mad( (" << sdtype8 << ")(arow.w), (" << sdtype8 << ")(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot ); \\ " << std::endl;
  stream << "                         " << std::endl;
  stream << "        MM_DOT_PRODUCT( 0, dot00 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 1, dot01 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 2, dot02 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 3, dot03 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 4, dot04 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 5, dot05 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 6, dot06 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( 7, dot07 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "#define REDUCE(_dot) \\ " << std::endl;
  stream << "    _dot.s0 = intel_sub_group_shuffle(_dot.s0, 0) + intel_sub_group_shuffle(_dot.s0, 1) + intel_sub_group_shuffle(_dot.s0, 2) + intel_sub_group_shuffle(_dot.s0, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s0, 4) + intel_sub_group_shuffle(_dot.s0, 5) + intel_sub_group_shuffle(_dot.s0, 6) + intel_sub_group_shuffle(_dot.s0, 7); \\ " << std::endl;
  stream << "    _dot.s1 = intel_sub_group_shuffle(_dot.s1, 0) + intel_sub_group_shuffle(_dot.s1, 1) + intel_sub_group_shuffle(_dot.s1, 2) + intel_sub_group_shuffle(_dot.s1, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s1, 4) + intel_sub_group_shuffle(_dot.s1, 5) + intel_sub_group_shuffle(_dot.s1, 6) + intel_sub_group_shuffle(_dot.s1, 7); \\ " << std::endl;
  stream << "    _dot.s2 = intel_sub_group_shuffle(_dot.s2, 0) + intel_sub_group_shuffle(_dot.s2, 1) + intel_sub_group_shuffle(_dot.s2, 2) + intel_sub_group_shuffle(_dot.s2, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s2, 4) + intel_sub_group_shuffle(_dot.s2, 5) + intel_sub_group_shuffle(_dot.s2, 6) + intel_sub_group_shuffle(_dot.s2, 7); \\ " << std::endl;
  stream << "    _dot.s3 = intel_sub_group_shuffle(_dot.s3, 0) + intel_sub_group_shuffle(_dot.s3, 1) + intel_sub_group_shuffle(_dot.s3, 2) + intel_sub_group_shuffle(_dot.s3, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s3, 4) + intel_sub_group_shuffle(_dot.s3, 5) + intel_sub_group_shuffle(_dot.s3, 6) + intel_sub_group_shuffle(_dot.s3, 7); \\ " << std::endl;
  stream << "    _dot.s4 = intel_sub_group_shuffle(_dot.s4, 0) + intel_sub_group_shuffle(_dot.s4, 1) + intel_sub_group_shuffle(_dot.s4, 2) + intel_sub_group_shuffle(_dot.s4, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s4, 4) + intel_sub_group_shuffle(_dot.s4, 5) + intel_sub_group_shuffle(_dot.s4, 6) + intel_sub_group_shuffle(_dot.s4, 7); \\ " << std::endl;
  stream << "    _dot.s5 = intel_sub_group_shuffle(_dot.s5, 0) + intel_sub_group_shuffle(_dot.s5, 1) + intel_sub_group_shuffle(_dot.s5, 2) + intel_sub_group_shuffle(_dot.s5, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s5, 4) + intel_sub_group_shuffle(_dot.s5, 5) + intel_sub_group_shuffle(_dot.s5, 6) + intel_sub_group_shuffle(_dot.s5, 7); \\ " << std::endl;
  stream << "    _dot.s6 = intel_sub_group_shuffle(_dot.s6, 0) + intel_sub_group_shuffle(_dot.s6, 1) + intel_sub_group_shuffle(_dot.s6, 2) + intel_sub_group_shuffle(_dot.s6, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s6, 4) + intel_sub_group_shuffle(_dot.s6, 5) + intel_sub_group_shuffle(_dot.s6, 6) + intel_sub_group_shuffle(_dot.s6, 7); \\ " << std::endl;
  stream << "    _dot.s7 = intel_sub_group_shuffle(_dot.s7, 0) + intel_sub_group_shuffle(_dot.s7, 1) + intel_sub_group_shuffle(_dot.s7, 2) + intel_sub_group_shuffle(_dot.s7, 3) +  \\ " << std::endl;
  stream << "           intel_sub_group_shuffle(_dot.s7, 4) + intel_sub_group_shuffle(_dot.s7, 5) + intel_sub_group_shuffle(_dot.s7, 6) + intel_sub_group_shuffle(_dot.s7, 7); \\ " << std::endl;
  stream << "     " << std::endl;
  stream << "    REDUCE(dot00); " << std::endl;
  stream << "    REDUCE(dot01); " << std::endl;
  stream << "    REDUCE(dot02); " << std::endl;
  stream << "    REDUCE(dot03); " << std::endl;
  stream << "    REDUCE(dot04); " << std::endl;
  stream << "    REDUCE(dot05); " << std::endl;
  stream << "    REDUCE(dot06); " << std::endl;
  stream << "    REDUCE(dot07); " << std::endl;
  stream << "#undef REDUCE " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype << " output = 0.0f; " << std::endl;
  stream << "#define OUTPUT( _dot) \\ " << std::endl;
  stream << "    output = (local_x == 0) ? _dot.s0 : output; \\ " << std::endl;
  stream << "    output = (local_x == 1) ? _dot.s1 : output; \\ " << std::endl;
  stream << "    output = (local_x == 2) ? _dot.s2 : output; \\ " << std::endl;
  stream << "    output = (local_x == 3) ? _dot.s3 : output; \\ " << std::endl;
  stream << "    output = (local_x == 4) ? _dot.s4 : output; \\ " << std::endl;
  stream << "    output = (local_x == 5) ? _dot.s5 : output; \\ " << std::endl;
  stream << "    output = (local_x == 6) ? _dot.s6 : output; \\ " << std::endl;
  stream << "    output = (local_x == 7) ? _dot.s7 : output; \\ " << std::endl;
  stream << "    dst_write0[0] = mad(output, (" << sdtype << ")alpha, (" << sdtype << ")beta * dst_write0[0]); \\ " << std::endl;
  stream << "    dst_write0 += ldC; " << std::endl;
  stream << " " << std::endl;
  stream << "    if(global_x < N && global_y * 8 < M) { " << std::endl;
  stream << "        OUTPUT(dot00); " << std::endl;
  stream << "        if(mad24(global_y, 8, 1) < M) { OUTPUT(dot01); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 2) < M) { OUTPUT(dot02); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 3) < M) { OUTPUT(dot03); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 4) < M) { OUTPUT(dot04); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 5) < M) { OUTPUT(dot05); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 6) < M) { OUTPUT(dot06); } " << std::endl;
  stream << "        if(mad24(global_y, 8, 7) < M) { OUTPUT(dot07); } " << std::endl;
  stream << "    } " << std::endl;
  stream << "#undef OUTPUT " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#undef VEC_SIZE " << std::endl;
  stream << "#undef LWG_HEIGHT " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;
  stream << "#undef SLM_BLOCK " << std::endl;
  stream << " " << std::endl;
  stream << "#define VEC_SIZE        4 " << std::endl;
  stream << "#define LWG_HEIGHT      4 " << std::endl;
  stream << "#define TILE_M          8 " << std::endl;
  stream << "#define TILE_K          16 " << std::endl;
  stream << "#define TILE_N          32 " << std::endl;
  stream << " " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1))) " << std::endl;
  stream << "__kernel void intelblas_gemm_buffer_TN_" << suffix << "( " << std::endl;
  stream << "    const __global " << sdtype << " *src0, int off0, " << std::endl;
  stream << "    const __global " << sdtype << " *src1, int off1, " << std::endl;
  stream << "    __global " << sdtype << " *dst, int offd, " << std::endl;
  stream << "    int M, " << std::endl;
  stream << "    int N, " << std::endl;
  stream << "    int K, " << std::endl;
  stream << "    float alpha, " << std::endl;
  stream << "    float beta, " << std::endl;
  stream << "    int ldA, " << std::endl;
  stream << "    int ldB, " << std::endl;
  stream << "    int ldC, " << std::endl;
  stream << "    int start_index, " << std::endl;
  stream << "    int stride) " << std::endl;
  stream << " " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int group_x = get_group_id(0); " << std::endl;
  stream << "    const int group_y = get_group_id(1); " << std::endl;
  stream << "    const int local_x = get_local_id(0); " << std::endl;
  stream << "    const int local_y = get_local_id(1); " << std::endl;
  stream << "    const int global_x = get_global_id(0); " << std::endl;
  stream << "    const int global_y = get_global_id(1); " << std::endl;
  stream << "    " << std::endl;
  stream << "    " << sdtype4 << " brow; " << std::endl;
  stream << " " << std::endl;
  stream << "    __global " << sdtype << " *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * ldA + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1; " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " dot00 = (start_index != 0) ? vload4(0, dst_write0): (" << sdtype << ")beta * vload4(0, dst_write0); " << std::endl;
  stream << "    " << sdtype4 << " dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 1 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 2 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 3 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 4 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 5 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 6 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 7 * ldC); " << std::endl;
  stream << " " << std::endl;
  stream << "    int end_index = min(start_index + stride, K); " << std::endl;
  stream << "    while( start_index + TILE_K <= end_index ) { " << std::endl;
  stream << "        " << sdtype8 << " arow0 = (" << sdtype << ")alpha * vload8(0, src0_read); " << std::endl;
  stream << "        " << sdtype8 << " arow1 = (" << sdtype << ")alpha * vload8(0, src0_read + ldA); " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _arow, index ) \\ " << std::endl;
  stream << "        brow = vload4(0, src1_read0);  src1_read0 += ldB; \\ " << std::endl;
  stream << "        dot00 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s0, index)), brow, dot00 ); \\ " << std::endl;
  stream << "        dot01 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s1, index)), brow, dot01 ); \\ " << std::endl;
  stream << "        dot02 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s2, index)), brow, dot02 ); \\ " << std::endl;
  stream << "        dot03 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s3, index)), brow, dot03 ); \\ " << std::endl;
  stream << "        dot04 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s4, index)), brow, dot04 ); \\ " << std::endl;
  stream << "        dot05 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s5, index)), brow, dot05 ); \\ " << std::endl;
  stream << "        dot06 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s6, index)), brow, dot06 ); \\ " << std::endl;
  stream << "        dot07 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s7, index)), brow, dot07 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 0 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 0 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 1 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 1 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 2 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 2 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 3 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 3 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 4 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 4 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 5 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 5 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 6 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 6 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 7 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 7 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << " " << std::endl;
  stream << "        src0_read += TILE_K * ldA; " << std::endl;
  stream << "        start_index += TILE_K; " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(start_index < end_index) { " << std::endl;
  stream << "        " << sdtype8 << " arow0 = ((start_index + local_x * 2) < K) ? ((" << sdtype << ")alpha * vload8(0, src0_read)) : (" << sdtype << ")0.0f; " << std::endl;
  stream << "        " << sdtype8 << " arow1 = ((start_index + local_x * 2 + 1) < K) ? ((" << sdtype << ")alpha * vload8(0, src0_read + ldA)) : (" << sdtype << ")0.0f; " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _arow, index ) \\ " << std::endl;
  stream << "        brow = (start_index < K) ? vload4(0, src1_read0) : (" << sdtype << ")0.0f;  src1_read0 += ldB; start_index++; \\ " << std::endl;
  stream << "        dot00 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s0, index)), brow, dot00 ); \\ " << std::endl;
  stream << "        dot01 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s1, index)), brow, dot01 ); \\ " << std::endl;
  stream << "        dot02 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s2, index)), brow, dot02 ); \\ " << std::endl;
  stream << "        dot03 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s3, index)), brow, dot03 ); \\ " << std::endl;
  stream << "        dot04 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s4, index)), brow, dot04 ); \\ " << std::endl;
  stream << "        dot05 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s5, index)), brow, dot05 ); \\ " << std::endl;
  stream << "        dot06 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s6, index)), brow, dot06 ); \\ " << std::endl;
  stream << "        dot07 = mad( (" << sdtype4 << ")(intel_sub_group_shuffle(_arow.s7, index)), brow, dot07 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 0 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 0 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 1 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 1 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 2 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 2 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 3 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 3 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 4 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 4 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 5 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 5 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 6 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 6 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow0, 7 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( arow1, 7 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(global_x * 4 < N && global_y * 8 < M) { " << std::endl;
  stream << "        if(mad24(global_x, 4, 3) < N) { " << std::endl;
  stream << "            vstore4(dot00, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 2) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); " << std::endl;
  stream << "            dst_write0[2] = dot00.z; " << std::endl;
  stream << "            dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { " << std::endl;
  stream << "                vstore2(dot01.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot01.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { " << std::endl;
  stream << "                vstore2(dot02.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot02.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { " << std::endl;
  stream << "                vstore2(dot03.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot03.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { " << std::endl;
  stream << "                vstore2(dot04.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot04.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { " << std::endl;
  stream << "                vstore2(dot05.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot05.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { " << std::endl;
  stream << "                vstore2(dot06.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot06.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { " << std::endl;
  stream << "                vstore2(dot07.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot07.z; " << std::endl;
  stream << "            } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 1) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); } " << std::endl;
  stream << "        } else { " << std::endl;
  stream << "            dst_write0[0] = dot00.x; dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; } " << std::endl;
  stream << "        } " << std::endl;
  stream << "    } " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#undef VEC_SIZE " << std::endl;
  stream << "#undef LWG_HEIGHT " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;
  stream << " " << std::endl;
  stream << "#define VEC_SIZE        4 " << std::endl;
  stream << "#define LWG_HEIGHT      4 " << std::endl;
  stream << "#define TILE_M          8 " << std::endl;
  stream << "#define TILE_K          16 " << std::endl;
  stream << "#define TILE_N          32 " << std::endl;
  stream << " " << std::endl;
  stream << "__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1))) " << std::endl;
  stream << "__kernel void intelblas_gemm_buffer_TT_" << suffix << "( " << std::endl;
  stream << "    const __global " << sdtype << " *src0, int off0, " << std::endl;
  stream << "    const __global " << sdtype << " *src1, int off1, " << std::endl;
  stream << "    __global " << sdtype << " *dst, int offd, " << std::endl;
  stream << "    int M, " << std::endl;
  stream << "    int N, " << std::endl;
  stream << "    int K, " << std::endl;
  stream << "    float alpha, " << std::endl;
  stream << "    float beta, " << std::endl;
  stream << "    int ldA, " << std::endl;
  stream << "    int ldB, " << std::endl;
  stream << "    int ldC, " << std::endl;
  stream << "    int start_index, " << std::endl;
  stream << "    int stride) " << std::endl;
  stream << " " << std::endl;
  stream << "{ " << std::endl;
  stream << "    const int group_x = get_group_id(0); " << std::endl;
  stream << "    const int group_y = get_group_id(1); " << std::endl;
  stream << "    const int local_x = get_local_id(0); " << std::endl;
  stream << "    const int local_y = get_local_id(1); " << std::endl;
  stream << "    const int global_x = get_global_id(0); " << std::endl;
  stream << "    const int global_y = get_global_id(1); " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype8 << " dot0 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot1 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot2 = 0.f; " << std::endl;
  stream << "    " << sdtype8 << " dot3 = 0.f; " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype16 << " brow0; " << std::endl;
  stream << "    " << sdtype16 << " brow1; " << std::endl;
  stream << "    " << sdtype16 << " brow2; " << std::endl;
  stream << "    " << sdtype16 << " brow3; " << std::endl;
  stream << " " << std::endl;
  stream << "    __global " << sdtype << " *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * ldA + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0; " << std::endl;
  stream << " " << std::endl;
  stream << "    const __global " << sdtype << " *src1_read0 = src1 + (local_x * VEC_SIZE + ( group_x * TILE_N )) * ldB + start_index + off1; " << std::endl;
  stream << " " << std::endl;
  stream << "    " << sdtype4 << " dot00 = (start_index != 0) ? vload4(0, dst_write0) : (" << sdtype << ")beta * vload4(0, dst_write0); " << std::endl;
  stream << "    " << sdtype4 << " dot01 = (start_index != 0) ? vload4(0, dst_write0 + ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 2 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 3 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 4 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 5 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 6 * ldC); " << std::endl;
  stream << "    " << sdtype4 << " dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : (" << sdtype << ")beta * vload4(0, dst_write0 + 7 * ldC); " << std::endl;
  stream << " " << std::endl;
  stream << "    int end_index = min(start_index + stride, K); " << std::endl;
  stream << "    while( start_index + TILE_K <= end_index ) { " << std::endl;
  stream << "        brow0 = vload16(0, src1_read0); " << std::endl;
  stream << "        brow1 = vload16(0, src1_read0 + ldB); " << std::endl;
  stream << "        brow2 = vload16(0, src1_read0 + 2 * ldB); " << std::endl;
  stream << "        brow3 = vload16(0, src1_read0 + 3 * ldB); " << std::endl;
  stream << " " << std::endl;
  stream << "        " << sdtype8 << " arow0 = (" << sdtype << ")alpha * vload8(0, src0_read); " << std::endl;
  stream << "        " << sdtype8 << " arow1 = (" << sdtype << ")alpha * vload8(0, src0_read + ldA); " << std::endl;
  stream << " " << std::endl;
  stream << "#define DOT_PRODUCT( _dot, _arow, index, _brow) \\ " << std::endl;
  stream << "        _dot.s0 = mad( intel_sub_group_shuffle( _arow.s0, index ), _brow, _dot.s0 ); \\ " << std::endl;
  stream << "        _dot.s1 = mad( intel_sub_group_shuffle( _arow.s1, index ), _brow, _dot.s1 ); \\ " << std::endl;
  stream << "        _dot.s2 = mad( intel_sub_group_shuffle( _arow.s2, index ), _brow, _dot.s2 ); \\ " << std::endl;
  stream << "        _dot.s3 = mad( intel_sub_group_shuffle( _arow.s3, index ), _brow, _dot.s3 ); \\ " << std::endl;
  stream << "        _dot.s4 = mad( intel_sub_group_shuffle( _arow.s4, index ), _brow, _dot.s4 ); \\ " << std::endl;
  stream << "        _dot.s5 = mad( intel_sub_group_shuffle( _arow.s5, index ), _brow, _dot.s5 ); \\ " << std::endl;
  stream << "        _dot.s6 = mad( intel_sub_group_shuffle( _arow.s6, index ), _brow, _dot.s6 ); \\ " << std::endl;
  stream << "        _dot.s7 = mad( intel_sub_group_shuffle( _arow.s7, index ), _brow, _dot.s7 ); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _brow, _dot) \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 0, _brow.s0); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 0, _brow.s1); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 1, _brow.s2); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 1, _brow.s3); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 2, _brow.s4); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 2, _brow.s5); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 3, _brow.s6); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 3, _brow.s7); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 4, _brow.s8); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 4, _brow.s9); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 5, _brow.sa); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 5, _brow.sb); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 6, _brow.sc); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 6, _brow.sd); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 7, _brow.se); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 7, _brow.sf); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow0, dot0 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow1, dot1 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow2, dot2 ); " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow3, dot3 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "#undef DOT_PRODUCT " << std::endl;
  stream << " " << std::endl;
  stream << "        src1_read0 += TILE_K; " << std::endl;
  stream << "        src0_read += TILE_K * ldA; " << std::endl;
  stream << "        start_index += TILE_K; " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    if(start_index < end_index) { " << std::endl;
  stream << "        brow0 = vload16(0, src1_read0);  src1_read0 += ldB; " << std::endl;
  stream << "        brow1 = vload16(0, src1_read0);  src1_read0 += ldB; " << std::endl;
  stream << "        brow2 = vload16(0, src1_read0);  src1_read0 += ldB; " << std::endl;
  stream << "        brow3 = vload16(0, src1_read0); " << std::endl;
  stream << " " << std::endl;
  stream << "        " << sdtype8 << " arow0 = (" << sdtype << ")alpha * vload8(0, src0_read); " << std::endl;
  stream << "        " << sdtype8 << " arow1 = (" << sdtype << ")alpha * vload8(0, src0_read + ldA); " << std::endl;
  stream << " " << std::endl;
  stream << "#define DOT_PRODUCT( _dot, _arow, index, _brow) \\ " << std::endl;
  stream << "        _dot.s0 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s0, index ), _brow, _dot.s0 ) : _dot.s0; \\ " << std::endl;
  stream << "        _dot.s1 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s1, index ), _brow, _dot.s1 ) : _dot.s1; \\ " << std::endl;
  stream << "        _dot.s2 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s2, index ), _brow, _dot.s2 ) : _dot.s2; \\ " << std::endl;
  stream << "        _dot.s3 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s3, index ), _brow, _dot.s3 ) : _dot.s3; \\ " << std::endl;
  stream << "        _dot.s4 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s4, index ), _brow, _dot.s4 ) : _dot.s4; \\ " << std::endl;
  stream << "        _dot.s5 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s5, index ), _brow, _dot.s5 ) : _dot.s5; \\ " << std::endl;
  stream << "        _dot.s6 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s6, index ), _brow, _dot.s6 ) : _dot.s6; \\ " << std::endl;
  stream << "        _dot.s7 = (w++ < K) ? mad( intel_sub_group_shuffle( _arow.s7, index ), _brow, _dot.s7 ) : _dot.s7; \\ " << std::endl;
  stream << " " << std::endl;
  stream << "#define MM_DOT_PRODUCT( _brow, _dot) \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 0, _brow.s0); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 0, _brow.s1); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 1, _brow.s2); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 1, _brow.s3); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 2, _brow.s4); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 2, _brow.s5); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 3, _brow.s6); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 3, _brow.s7); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 4, _brow.s8); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 4, _brow.s9); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 5, _brow.sa); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 5, _brow.sb); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 6, _brow.sc); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 6, _brow.sd); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow0, 7, _brow.se); \\ " << std::endl;
  stream << "        DOT_PRODUCT(_dot, arow1, 7, _brow.sf); \\ " << std::endl;
  stream << " " << std::endl;
  stream << "        int w = start_index; " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow0, dot0 ); " << std::endl;
  stream << "        w = start_index; " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow1, dot1 ); " << std::endl;
  stream << "        w = start_index; " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow2, dot2 ); " << std::endl;
  stream << "        w = start_index; " << std::endl;
  stream << "        MM_DOT_PRODUCT( brow3, dot3 ); " << std::endl;
  stream << "#undef MM_DOT_PRODUCT " << std::endl;
  stream << "#undef DOT_PRODUCT " << std::endl;
  stream << "    } " << std::endl;
  stream << " " << std::endl;
  stream << "    dot00 += (" << sdtype4 << ")(dot0.s0, dot1.s0, dot2.s0, dot3.s0); " << std::endl;
  stream << "    dot01 += (" << sdtype4 << ")(dot0.s1, dot1.s1, dot2.s1, dot3.s1); " << std::endl;
  stream << "    dot02 += (" << sdtype4 << ")(dot0.s2, dot1.s2, dot2.s2, dot3.s2); " << std::endl;
  stream << "    dot03 += (" << sdtype4 << ")(dot0.s3, dot1.s3, dot2.s3, dot3.s3); " << std::endl;
  stream << "    dot04 += (" << sdtype4 << ")(dot0.s4, dot1.s4, dot2.s4, dot3.s4); " << std::endl;
  stream << "    dot05 += (" << sdtype4 << ")(dot0.s5, dot1.s5, dot2.s5, dot3.s5); " << std::endl;
  stream << "    dot06 += (" << sdtype4 << ")(dot0.s6, dot1.s6, dot2.s6, dot3.s6); " << std::endl;
  stream << "    dot07 += (" << sdtype4 << ")(dot0.s7, dot1.s7, dot2.s7, dot3.s7); " << std::endl;
  stream << " " << std::endl;
  stream << "    if(global_x * 4 < N && global_y * 8 < M) { " << std::endl;
  stream << "        if(mad24(global_x, 4, 3) < N) { " << std::endl;
  stream << "            vstore4(dot00, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 2) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); " << std::endl;
  stream << "            dst_write0[2] = dot00.z; " << std::endl;
  stream << "            dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { " << std::endl;
  stream << "                vstore2(dot01.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot01.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { " << std::endl;
  stream << "                vstore2(dot02.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot02.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { " << std::endl;
  stream << "                vstore2(dot03.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot03.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { " << std::endl;
  stream << "                vstore2(dot04.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot04.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { " << std::endl;
  stream << "                vstore2(dot05.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot05.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { " << std::endl;
  stream << "                vstore2(dot06.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot06.z; " << std::endl;
  stream << "                dst_write0 += ldC; " << std::endl;
  stream << "            } else " << std::endl;
  stream << "                return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { " << std::endl;
  stream << "                vstore2(dot07.xy, 0, dst_write0); " << std::endl;
  stream << "                dst_write0[2] = dot07.z; " << std::endl;
  stream << "            } " << std::endl;
  stream << "        } else if(mad24(global_x, 4, 1) < N) { " << std::endl;
  stream << "            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); } " << std::endl;
  stream << "        } else { " << std::endl;
  stream << "            dst_write0[0] = dot00.x; dst_write0 += ldC; " << std::endl;
  stream << "            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; } " << std::endl;
  stream << "            else return; " << std::endl;
  stream << "            if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; } " << std::endl;
  stream << "        } " << std::endl;
  stream << "    } " << std::endl;
  stream << "} " << std::endl;
  stream << " " << std::endl;
  stream << "#undef VEC_SIZE " << std::endl;
  stream << "#undef LWG_HEIGHT " << std::endl;
  stream << "#undef TILE_M " << std::endl;
  stream << "#undef TILE_K " << std::endl;
  stream << "#undef TILE_N " << std::endl;

  return stream.str();
}

void intelblas_gemm::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  namespace drv = driver;
  (void) queue;
  (void) suffix;
  //Get GEMM info
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(control.x(), args, A_trans_);
  int_t M = MNK[0], N = MNK[1], K = MNK[2];

  int offA = args.A->array.start, offB = args.B->array.start, offC = args.C->array.start;
  int ldA = args.A->ld[1];
  int ldB = args.B->ld[1];
  int ldC = args.C->ld[1];

  //Default order in isaac is column major.
  //This kernel is implemented in row major.
  //Need to swap matrix A and B each time.
  std::swap(args.A, args.B);
  std::swap(offA, offB);
  std::swap(ldA, ldB);
  std::swap(M, N);

  std::string name[4] = {"intelblas_gemm_buffer_NN_", "intelblas_gemm_buffer_NT_", "intelblas_gemm_buffer_TN_", "intelblas_gemm_buffer_TT_"};
  if(M % 32 == 0 && N % 32 == 0 && K % 16 == 0) {
    name[0] += "sp_";
  }

  name[0] += suffix;
  name[1] += suffix;
  name[2] += suffix;
  name[3] += suffix;

  std::vector<driver::Kernel> kernels;
  if(args.type == GEMM_NN)
    kernels.push_back(driver::Kernel(program, name[0].c_str()));
  else if(args.type == GEMM_NT)
    kernels.push_back(driver::Kernel(program, name[2].c_str()));
  else if(args.type == GEMM_TN)
    kernels.push_back(driver::Kernel(program, name[1].c_str()));
  else
    kernels.push_back(driver::Kernel(program, name[3].c_str()));
    
  driver::Kernel & kernel = kernels[0];
  unsigned int n_arg = 0;

  kernel.setArg(n_arg++, args.A->array.handle.cl);
  kernel.setSizeArg(n_arg++, offA);
  kernel.setArg(n_arg++, args.B->array.handle.cl);
  kernel.setSizeArg(n_arg++, offB);
  kernel.setArg(n_arg++, args.C->array.handle.cl);
  kernel.setSizeArg(n_arg++, offC);
  kernel.setSizeArg(n_arg++, M);
  kernel.setSizeArg(n_arg++, N);
  kernel.setSizeArg(n_arg++, K);
  kernel.setArg(n_arg++, args.alpha);
  kernel.setArg(n_arg++, args.beta);
  kernel.setSizeArg(n_arg++, ldA);
  kernel.setSizeArg(n_arg++, ldB);
  kernel.setSizeArg(n_arg++, ldC);

  int lx = 8;
  int ly = (args.type == GEMM_TN) ? 16 : 4;
  int dx = (args.type == GEMM_TN) ? 1 : 4;
  int dy = 8;
  size_t gx = (size_t)(N + dx - 1) / dx;
  size_t gy = (size_t)(M + dy - 1) / dy;
  driver::NDRange local(lx, ly);
  driver::NDRange global((gx + lx - 1) / lx * lx, (gy + ly - 1) / ly * ly);

  if(args.type == GEMM_TN) {
    control.execution_options().enqueue(program.context(), kernel, global, local);
  } else {
    int stride = (M * N < 1024 * 1024) ? 10000000 : 256;
    for(int start_index = 0; start_index < K; start_index += stride) {
      kernel.setSizeArg(n_arg, start_index);
      kernel.setSizeArg(n_arg + 1, stride);
      control.execution_options().enqueue(program.context(), kernel, global, local);
    }
  }
}

/* -------------------------------------------- */
unsigned int gemm::lmem_usage(expression_tree const & expression) const
{
  unsigned int N = 0;
  size_t llda = (A_trans_=='N')?mL_:kL_+vwidth_;
  size_t lnda = (A_trans_=='N')?kL_:mL_;
  size_t lldb = (B_trans_=='T')?nL_:kL_+vwidth_;
  size_t lndb = (B_trans_=='T')?kL_:nL_;
  N += llda*lnda;
  N += lldb*lndb;
  return N*size_of(expression.dtype());
}

expression_type gemm::type() const
{
  if(A_trans_=='N' && B_trans_=='N')
    return GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N')
    return GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T')
    return GEMM_NT;
  else
    return GEMM_TT;
}


unsigned int gemm::registers_usage(expression_tree const & expression) const
{
  unsigned int N = mS_ * nS_ + mS_ * kS_ + kS_ * nS_;
  return N*size_of(expression.dtype());
}

unsigned int gemm::temporary_workspace(expression_tree const & expressions) const
{
  std::vector<int_t> MNK = input_sizes(expressions);
  int_t M = MNK[0]; int_t N = MNK[1];
  if(depth_ > 1)
    return M*N*depth_;
  return 0;
}

int gemm::is_invalid_impl(driver::Device const &, expression_tree const &) const
{
  if ((mS_ % vwidth_) > 0 || (nS_ % vwidth_) > 0)
    return TEMPLATE_MS_NS_MUST_BE_SIMD_WIDTH_MULTIPLE;

  if(mL_ > 256 || nL_ > 256)
    return TEMPLATE_BLOCK_SIZE_TOO_LARGE;

  if ( kS_ % kL_ == 0)
    return TEMPLATE_KS_MUST_BE_SMALLER_THAN_KL;

  if ((lf0_*lf1_) !=(ls0_*ls1_))
    return TEMPLATE_LOCAL_FETCH_PRODUCT_MUST_MATCH_LOCAL_SIZE_PRODUCT;

  {
    unsigned int bound1 = (A_trans_=='N')?kL_:mL_;
    unsigned int bound0 = (A_trans_=='N')?mL_:kL_;

    if (lf1_>0 && (bound1 % lf1_)> 0)
      return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    if (lf0_>0 && (bound0 % (lf0_*vwidth_)) > 0)
      return A_trans_=='N'?TEMPLATE_LOCAL_FETCH_0_MUST_BE_NL_MULTIPLE:TEMPLATE_LOCAL_FETCH_0_MUST_BE_KL_MULTIPLE;

  }

  {
    unsigned int bound1 = (B_trans_=='T')?kL_:nL_;
    unsigned int bound0 = (B_trans_=='T')?nL_:kL_;

    if (lf1_>0 && (bound1 % lf1_)> 0)
      return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

    if (lf0_>0 && (bound0 % (lf0_*vwidth_)) > 0)
      return B_trans_=='T'?TEMPLATE_LOCAL_FETCH_1_MUST_BE_KL_MULTIPLE:TEMPLATE_LOCAL_FETCH_1_MUST_BE_ML_MULTIPLE;

  }

  return TEMPLATE_VALID;
}

std::string gemm::generate_impl(std::string const & suffix, expression_tree const & tree, driver::Device const & device, symbolic::symbols_table const &) const
{
  using std::string;
  using tools::to_string;

  driver::backend_type backend = device.backend();
  bool has_depth = depth_ > 1;
#define VLOAD(offset, ptr) vload(vwidth_, sdtype, offset, ptr, "1", backend, true)
#define VLOAD_MISALIGNED(offset, ptr) vload(vwidth_, sdtype, offset, ptr, "1", backend, false)
#define VSTORE_LDSA(value, offset, ptr) vstore(vwidth_, sdtype, value, offset, ptr, "1", backend, llda%vwidth_==0)
#define VSTORE_LDSB(value, offset, ptr) vstore(vwidth_, sdtype, value, offset, ptr, "1", backend, lldb%vwidth_==0)

  symbolic::preset::gemm::args args;
  infos(tree, args, A_trans_);
  std::string ASTRIDE1 = (args.A->ld[0] > 1)?"*Astride1":"";
  std::string BSTRIDE1 = (args.B->ld[0] > 1)?"*Bstride1":"";
  std::string CSTRIDE1 = (args.C->ld[0] > 1)?"*Cstride1":"";

  //////////////////
  /// INIT
  /// //////////////
  kernel_generation_stream stream(backend);
  numeric_type dtype = tree.dtype();
  std::string sdtype = to_string(dtype);
  std::string vdtype = append_width(sdtype, vwidth_);
  std::string abdtype = (sdtype == "half")? "float" : sdtype;

  //////////////////
  /// DECLARATIONS
  /// //////////////
  std::string gemm_name = "gemm";
  std::string reduce_name = "reduce";

  gemm_name += suffix;
  reduce_name += suffix;

  switch(backend)
  {
  case driver::OPENCL:
    if(tree.dtype()==HALF_TYPE)
      stream << "#pragma OPENCL EXTENSION cl_khr_fp16: enable" << std::endl;
    stream << " __attribute__((reqd_work_group_size(" << ls0_ << "," << ls1_ << ",1)))" << std::endl;
    break;
  default:
    break;
  }
  stream << "$KERNEL void gemm" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T K, "
         << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T offc, $SIZE_T Cstride1, "
         << abdtype << " alpha,"
         << "$GLOBAL " << sdtype << "* A, $SIZE_T lda, $SIZE_T offa, $SIZE_T Astride1,"
         << "$GLOBAL " << sdtype << "* B, $SIZE_T ldb, $SIZE_T offb, $SIZE_T Bstride1,"
         << abdtype << " beta)"
         << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();

  ///Declare
  stream << "//blocks" << std::endl;
  stream << sdtype << " rC[" << mS_ << "][" << nS_ << "] = {{0}};" << std::endl;
  stream << vdtype << " rA[" << kS_ << "][" << mS_/vwidth_ << "];" << std::endl;
  stream << vdtype << " rB[" << kS_ << "][" << nS_/vwidth_ << "];" << std::endl;
  stream << std::endl;

  stream << "//pointers" << std::endl;
  size_t llda = (A_trans_=='N')?mL_:kL_+vwidth_;
  size_t lnda = (A_trans_=='N')?kL_:mL_;
  size_t lldb = (B_trans_=='T')?nL_:kL_+vwidth_;
  size_t lndb = (B_trans_=='T')?kL_:nL_;
  stream << "$LOCAL " << sdtype << " lA[" << llda*lnda << "];" << std::endl;
  stream << "$LOCAL " << sdtype << " lB[" << lldb*lndb << "];" << std::endl;
  unsigned int npA = mL_/(A_trans_=='N'?lf0_*vwidth_:lf1_);
  unsigned int npB = nL_/(B_trans_=='T'?lf0_*vwidth_:lf1_);
  stream << "$GLOBAL " << sdtype << "* Ai[" << npA << "];" << std::endl;
  stream << "$GLOBAL " << sdtype << "* Bi[" << npB << "];" << std::endl;
  stream << std::endl;

  stream << "//identifiers" << std::endl;
  stream << "int2 idT;" << std::endl;
  stream << "int idt;" << std::endl;
  if(has_depth)
    stream << "int gidz, div, offz;" << std::endl;
  stream << "uint4 ids;" << std::endl;
  stream << "ids.x = $GROUP_IDX_0;" << std::endl;
  stream << "ids.y = $GROUP_IDX_1;" << std::endl;
  stream << "ids.z = $LOCAL_IDX_0;" << std::endl;
  stream << "ids.w = $LOCAL_IDX_1;" << std::endl;
  stream << std::endl;

  stream << "//offsets" << std::endl;
  stream << "A += offa;" << std::endl;
  stream << "B += offb;" << std::endl;
  stream << "C += offc;" << std::endl;

  if(has_depth)
  {
    stream << "gidz = $GROUP_IDX_2;" << std::endl;
    stream << "div = (K+" << depth_-1 << ")/" << depth_ << ";" << std::endl;
    stream << "offz = div*gidz;" << std::endl;
    stream << "K = max(0, min(K - div*gidz, ($SIZE_T)div));" << std::endl;
  }

  stream << "idt = " << ls0_ << "*ids.w + ids.z;" << std::endl;
  stream << "idT.y = idt/" << lf0_ << ";" << std::endl;
  stream << "idT.x = idt - " << lf0_ << "*idT.y;" << std::endl;
  stream << std::endl;

  stream << "//Adjust pointers and bounds per work-item" << std::endl;
  stream << "ids.x *= " << mL_ << ";" << std::endl;
  stream << "ids.y *= " << nL_ << ";" << std::endl;
  stream << "idT.x *= " << vwidth_ << ";" << std::endl;

  stream << "M -= ids.x;" << std::endl;
  if(A_trans_=='N')
    stream << "M -= idT.x;" << std::endl;
  else
    stream << "M -= idT.y;" << std::endl;

  stream << "N -= ids.y;" << std::endl;
  if(B_trans_=='T')
    stream << "N -= idT.x;" << std::endl;
  else
    stream << "N -= idT.y;" << std::endl;

  if (A_trans_=='N')
  {
    stream << "A += ids.x" << ASTRIDE1 << ";" << std::endl;
    stream << "A += idT.y*lda;" << std::endl;
    if(has_depth)
      stream << "A += offz*lda;" << std::endl;

  }
  else
  {
    stream << "A += ids.x*lda;" << std::endl;
    stream << "A += idT.x" << ASTRIDE1 << ";" << std::endl;
    if(has_depth)
      stream << "A += offz;" << std::endl;
  }

  if(B_trans_=='T')
  {
    stream << "B += ids.y" << BSTRIDE1 << ";" << std::endl;
    stream << "B += idT.y*ldb;" << std::endl;
    if(has_depth)
      stream << "B += offz*ldb;" << std::endl;
  }
  else
  {
    stream << "B += ids.y*ldb;" << std::endl;
    stream << "B += idT.x" << BSTRIDE1 << ";" << std::endl;
    if(has_depth)
      stream << "B += offz;" << std::endl;
  }

  stream << "#pragma unroll" << std::endl;
  stream << "for(int i = 0 ; i < " << npA << " ; ++i){" << std::endl;
  stream.inc_tab();
  stream << "Ai[i] = A;" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream << std::endl;

  stream << "#pragma unroll" << std::endl;
  stream << "for(int i = 0 ; i < " << npB << " ; ++i){" << std::endl;
  stream.inc_tab();
  stream << "Bi[i] = B;" << std::endl;
  stream.dec_tab();
  stream << "}" << std::endl;
  stream << std::endl;

  for(unsigned int i = 0 ; i < npA ; i++ )
    if (A_trans_=='N')
      stream << "Ai[" << i << "] += " << Select(backend, to_string(i*lf0_*vwidth_) + " < M", "(int)((idT.x + " + to_string(i*lf0_*vwidth_) + ")" + ASTRIDE1 + ")", "0") << ";" << std::endl;
    else
      stream << "Ai[" << i << "] += " << Select(backend, to_string(i*lf1_) + " < M", "(int)((idT.y + " + to_string(i*lf1_) + ")*lda)", "0") << ";" << std::endl;

  for(unsigned int i = 0 ; i < npB ; i++ )
    if (B_trans_=='T')
      stream << "Bi[" << i << "] += " << Select(backend, to_string(i*lf0_*vwidth_) + " < N", "(int)((idT.x + " + to_string(i*lf0_*vwidth_) + ")" + BSTRIDE1 + ")", "0") << ";" << std::endl;
    else
      stream << "Bi[" << i << "] += " << Select(backend, to_string(i*lf1_) + " < N", "(int)((idT.y + " + to_string(i*lf1_) + ")*ldb)", "0") << ";" << std::endl;

  stream << std::endl;
  stream << "//Outer loop" << std::endl;
  stream << "while(K >=" << kL_ << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();


  auto fetch_to_lds = [&](bool last_iteration)
  {
    stream << "$LOCAL_BARRIER;" << std::endl;
    stream << "$LOCAL_PTR " << sdtype << "* ldsA = lA + idT.y*" << llda << " + idT.x;" << std::endl;
    stream << "$LOCAL_PTR " << sdtype << "* ldsB = lB + idT.y*" << lldb << " + idT.x;" << std::endl;

    stream << "//Fetch A to local memory" << std::endl;
    if (A_trans_=='N')
    {
      for(unsigned int k = 0; k < kL_; k += lf1_)
        for(unsigned int m = 0; m < mL_; m += lf0_*vwidth_)
        {
          std::string mm = to_string(m/(vwidth_*lf0_));
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsA[" << k*llda + m + s << "] = (condy" << k << " && " << s << "< M)? Ai[" << mm << "][" << k << "*lda + " << s << "] : 0;" << std::endl;
          else
            stream << VSTORE_LDSA(VLOAD_MISALIGNED("0" ,"&Ai[" + mm +"][" + kk + "*lda]"), "0", "ldsA + " + to_string(k*llda+m)) << ";" << std::endl;
        }
    }
    else
    {
      for(unsigned int k = 0; k < kL_; k += lf0_*vwidth_)
        for(unsigned int m = 0; m < mL_; m += lf1_)
        {
          std::string mm = to_string(m/lf1_);
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsA[" << m*llda + k + s << "] = condx" << k + s << "? Ai[" << mm << "][" << k + s << ASTRIDE1 << "] : 0;" << std::endl;

          else
            stream << VSTORE_LDSA(VLOAD_MISALIGNED("0", "&Ai[" + mm + "][" + kk + ASTRIDE1 + "]"), "0", "ldsA + " + to_string(m*llda+k)) << ";" << std::endl;
        }
    }

    stream << "//Fetch B to local memory" << std::endl;
    if (B_trans_=='T')
    {
      for(unsigned int k = 0; k < kL_; k += lf1_)
        for(unsigned int n = 0; n < nL_; n += lf0_*vwidth_)
        {
          std::string nn = to_string(n/(vwidth_*lf0_));
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsB[" << k*lldb + n + s << "] = (condy" << k << " && " << s << "< N)? Bi[" <<  nn << "][" << kk << "*ldb +" << s << "] : 0;" << std::endl;
          else
            stream << VSTORE_LDSB(VLOAD_MISALIGNED("0" ,"&Bi[" + nn +"][" + kk + "*ldb]"), "0", "ldsB + " + to_string(k*lldb+n)) << ";" << std::endl;
        }
    }
    else
    {
      for(unsigned int k = 0; k < kL_; k += lf0_*vwidth_)
        for(unsigned int n = 0; n < nL_; n += lf1_)
        {
          std::string nn = to_string(n/lf1_);
          std::string kk = to_string(k);
          if(last_iteration)
            for(unsigned int s = 0 ; s < vwidth_ ; ++s)
              stream << "ldsB[" << n*lldb + k + s << "] = condx" << k + s << "? Bi[" << nn << "][" << k + s << BSTRIDE1 << "] : 0;" << std::endl;

          else
            stream << VSTORE_LDSB(VLOAD_MISALIGNED("0", "&Bi[" + nn + "][" + kk + BSTRIDE1 + "]"), "0", "ldsB + " + to_string(n*lldb+k)) << ";" << std::endl;
        }
    }

    if(A_trans_=='N')
      stream << "ldsA = lA + ids.z*" << vwidth_ << ";" << std::endl;
    else
      stream << "ldsA = lA + ids.z*" << llda*vwidth_ << ";" << std::endl;

    if(B_trans_=='T')
      stream << "ldsB = lB + ids.w*" << vwidth_ << ";" << std::endl;
    else
      stream << "ldsB = lB + ids.w*" << lldb*vwidth_ << ";" << std::endl;

    stream << "$LOCAL_BARRIER;" << std::endl;
    std::string bound = last_iteration?"K":tools::to_string(kL_);
    size_t ks = last_iteration?1:kS_;
    stream << "//Inner loop" << std::endl;
    stream << "for(unsigned int k = 0; k < " << bound << "; k+=" << ks << "){" << std::endl;
    stream.inc_tab();

    stream << "//Fetch A to registers" << std::endl;
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << ks << "; kk++)" << std::endl;
    stream << "#pragma unroll " << mS_/vwidth_ << std::endl;
    stream << "for(unsigned int mm = 0; mm < " << mS_/vwidth_ << "; mm++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(A_trans_=='N')
      stream << "rA[kk][mm] = "  << VLOAD("0", "ldsA + k*" + to_string(llda) + " + mm*" + to_string(ls0_*vwidth_) + "+ kk*" + to_string(llda)) << ";" << std::endl;
    else
    {
      if(vwidth_==1)
        stream << "rA[kk][mm] = ldsA[k + mm*" << ls0_*llda <<  "+ kk"  << "];" << std::endl;
      else
        for(unsigned int s = 0 ; s < vwidth_ ; ++s)
          stream << access_vector_type("rA[kk][mm]", s) << " = ldsA[k + (mm*" << vwidth_*ls0_ << " + " << s << ")*" << llda <<  "+ kk];" << std::endl;
    }

    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "//Fetch B to registers" << std::endl;
    stream << "#pragma unroll " << ks << std::endl;
    stream << "for(unsigned int kk = 0; kk < " << ks << "; kk++)" << std::endl;
    stream << "#pragma unroll " << nS_/vwidth_ << std::endl;
    stream << "for(unsigned int nn = 0; nn < " << nS_/vwidth_ << "; nn++)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    if(B_trans_=='T')
      stream << "rB[kk][nn] = " << VLOAD("0", "ldsB + k*" + to_string(lldb) + " + nn*" + to_string(ls1_*vwidth_)  + "+ kk*" + to_string(lldb)) << ";" << std::endl;
    else
    {
      if(vwidth_==1)
        stream << "rB[kk][nn] = ldsB[k"  << " + nn*" << ls1_*lldb <<  "+ kk"  << "];" << std::endl;
      else
        for(unsigned int s = 0 ; s < vwidth_ ; ++s)
          stream << access_vector_type("rB[kk][nn]", s) << " = ldsB[k"  << " + (nn*" << vwidth_*ls1_ << " + " << s << ")*" << lldb <<  "+ kk];" << std::endl;
    }
    stream.dec_tab();
    stream << "}" << std::endl;

    stream << "//FMA computations" << std::endl;
    stream << "#pragma unroll" << std::endl;
    stream << "for(unsigned int kk = 0 ; kk < " << ks << "; ++kk){" << std::endl;
    stream.inc_tab();
    for(unsigned int nn=0; nn < nS_; ++nn)
      for(unsigned int mm=0; mm < mS_; ++mm){
        string res_str, lhs_str, rhs_str;
        res_str = "rC[" + to_string(mm) + "][" + to_string(nn) + "]";
        if (vwidth_==1)
          lhs_str = "rA[kk][" + to_string(mm) + "]";
        else
          lhs_str = access_vector_type("rA[kk][" + to_string(mm/vwidth_) + "]", mm%vwidth_);
        if (vwidth_==1)
          rhs_str = "rB[kk]["+to_string(nn)+"]";
        else
          rhs_str = access_vector_type("rB[kk]["+to_string(nn/vwidth_)+"]", nn%vwidth_);
        stream << res_str << "= $MAD(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
      }
    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream << "K -= " << kL_ << ";" << std::endl;

    //Increment A pointers to global memory
    if (A_trans_=='N')
      for(unsigned int i = 0 ; i < npA ; ++i)
        stream << "Ai[" << i << "] += "  << kL_ << "*lda;" << std::endl;
    else
      for(unsigned int i = 0 ; i < npA ; ++i)
        stream << "Ai[" << i << "] += "  << kL_ << ASTRIDE1 << ";" << std::endl;

    //Increment B pointers to global memory
    if (B_trans_=='T')
      for(unsigned int i = 0 ; i < npB ; ++i)
        stream << "Bi[" << i << "] += " << kL_ << "*ldb;" << std::endl;
    else
      for(unsigned int i = 0 ; i < npB ; ++i)
        stream << "Bi[" << i << "] += " << kL_ << BSTRIDE1 << ";" << std::endl;
  };
  fetch_to_lds(false);
  stream.dec_tab();
  stream << "}" << std::endl;


  if(A_trans_=='N' || B_trans_=='T')
  {
    stream << "int Ky = K - idT.y;" << std::endl;
    for(unsigned int k = 0; k < kL_; k += lf1_)
      stream << "int condy" << k << " = " << k << " < Ky;" << std::endl;
  }

  if(A_trans_=='T' || B_trans_=='N')
  {
    stream << "int Kx = K - idT.x;" << std::endl;
    for(unsigned int k = 0 ; k < kL_ ; k += lf0_*vwidth_)
      for(unsigned int s = 0 ; s < vwidth_ ; ++s)
        stream << "int condx" << k + s << " = " << k + s << " < Kx;" << std::endl;
  }
  fetch_to_lds(true);

  stream << "//Write back C" << std::endl;
  stream << "M += ids.x;" << std::endl;
  if(A_trans_=='N')
    stream << "M += idT.x;" << std::endl;
  else
    stream << "M += idT.y;" << std::endl;

  if(B_trans_=='T')
    stream << "N += idT.x;" << std::endl;
  else
    stream << "N += idT.y;" << std::endl;
  stream << "N += ids.y;" << std::endl;

  stream << "C += ids.x" << CSTRIDE1 << ";" << std::endl;
  stream << "C += ids.z*" << vwidth_ << CSTRIDE1 << ";" << std::endl;
  stream << "C += ids.y*ldc;" << std::endl;
  stream << "C += ids.w*" << vwidth_ << "*ldc;" << std::endl;
  if(has_depth)
    stream << "C += gidz*ldc*N;" << std::endl;

  stream << "M -= ids.x;" << std::endl;
  stream << "M -= ids.z*" << vwidth_ << ";" << std::endl;

  stream << "N -= ids.y;" << std::endl;
  stream << "N -= ids.w*" << vwidth_ <<  ";" << std::endl;

  for(unsigned int n=0; n < nS_; ++n)
  {
    string Cj = to_string((n/vwidth_)*(ls1_*vwidth_) + n%vwidth_);
    stream << "if(" << Cj << " >= N) return;" << std::endl;
    for(unsigned int m=0; m < mS_; ++m)
      stream << "rC[" << m << "][" << n << "] *= alpha;" << std::endl;
    for(unsigned int m=0; m < mS_; ++m)
    {
      string Ci = to_string((m/vwidth_)*(ls0_*vwidth_) + m%vwidth_);
      stream << "if(" << Ci << "< M) ";
      if(has_depth)
        stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "];" << std::endl;
      else
        stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "] + ((beta != (" << sdtype << ")0)?(beta*" << "C[" << Ci << CSTRIDE1 << "]):0);" << std::endl;
    }
    if((n+1)%vwidth_==0){
      stream << "C += ldc*" << ls1_*vwidth_ - vwidth_ + 1 << ";" << std::endl;
    }
    else{
      stream << "C += ldc;" << std::endl;
    }

  }

  stream.dec_tab();
  stream << "}" << std::endl;

  if(has_depth)
  {
    stream << "$KERNEL void reduce" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T D, "
           << "$GLOBAL " << sdtype << "* Z, $SIZE_T Zld,"
           << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T Cstart, $SIZE_T Cstride,"
           << abdtype << " beta)"
           << std::endl;

    stream << "{" << std::endl;
    stream.inc_tab();

    stream << "C += Cstart;" << std::endl;
    stream << "for(unsigned int i = $GLOBAL_IDX_0 ;  i < M ;  i += $GLOBAL_SIZE_0)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    stream << "for(unsigned int j = $GLOBAL_IDX_1 ;  j < N ;  j += $GLOBAL_SIZE_1)" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    stream << sdtype << " acc = 0;" << std::endl;
    stream << "for(unsigned int k = 0 ;  k < D ;  k++)" << std::endl;
    stream.inc_tab();
    stream << "acc += Z[i + j*Zld + k*Zld*N];" << std::endl;
    stream.dec_tab();
    stream << "C[i*Cstride + j*ldc] = acc + ((beta != (" << sdtype << ")0)?(beta*C[i*Cstride + j*ldc]):0);" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;

    stream.dec_tab();
    stream << "}" << std::endl;
  }

  return stream.str();

#undef VLOAD
#undef VST0RE
}

void gemm::enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K,
                         expression_tree::node const & A, expression_tree::node const & B, expression_tree::node const & C,
                         value_scalar const & alpha, value_scalar const & beta,
                         driver::Program const & program, std::string const & suffix, runtime::execution_options_type const & options)
{
  using tools::align;

  if(M==0 || N==0 || K==0)
    return;

  driver::backend_type backend = queue.context().backend();

  std::string gemm_name = "gemm";
  std::string reduce_name = "reduce";

  gemm_name += suffix;
  reduce_name += suffix;

  driver::Kernel gemm(program, gemm_name.c_str());
  driver::NDRange local(ls0_, ls1_, 1);
  driver::NDRange global(align(align(M,mS_)/mS_, ls0_), align(align(N,nS_)/nS_, ls1_), depth_);

  unsigned int current_arg = 0;

  driver::Buffer& workspace = driver::backend::workspaces::get(options.queue(queue.context()));
  gemm.setSizeArg(current_arg++, M);
  gemm.setSizeArg(current_arg++, N);
  gemm.setSizeArg(current_arg++, K);
  if(depth_==1)
  {
    if(backend==driver::OPENCL)
      gemm.setArg(current_arg++, C.array.handle.cl);
    else
      gemm.setArg(current_arg++, C.array.handle.cu);
    gemm.setSizeArg(current_arg++, C.ld[1]);
    gemm.setSizeArg(current_arg++, C.array.start);
    gemm.setSizeArg(current_arg++, C.ld[0]);
  }
  else
  {
    gemm.setArg(current_arg++, workspace);
    gemm.setSizeArg(current_arg++, M);
    gemm.setSizeArg(current_arg++, 0);
    gemm.setSizeArg(current_arg++, 1);
  }


  gemm.setArg(current_arg++, alpha);
  if(backend==driver::OPENCL)
    gemm.setArg(current_arg++, A.array.handle.cl);
  else
    gemm.setArg(current_arg++, A.array.handle.cu);
  gemm.setSizeArg(current_arg++, A.ld[1]);
  gemm.setSizeArg(current_arg++, A.array.start);
  gemm.setSizeArg(current_arg++, A.ld[0]);

  if(backend==driver::OPENCL)
    gemm.setArg(current_arg++, B.array.handle.cl);
  else
    gemm.setArg(current_arg++, B.array.handle.cu);
  gemm.setSizeArg(current_arg++, B.ld[1]);
  gemm.setSizeArg(current_arg++, B.array.start);
  gemm.setSizeArg(current_arg++, B.ld[0]);

  gemm.setArg(current_arg++, beta);
  options.enqueue(program.context(), gemm, global, local);

  if(depth_ > 1)
  {
    unsigned int current_arg = 0;
    driver::Kernel reduce(program, reduce_name.c_str());
    driver::NDRange local(ls0_, ls1_);
    driver::NDRange global(align(M, ls0_), align(N, ls1_));
    reduce.setSizeArg(current_arg++, M);
    reduce.setSizeArg(current_arg++, N);
    reduce.setSizeArg(current_arg++, depth_);
    reduce.setArg(current_arg++, workspace);
    reduce.setSizeArg(current_arg++, M);
    if(backend==driver::OPENCL)
      reduce.setArg(current_arg++, C.array.handle.cl);
    else
      reduce.setArg(current_arg++, C.array.handle.cu);
    reduce.setSizeArg(current_arg++, C.ld[1]);
    reduce.setSizeArg(current_arg++, C.array.start);
    reduce.setSizeArg(current_arg++, C.ld[0]);
    reduce.setArg(current_arg++, beta);
    options.enqueue(program.context(), reduce, global, local);
  }

}

gemm::gemm(unsigned int vwidth
           ,int_t ls0, int_t kL, int_t ls1, int_t D
           ,int_t ms, int_t ks, int_t ns
           ,int_t lf0, int_t lf1, char A_trans, char B_trans) :
  parameterized_base(vwidth, ls0, ls1), mL_(ms*ls0), kL_(kL), nL_(ns*ls1), depth_(D), mS_(ms), kS_(ks)
                                     , nS_(ns), lf0_(lf0), lf1_(lf1), A_trans_(A_trans), B_trans_(B_trans)
{
  if(A_trans_=='N' && B_trans_=='N') type_ = GEMM_NN;
  else if(A_trans_=='T' && B_trans_=='N') type_ = GEMM_TN;
  else if(A_trans_=='N' && B_trans_=='T') type_ = GEMM_NT;
  else if(A_trans_=='T' && B_trans_=='T') type_ = GEMM_TT;
  else throw;
}

std::vector<int_t> gemm::input_sizes(expression_tree const & expressions) const
{
  symbolic::preset::gemm::args dummy;
  return infos((expression_tree&)expressions, dummy, A_trans_);
}

void gemm::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix, runtime::execution_handler const & control)
{
  expression_tree const & expressions = control.x();
  symbolic::preset::gemm::args args;
  std::vector<int_t> MNK = infos(expressions, args, A_trans_);
  int_t M = MNK[0];
  int_t N = MNK[1];
  int_t K = MNK[2];
  //Skip if empty
  if(M==0 || N == 0 || K ==0)
    return;
  //Enqueue
  runtime::execution_options_type const & options = control.execution_options();
  enqueue_block(queue,  M, N, K, *args.A, *args.B, *args.C, args.alpha, args.beta, program, suffix, options);
}

//
gemm_nn::gemm_nn(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'N', 'N')
{
}

//
gemm_tn::gemm_tn(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'T', 'N')
{ }

//
gemm_nt::gemm_nt(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'N', 'T')
{ }

//
gemm_tt::gemm_tt(unsigned int vwidth
                 , int_t ls0, int_t KL, int_t ls1, int_t D
                 , int_t ms, int_t ks, int_t ns
                 , int_t lf0, int_t lf1) :
  gemm(vwidth, ls0, KL, ls1, D, ms, ks, ns, lf0, lf1, 'T', 'T')
{ }

}
}

