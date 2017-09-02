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


#include "isaac/runtime/predict.h"
#include "database/sm_5_2/conv.hpp"
#include "database/sm_5_2/gemm.hpp"

#include "database/sm_6_0/conv.hpp"
#include "database/sm_6_0/gemm.hpp"

#include "database/sm_6_1/conv.hpp"
#include "database/sm_6_1/gemm.hpp"

#include "database/sm_7_0/gemm.hpp"

namespace isaac{
namespace runtime{

typedef driver::Device::Architecture Architecture;

const std::map<std::pair<driver::Device::Architecture, OperationType>, std::shared_ptr<Profile> > database =
{
  {{Architecture::SM_5_0, CONV}, std::make_shared<ConvProfile>((u_char*)sm_5_2::conv)},
  {{Architecture::SM_5_0, GEMM}, std::make_shared<GEMMProfile>((u_char*)sm_5_2::gemm)},

  {{Architecture::SM_5_2, CONV}, std::make_shared<ConvProfile>((u_char*)sm_5_2::conv)},
  {{Architecture::SM_5_2, GEMM}, std::make_shared<GEMMProfile>((u_char*)sm_5_2::gemm)},

  {{Architecture::SM_6_0, CONV}, std::make_shared<ConvProfile>((u_char*)sm_6_0::conv)},
  {{Architecture::SM_6_0, GEMM}, std::make_shared<GEMMProfile>((u_char*)sm_6_0::gemm)},

  {{Architecture::SM_6_1, CONV}, std::make_shared<ConvProfile>((u_char*)sm_6_1::conv)},
  {{Architecture::SM_6_1, GEMM}, std::make_shared<GEMMProfile>((u_char*)sm_6_1::gemm)},

  {{Architecture::SM_7_0, CONV}, std::make_shared<ConvProfile>((u_char*)sm_6_1::conv)},
  {{Architecture::SM_7_0, GEMM}, std::make_shared<GEMMProfile>((u_char*)sm_7_0::gemm)}

};

}
}
