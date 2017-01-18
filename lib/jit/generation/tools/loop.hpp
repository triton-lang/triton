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

#include "isaac/jit/generation/engine/stream.h"
#include "isaac/jit/generation/base.h"
#include <string>
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{
namespace templates
{

template<class Fun>
inline void element_wise_loop_1D(kernel_generation_stream & stream, unsigned int vwidth,
                                 std::string const & i, std::string const & bound, std::string const & domain_id, std::string const & domain_size, Fun const & generate_body)
{
  std::string svwidth = tools::to_string(vwidth);
  std::string init = domain_id + "*" + svwidth;
  std::string lbound = bound + "/" + svwidth + "*" + svwidth;
  std::string inc = domain_size + "*" + svwidth;
  stream << "for(unsigned int " << i << " = " << init << "; " << i << " < " << lbound << "; " << i << " += " << inc << ")" << std::endl;
  stream << "{" << std::endl;
  stream.inc_tab();
  generate_body(vwidth);
  stream.dec_tab();
  stream << "}" << std::endl;

  if (vwidth>1)
  {
    stream << "for(unsigned int " << i << " = " << lbound << " + " << domain_id << "; " << i << " < " << bound << "; " << i << " += " + domain_size + ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();
    generate_body(1);
    stream.dec_tab();
    stream << "}" << std::endl;
  }
}

}
}
