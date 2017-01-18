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

#include <algorithm>
#include <tuple>
#include "isaac/jit/syntax/engine/macro.h"
#include "isaac/tools/cpp/string.hpp"

namespace isaac
{

namespace symbolic
{

macro::macro(std::string const & code): code_(code)
{
  size_t pos_po = code_.find('(');
  size_t pos_pe = code_.find(')');
  name_ = code.substr(0, pos_po);
  args_ = tools::split(code.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
  tokens_ = tools::tokenize(code_.substr(code_.find(":") + 1), "()[],*+/-=>< ");
}

macro::macro(const char *code) : macro(std::string(code))
{

}

int macro::expand(std::string & str) const
{
  size_t pos = 0;
  size_t num_touched = 0;
  while((pos=str.find(name_ + "(", pos==0?0:pos + 1))!=std::string::npos){
    size_t pos_po = str.find('(', pos);
    size_t pos_pe = str.find(')', pos_po);
    size_t next = str.find('(', pos_po + 1);
    while(next < pos_pe){
      pos_pe = str.find(')', pos_pe + 1);
      if(next < pos_pe)
        next = str.find('(', next + 1);
    }

    std::vector<std::string> args = tools::split(str.substr(pos_po + 1, pos_pe - pos_po - 1), ',');
    if(args_.size() != args.size()){
      pos = pos_pe;
      continue;
    }

    //Process
    std::vector<std::string> tokens = tokens_;
    for(size_t i = 0 ; i < args_.size() ; ++i)
      std::replace(tokens.begin(), tokens.end(), args_[i], args[i]);

    //Replace
    str.replace(pos, pos_pe + 1 - pos, tools::join(tokens.begin(), tokens.end(), ""));
    num_touched++;
  }
  return num_touched;
}

bool macro::operator<(macro const & o) const
{
  return std::make_tuple(name_, args_.size()) < std::make_tuple(o.name_, o.args_.size());
}

}
}
