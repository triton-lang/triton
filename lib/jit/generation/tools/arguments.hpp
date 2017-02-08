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

#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "isaac/jit/syntax/engine/object.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/array.h"

namespace isaac
{
namespace templates
{

//Generate
inline std::vector<std::string> kernel_arguments(driver::Device const &, symbolic::symbols_table const & symbols, expression_tree const & expressions)
{
    std::vector<std::string> result;
    for(symbolic::object* obj: symbolic::extract<symbolic::object>(expressions, symbols))
    {
      if(symbolic::host_scalar* sym = dynamic_cast<symbolic::host_scalar*>(obj))
        result.push_back(sym->process("#scalartype #name_value"));
      if(symbolic::buffer* sym = dynamic_cast<symbolic::buffer*>(obj))
      {
        std::string pointer_name = sym->process("#scalartype* #pointer");
        if(sym->hasattr("inc0") && !sym->hasattr("inc1"))
          result.push_back("$GLOBAL " + pointer_name+"_bk");
        else
          result.push_back("$GLOBAL " + pointer_name);
        if(sym->hasattr("off")) result.push_back("$SIZE_T " + sym->process("#off"));
        if(sym->hasattr("inc0")) result.push_back("$SIZE_T " + sym->process("#inc0"));
        if(sym->hasattr("inc1")) result.push_back("$SIZE_T " + sym->process("#inc1"));
      }
      if(symbolic::reshape* sym = dynamic_cast<symbolic::reshape*>(obj))
      {
        if(sym->hasattr("new_inc1")) result.push_back("$SIZE_T " + sym->process("#new_inc1"));
        if(sym->hasattr("old_inc1")) result.push_back("$SIZE_T " + sym->process("#old_inc1"));
      }
    }
    return result;
}


inline std::vector<std::string> negative_inc_process(driver::Device const &, symbolic::symbols_table const & symbols, expression_tree const & expressions)
{
    std::vector<std::string> result;
    for(symbolic::object* obj: symbolic::extract<symbolic::object>(expressions, symbols))
    {
      if(symbolic::buffer* sym = dynamic_cast<symbolic::buffer*>(obj))
        if( sym->hasattr("inc0")  && ! sym->hasattr("inc1"))
        {
          std::string pointer = sym->process("#scalartype* #pointer");
          {
            int pointer_pos = pointer.find_first_of(" ");
            std::string pointer_name = pointer.substr(pointer_pos+1, pointer.length());
            std::string inc0 = sym->process("#inc0");
            std::string type = pointer.substr(0,pointer_pos);
            std::string pointer_dec = "$GLOBAL " + type + " " + pointer_name;
            std::string pointer_def = pointer_dec + " = " + pointer_name + "_bk;";
            std::string judge = "  if(" + inc0 + " < 0)";
            std::string re = pointer_def + "\n"+judge + "\n" +  "    " + pointer_name + " += (1-N) * " + inc0+";\n";
            result.push_back(re);
          }
        }
    }
    return result;
}

inline std::vector<std::string> reduce_2d_negative_inc_process(driver::Device const &, symbolic::symbols_table const & symbols, expression_tree const & expressions)
{
    std::vector<std::string> result;
    for(symbolic::object* obj: symbolic::extract<symbolic::object>(expressions, symbols))
    {
      if(symbolic::buffer* sym = dynamic_cast<symbolic::buffer*>(obj))
        if( sym->hasattr("inc0")  && ! sym->hasattr("inc1"))
        {
          std::string pointer = sym->process("#scalartype* #pointer");
          {
            int pointer_pos = pointer.find_first_of(" ");
            std::string pointer_name = pointer.substr(pointer_pos+1, pointer.length());
            std::string inc0 = sym->process("#inc0");
            std::string type = pointer.substr(0,pointer_pos);
            std::string pointer_dec = "$GLOBAL " + type + " " + pointer_name;
            std::string pointer_def = pointer_dec + " = " + pointer_name + "_bk;";
            std::string judge = "  if(" + inc0 + " < 0)";
            std::string re;
            if(pointer.find("obj3") == std::string::npos )
              re = pointer_def + "\n"+judge + "\n" +  "    " + pointer_name + " += (1-M) * " + inc0+";\n";
            else
              re = pointer_def + "\n"+judge + "\n" +  "    " + pointer_name + " += (1-N) * " + inc0+";\n";
            result.push_back(re);
          }
        }
    }
    return result;
}

}
}
