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
#ifndef ISAAC_SYMBOLIC_ENGINE_PROCESS
#define ISAAC_SYMBOLIC_ENGINE_PROCESS

#include <functional>
#include <typeinfo>
#include "isaac/tools/cpp/string.hpp"
#include "isaac/jit/syntax/expression/expression.h"
#include "isaac/jit/syntax/engine/binder.h"
#include "isaac/jit/syntax/engine/object.h"
#include "isaac/array.h"

namespace isaac
{
namespace symbolic
{

//Traverse
template<class FUN>
inline void traverse(expression_tree const & tree, size_t root, FUN const & fun,
                     std::function<bool(size_t)> const & recurse)
{
  expression_tree::node const & node = tree[root];
  if (node.type==COMPOSITE_OPERATOR_TYPE && recurse(root)){
    traverse(tree, node.binary_operator.lhs, fun, recurse);
    traverse(tree, node.binary_operator.rhs, fun, recurse);
  }
  if (node.type != INVALID_SUBTYPE)
    fun(root);
}

template<class FUN>
inline void traverse(expression_tree const & tree, size_t root, FUN const & fun)
{ return traverse(tree, root, fun,  [](size_t){return true;}); }

template<class FUN>
inline void traverse(expression_tree const & tree, FUN const & fun)
{ return traverse(tree, tree.root(), fun); }


//Extract symbolic types
template<class T>
inline void extract(expression_tree const & tree, symbols_table const & table,
                    size_t root, std::set<std::string>& processed, std::vector<T*>& result, bool array_recurse = true)
{
  auto extract_impl = [&](size_t index)
  {
    symbols_table::const_iterator it = table.find(index);
    if(it!=table.end())
    {
      T* obj = dynamic_cast<T*>(&*it->second);
      if(obj && processed.insert(obj->process("#name")).second)
        result.push_back(obj);
    }
  };
  auto recurse = [&](size_t index){ return array_recurse?true:dynamic_cast<index_modifier*>(&*table.at(index))==0;};
  traverse(tree, root, extract_impl, recurse);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table, std::vector<size_t> roots, bool array_recurse = true)
{
  std::vector<T*> result;
  std::set<std::string> processed;
  for(size_t root: roots)
     extract(tree, table, root, processed, result, array_recurse);
  return result;
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table, size_t root, bool array_recurse = true)
{
  return extract<T>(tree, table, std::vector<size_t>{root}, array_recurse);
}

template<class T>
inline std::vector<T*> extract(expression_tree const & tree, symbols_table const & table)
{
  return extract<T>(tree, table, tree.root());
}

// Filter nodes
std::vector<size_t> find(expression_tree const & tree, size_t root, std::function<bool (expression_tree::node const &)> const & pred);
std::vector<size_t> find(expression_tree const & tree, std::function<bool (expression_tree::node const &)> const & pred);

std::vector<size_t> assignments(expression_tree const & tree);
std::vector<size_t> lhs_of(expression_tree const & tree, std::vector<size_t> const & in);
std::vector<size_t> rhs_of(expression_tree const & tree, std::vector<size_t> const & in);

// Hash
std::string hash(expression_tree const & tree);

//Set arguments
void set_arguments(expression_tree const & tree, driver::Kernel & kernel, unsigned int& current_arg);

//Symbolize
symbols_table symbolize(isaac::expression_tree const & expression);

}
}


#endif
