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

#include "isaac/jit/syntax/engine/process.h"

namespace isaac
{
namespace symbolic
{

// Filter nodes
std::vector<size_t> find(expression_tree const & tree, size_t root, std::function<bool (expression_tree::node const &)> const & pred)
{
  std::vector<size_t> result;
  auto fun = [&](size_t index) { if(pred(tree[index])) result.push_back(index); };
  traverse(tree, root, fun);
  return result;
}

std::vector<size_t> find(expression_tree const & tree, std::function<bool (expression_tree::node const &)> const & pred)
{
  return find(tree, tree.root(), pred);
}

std::vector<size_t> assignments(expression_tree const & tree)
{
  return find(tree, [&](expression_tree::node const & node)
            {return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);}
          );
}

std::vector<size_t> lhs_of(expression_tree const & tree, std::vector<size_t> const & in)
{
  std::vector<size_t> result;
  for(size_t idx: in)
    result.push_back(tree[idx].binary_operator.lhs);
  return result;
}

std::vector<size_t> rhs_of(expression_tree const & tree, std::vector<size_t> const & in)
{
  std::vector<size_t> result;
  for(size_t idx: in)
    result.push_back(tree[idx].binary_operator.rhs);
  return result;
}


// Hash
std::string hash(expression_tree const & tree)
{
  driver::backend_type backend = tree.context().backend();

  char program_name[256];
  char* ptr = program_name;
  bind_independent binder(backend);

  auto hash_impl = [&](size_t idx)
  {
    expression_tree::node const & node = tree.data()[idx];
    if(node.type==DENSE_ARRAY_TYPE)
    {
      for(size_t i = 0 ; i < node.shape.size() ; ++i)
        *ptr++= node.shape[i]>1?'n':'1';
      if(node.ld[0]>1) *ptr++= 's';
      *ptr++=(char)node.dtype;
      tools::fast_append(ptr, binder.get(node.array.handle, false));
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE)
      tools::fast_append(ptr,node.binary_operator.op.type);
  };

  traverse(tree, hash_impl);

  *ptr='\0';

  return std::string(program_name);
}

//Set arguments
void set_arguments(expression_tree const & tree, driver::Kernel & kernel, unsigned int& current_arg)
{
  driver::backend_type backend = tree.context().backend();

  //Create binder
  std::unique_ptr<symbolic_binder> binder(new bind_independent(backend));
  //assigned
  std::vector<size_t> assignee = symbolic::find(tree, [&](expression_tree::node const & node){return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);});
  for(size_t& x: assignee) x = tree[x].binary_operator.lhs;

  //set_arguments_impl
  auto set_arguments_impl = [&](size_t index)
  {
    expression_tree::node const & node = tree.data()[index];
    if(node.type==VALUE_SCALAR_TYPE)
      kernel.setArg(current_arg++,value_scalar(node.scalar,node.dtype));
    else if(node.type==DENSE_ARRAY_TYPE)
    {
      array_holder const & array = node.array;
      bool is_assigned = std::find(assignee.begin(), assignee.end(), index)!=assignee.end();
      bool is_bound = binder->bind(array.handle, is_assigned);
      if (is_bound)
      {
          if(backend==driver::OPENCL)
            kernel.setArg(current_arg++, array.handle.cl);
          else
            kernel.setArg(current_arg++, array.handle.cu);
          kernel.setSizeArg(current_arg++, array.start);
          for(size_t i = 0 ; i < node.shape.size() ; i++)
          {
            if(node.shape[i] > 1)
              kernel.setSizeArg(current_arg++, node.ld[i]);
          }
      }
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE && node.binary_operator.op.type == RESHAPE_TYPE)
    {
      tuple const & new_shape = node.shape;
      int_t current = 1;
      for(size_t i = 1 ; i < new_shape.size() ; ++i){
        current *= new_shape[i-1];
        if(new_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }

      tuple const & old_shape = tree.data()[node.binary_operator.lhs].shape;
      current = 1;
      for(unsigned int i = 1 ; i < old_shape.size() ; ++i){
        current *= old_shape[i-1];
        if(old_shape[i] > 1)
          kernel.setSizeArg(current_arg++, current);
      }
    }
  };


  //Traverse
  traverse(tree, set_arguments_impl);
}

//Symbolize
template<class T, class... Args>
std::shared_ptr<object> make_symbolic(Args&&... args)
{
  return std::shared_ptr<object>(new T(std::forward<Args>(args)...));
}

symbols_table symbolize(isaac::expression_tree const & tree)
{
  driver::Context const & context = tree.context();

  //binder
  symbols_table table;
  std::unique_ptr<symbolic_binder> binder(new bind_independent(context.backend()));
  //assigned
  std::vector<size_t> assignee = symbolic::find(tree, [&](expression_tree::node const & node){return node.type==COMPOSITE_OPERATOR_TYPE && is_assignment(node.binary_operator.op.type);});
  for(size_t& x: assignee) x = tree[x].binary_operator.lhs;

  //symbolize_impl
  auto symbolize_impl = [&](size_t root)
  {
    expression_tree::node const & node = tree.data()[root];
    std::string dtype = to_string(node.dtype);
    if(node.type==VALUE_SCALAR_TYPE)
      table.insert({root, make_symbolic<host_scalar>(context, dtype, binder->get())});
    else if(node.type==DENSE_ARRAY_TYPE){
      bool is_assigned = std::find(assignee.begin(), assignee.end(), root)!=assignee.end();
      table.insert({root, make_symbolic<buffer>(context, dtype, binder->get(node.array.handle, is_assigned), node.shape, node.ld)});
    }
    else if(node.type==COMPOSITE_OPERATOR_TYPE)
    {
      unsigned int id = binder->get();
      op_element op = node.binary_operator.op;
      //Index modifier
      if(op.type==RESHAPE_TYPE)
        table.insert({root, make_symbolic<reshape>(dtype, id, root, op, tree, table)});
      else if(op.type==TRANS_TYPE)
        table.insert({root, make_symbolic<trans>(dtype, id, root, op, tree, table)});
      else if(op.type==DIAG_VECTOR_TYPE)
        table.insert({root, make_symbolic<diag_vector>(dtype, id, root, op, tree, table)});
      //Unary arithmetic
      else if(op.type_family==UNARY_ARITHMETIC)
        table.insert({root, make_symbolic<unary_arithmetic_node>(id, root, op, tree, table)});
      //Binary arithmetic
      else if(op.type_family==BINARY_ARITHMETIC)
        table.insert({root, make_symbolic<binary_arithmetic_node>(id, root, op, tree, table)});
      //1D Reduction
      else if (op.type_family==REDUCE)
        table.insert({root, make_symbolic<reduce_1d>(id, root, op, tree, table)});
      //2D reduction
      else if (op.type_family==REDUCE_ROWS || op.type_family==REDUCE_COLUMNS)
        table.insert({root, make_symbolic<reduce_2d>(id, root, op, tree, table)});
    }
  };

  //traverse
  traverse(tree, symbolize_impl);

  return table;
}

}
}
