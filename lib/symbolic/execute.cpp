#include <assert.h>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/types.h"
#include "isaac/array.h"
#include <CL/cl.hpp>
#include "isaac/model/model.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, lhs_rhs_element*> > breakpoints_t;


    inline bool is_mmprod(expression_type x)
    {
        return x==MATRIX_PRODUCT_NN_TYPE || x==MATRIX_PRODUCT_NT_TYPE ||
               x==MATRIX_PRODUCT_TN_TYPE || x==MATRIX_PRODUCT_TT_TYPE;
    }

    inline bool is_mvprod(expression_type x)
    {
        return x==ROW_WISE_REDUCTION_TYPE || x==COL_WISE_REDUCTION_TYPE;
    }

    inline bool has_temporary_impl(op_element op, expression_type expression, expression_type other, bool is_first)
    {
        bool result = false;
        switch(op.type_family)
        {
            case OPERATOR_UNARY_TYPE_FAMILY:
            case OPERATOR_BINARY_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || (result |= expression==ROW_WISE_REDUCTION_TYPE && other==COL_WISE_REDUCTION_TYPE)
                          || (result |= expression==COL_WISE_REDUCTION_TYPE && other==ROW_WISE_REDUCTION_TYPE);
                break;
            case OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY:
                result |= is_mvprod(expression)
                          || expression==REDUCTION_TYPE;
                break;
            case OPERATOR_ROWS_REDUCTION_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCTION_TYPE;
                break;
            case OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCTION_TYPE;
                break;
            case OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY:
                result |= (is_mmprod(expression) && !is_first)
                          || is_mvprod(expression)
                          || expression==REDUCTION_TYPE;
                break;
            default:
                break;
        }
        return result;
    }

    inline std::pair<bool, bool> has_temporary(op_element op, expression_type left, expression_type right, bool is_first)
    {
        bool has_temporary_left = has_temporary_impl(op, left, right, is_first);
        bool has_temporary_right = has_temporary_impl(op, right, left, is_first);
        return std::make_pair(has_temporary_left, has_temporary_right);
    }

    inline expression_type merge(op_element op, expression_type left, expression_type right)
    {
        switch(op.type_family)
        {
            case OPERATOR_UNARY_TYPE_FAMILY:
                if(is_mmprod(left))
                    return MATRIX_AXPY_TYPE;
                return left;
            case OPERATOR_BINARY_TYPE_FAMILY:
                if(left == ROW_WISE_REDUCTION_TYPE || right == ROW_WISE_REDUCTION_TYPE) return ROW_WISE_REDUCTION_TYPE;
                else if(left == COL_WISE_REDUCTION_TYPE || right == COL_WISE_REDUCTION_TYPE) return COL_WISE_REDUCTION_TYPE;
                else if(left == REDUCTION_TYPE || right == REDUCTION_TYPE) return REDUCTION_TYPE;
                else if(left == VECTOR_AXPY_TYPE || right == VECTOR_AXPY_TYPE) return op.type==OPERATOR_OUTER_PROD_TYPE?MATRIX_AXPY_TYPE:VECTOR_AXPY_TYPE;
                else if(left == MATRIX_AXPY_TYPE || right == MATRIX_AXPY_TYPE) return MATRIX_AXPY_TYPE;
                else if(is_mmprod(left) || is_mmprod(right)) return MATRIX_AXPY_TYPE;
                std::cout << left << " " << right << std::endl;
                throw;
            case OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY:
                return REDUCTION_TYPE;
            case OPERATOR_ROWS_REDUCTION_TYPE_FAMILY:
                return ROW_WISE_REDUCTION_TYPE;
            case OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY:
                return COL_WISE_REDUCTION_TYPE;
            case OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY:
                if(op.type==OPERATOR_MATRIX_PRODUCT_NN_TYPE) return MATRIX_PRODUCT_NN_TYPE;
                else if(op.type==OPERATOR_MATRIX_PRODUCT_TN_TYPE) return MATRIX_PRODUCT_TN_TYPE;
                else if(op.type==OPERATOR_MATRIX_PRODUCT_NT_TYPE) return MATRIX_PRODUCT_NT_TYPE;
                else return MATRIX_PRODUCT_TT_TYPE;
            default:
                throw;
        }
    }

    /** @brief Parses the breakpoints for a given expression tree */
    static void parse(array_expression::container_type&array, size_t idx,
               breakpoints_t & breakpoints,
               expression_type & final_type,
               bool is_first = true)
    {
      array_expression::node & node = array[idx];

      //Left
      expression_type type_left = INVALID_EXPRESSION_TYPE;
      if (node.lhs.type_family == COMPOSITE_OPERATOR_FAMILY)
          parse(array, node.lhs.node_index, breakpoints, type_left, false);
      else if(node.lhs.subtype == DENSE_ARRAY_TYPE)
      {
          if(node.lhs.array->nshape()==1)
              type_left = VECTOR_AXPY_TYPE;
          else
              type_left = MATRIX_AXPY_TYPE;
      }

      //Right
      expression_type type_right = INVALID_EXPRESSION_TYPE;
      if (node.rhs.type_family == COMPOSITE_OPERATOR_FAMILY)
          parse(array, node.rhs.node_index, breakpoints, type_right, false);
      else if(node.rhs.subtype == DENSE_ARRAY_TYPE)
      {
          if(node.rhs.array->nshape()==1)
              type_right = VECTOR_AXPY_TYPE;
          else
              type_right = MATRIX_AXPY_TYPE;
      }


      final_type = merge(array[idx].op, type_left, type_right);
      std::pair<bool, bool> tmp = has_temporary(array[idx].op, type_left, type_right, is_first);
      if(tmp.first)
          breakpoints.push_back(std::make_pair(type_left, &array[idx].lhs));
      if(tmp.second)
          breakpoints.push_back(std::make_pair(type_right, &array[idx].rhs));
    }
  }

  /** @brief Executes a array_expression on the given models map*/
  void execute(controller<array_expression> const & c, model_map_t & models)
  {
    array_expression expression = c.x();
    driver::Context const & context = expression.context();
    size_t rootidx = expression.root();
    array_expression::container_type & tree = const_cast<array_expression::container_type &>(expression.tree());
    array_expression::node root_save = tree[rootidx];

    //Todo: technically the datatype should be per temporary
    numeric_type dtype = root_save.lhs.dtype;

    expression_type final_type;
    //GEMM
    if(symbolic::preset::gemm::args args = symbolic::preset::gemm::check(tree, rootidx)){
        final_type = args.type;
    }
    //Default
    else
    {
        detail::breakpoints_t breakpoints;
        breakpoints.reserve(8);

        //Init
        expression_type current_type;
        if(root_save.lhs.array->nshape()==0)
          current_type = SCALAR_AXPY_TYPE;
        else if(root_save.lhs.array->nshape()==1)
          current_type=VECTOR_AXPY_TYPE;
        else
          current_type=MATRIX_AXPY_TYPE;
        final_type = current_type;

        /*----Parse required temporaries-----*/
        detail::parse(tree, rootidx, breakpoints, final_type);
        std::vector<tools::shared_ptr<array> > temporaries_;

        /*----Compute required temporaries----*/
        for(detail::breakpoints_t::iterator it = breakpoints.begin() ; it != breakpoints.end() ; ++it)
        {
          tools::shared_ptr<model> const & pmodel = models[std::make_pair(it->first, dtype)];
          array_expression::node const & node = tree[it->second->node_index];
          array_expression::node const & lmost = lhs_most(tree, node);

          //Creates temporary
          tools::shared_ptr<array> tmp;
          switch(it->first){
            case SCALAR_AXPY_TYPE:
            case REDUCTION_TYPE:           tmp = tools::shared_ptr<array>(new array(1, dtype, context));                                                        break;

            case VECTOR_AXPY_TYPE:         tmp = tools::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case ROW_WISE_REDUCTION_TYPE:  tmp = tools::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case COL_WISE_REDUCTION_TYPE:  tmp = tools::shared_ptr<array>(new array(lmost.lhs.array->shape()[1], dtype, context));                              break;

            case MATRIX_AXPY_TYPE:         tmp = tools::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], lmost.lhs.array->shape()[1], dtype, context)); break;
            case MATRIX_PRODUCT_NN_TYPE:   tmp = tools::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[1], dtype, context));   break;
            case MATRIX_PRODUCT_NT_TYPE:   tmp = tools::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[0], dtype, context));   break;
            case MATRIX_PRODUCT_TN_TYPE:   tmp = tools::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[1], dtype, context));   break;
            case MATRIX_PRODUCT_TT_TYPE:   tmp = tools::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[0], dtype, context));   break;

            default: throw std::invalid_argument("Unrecognized operation");
          }
          temporaries_.push_back(tmp);

          tree[rootidx].op.type = OPERATOR_ASSIGN_TYPE;
          fill(tree[rootidx].lhs, (array&)*tmp);
          tree[rootidx].rhs = *it->second;
          tree[rootidx].rhs.type_family = it->second->type_family;

          //Execute
          pmodel->execute(controller<expressions_tuple>(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
          tree[rootidx] = root_save;

          //Incorporates the temporary within the array_expression
          fill(*it->second, (array&)*tmp);
        }
    }

    /*-----Compute final expression-----*/
    models[std::make_pair(final_type, dtype)]->execute(controller<expressions_tuple>(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
  }

}
