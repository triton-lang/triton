#include <assert.h>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/types.h"
#include "isaac/array.h"
#include "isaac/model/database.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, lhs_rhs_element*> > breakpoints_t;


    inline bool is_mmprod(expression_type x)
    {
        return x==GEMM_NN_TYPE || x==GEMM_NT_TYPE ||
               x==GEMM_TN_TYPE || x==GEMM_TT_TYPE;
    }

    inline bool is_mvprod(expression_type x)
    {
        return x==GEMV_N_TYPE || x==GEMV_T_TYPE;
    }

    inline bool has_temporary_impl(op_element op, expression_type expression, expression_type other, bool is_first)
    {
        bool result = false;
        switch(op.type_family)
        {
            case OPERATOR_UNARY_TYPE_FAMILY:
            case OPERATOR_BINARY_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || (result |= expression==GEMV_N_TYPE && other==GEMV_T_TYPE)
                          || (result |= expression==GEMV_T_TYPE && other==GEMV_N_TYPE);
                break;
            case OPERATOR_VECTOR_DOT_TYPE_FAMILY:
                result |= is_mvprod(expression)
                          || expression==DOT_TYPE;
                break;
            case OPERATOR_ROWS_DOT_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==DOT_TYPE;
                break;
            case OPERATOR_COLUMNS_DOT_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==DOT_TYPE;
                break;
            case OPERATOR_GEMM_TYPE_FAMILY:
                result |= (is_mmprod(expression) && !is_first)
                          || is_mvprod(expression)
                          || expression==DOT_TYPE;
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
                    return GER_TYPE;
                return left;
            case OPERATOR_BINARY_TYPE_FAMILY:
                if(left == GEMV_N_TYPE || right == GEMV_N_TYPE) return GEMV_N_TYPE;
                else if(left == GEMV_T_TYPE || right == GEMV_T_TYPE) return GEMV_T_TYPE;
                else if(left == DOT_TYPE || right == DOT_TYPE) return DOT_TYPE;
                else if(left == AXPY_TYPE || right == AXPY_TYPE) return op.type==OPERATOR_OUTER_PROD_TYPE?GER_TYPE:AXPY_TYPE;
                else if(left == GER_TYPE || right == GER_TYPE) return GER_TYPE;
                else if(is_mmprod(left) || is_mmprod(right)) return GER_TYPE;
                throw;
            case OPERATOR_VECTOR_DOT_TYPE_FAMILY:
                return DOT_TYPE;
            case OPERATOR_ROWS_DOT_TYPE_FAMILY:
                return GEMV_N_TYPE;
            case OPERATOR_COLUMNS_DOT_TYPE_FAMILY:
                return GEMV_T_TYPE;
            case OPERATOR_GEMM_TYPE_FAMILY:
                if(op.type==OPERATOR_GEMM_NN_TYPE) return GEMM_NN_TYPE;
                else if(op.type==OPERATOR_GEMM_TN_TYPE) return GEMM_TN_TYPE;
                else if(op.type==OPERATOR_GEMM_NT_TYPE) return GEMM_NT_TYPE;
                else return GEMM_TT_TYPE;
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
              type_left = AXPY_TYPE;
          else
              type_left = GER_TYPE;
      }

      //Right
      expression_type type_right = INVALID_EXPRESSION_TYPE;
      if (node.rhs.type_family == COMPOSITE_OPERATOR_FAMILY)
          parse(array, node.rhs.node_index, breakpoints, type_right, false);
      else if(node.rhs.subtype == DENSE_ARRAY_TYPE)
      {
          if(node.rhs.array->nshape()==1)
              type_right = AXPY_TYPE;
          else
              type_right = GER_TYPE;
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
  void execute(controller<array_expression> const & c, database::map_type & models)
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
        if(root_save.lhs.array->nshape()<=1)
          current_type=AXPY_TYPE;
        else
          current_type=GER_TYPE;
        final_type = current_type;

        /*----Parse required temporaries-----*/
        detail::parse(tree, rootidx, breakpoints, final_type);
        std::vector<std::shared_ptr<array> > temporaries_;

        /*----Compute required temporaries----*/
        for(detail::breakpoints_t::iterator it = breakpoints.begin() ; it != breakpoints.end() ; ++it)
        {
          std::shared_ptr<model> const & pmodel = models[std::make_pair(it->first, dtype)];
          array_expression::node const & node = tree[it->second->node_index];
          array_expression::node const & lmost = lhs_most(tree, node);

          //Creates temporary
          std::shared_ptr<array> tmp;
          switch(it->first){
            case DOT_TYPE:           tmp = std::shared_ptr<array>(new array(1, dtype, context));                                                        break;

            case AXPY_TYPE:         tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case GEMV_N_TYPE:  tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case GEMV_T_TYPE:  tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[1], dtype, context));                              break;

            case GER_TYPE:         tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], lmost.lhs.array->shape()[1], dtype, context)); break;
            case GEMM_NN_TYPE:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[1], dtype, context));   break;
            case GEMM_NT_TYPE:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[0], dtype, context));   break;
            case GEMM_TN_TYPE:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[1], dtype, context));   break;
            case GEMM_TT_TYPE:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[0], dtype, context));   break;

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
