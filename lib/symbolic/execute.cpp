#include <cassert>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/types.h"
#include "isaac/array.h"
#include <CL/cl.hpp>
#include "isaac/model/model.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, lhs_rhs_element*> > breakpoints_t;

    /** @brief Determine if a particular operation requires a breakpoint
    *
    *  @return std::pair<bool, expression_type> the first element is weather or not a breakpoint is required
    *          The second element is the type of the new operation
    */
    static std::pair<bool, expression_type> is_breakpoint(expression_type current_type, op_element op, bool is_first)
    {
      using std::make_pair;

      switch(current_type)
      {

//BLAS1 Helpers
#define HANDLE_VECTOR_AXPY(tmp)       case OPERATOR_BINARY_TYPE_FAMILY\
                                      case OPERATOR_UNARY_TYPE_FAMILY:                  return make_pair(tmp, VECTOR_AXPY_TYPE)
#define HANDLE_VECTOR_REDUCTION(tmp)  case OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY:      return make_pair(tmp, REDUCTION_TYPE)
#define HANDLE_MATRIX_AXPY(tmp)       case OPERATOR_BINARY_TYPE_FAMILY:\
                                      case OPERATOR_UNARY_TYPE_FAMILY:                 return make_pair(tmp, MATRIX_AXPY_TYPE)

//BLAS2 Helpers
#define HANDLE_ROWS_REDUCTION(tmp)    case OPERATOR_ROWS_REDUCTION_TYPE_FAMILY:        return make_pair(tmp, ROW_WISE_REDUCTION_TYPE)
#define HANDLE_COLUMNS_REDUCTION(tmp) case OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY:     return make_pair(tmp, COL_WISE_REDUCTION_TYPE)

//BLAS3 Helpers
#define HANDLE_MATRIX_PRODUCT(tmp)    case OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY:\
                                          switch(op.type){\
                                              case OPERATOR_MATRIX_PRODUCT_NN_TYPE:      return make_pair(tmp, MATRIX_PRODUCT_NN_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_TN_TYPE:      return make_pair(tmp, MATRIX_PRODUCT_TN_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_NT_TYPE:      return make_pair(tmp, MATRIX_PRODUCT_NT_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_TT_TYPE:      return make_pair(tmp, MATRIX_PRODUCT_TT_TYPE);\
                                              default: assert(false && "This misformed expression shouldn't occur");\
                                          }

      // Inside a SCALAR AXPY
        case SCALAR_AXPY_TYPE:
          // Reduction: No temporary
          switch(op.type_family){
            HANDLE_VECTOR_REDUCTION(false);
            default: break;
          }
          break;

        // Inside a VECTOR AXPY
        case VECTOR_AXPY_TYPE:
          switch(op.type_family){
            HANDLE_VECTOR_REDUCTION(true);
            HANDLE_ROWS_REDUCTION(false);
            HANDLE_COLUMNS_REDUCTION(false);
            default: break;
          }
          break;

        // Inside a REDUCTION
        case REDUCTION_TYPE:
          switch(op.type_family){
            HANDLE_VECTOR_REDUCTION(true);
            default: break;
          }
          break;

        // Inside a MATRIX AXPY
        //  - MATRIX PRODUCTS are temporaries
        //  - REDUCTIONS are temporaries
        case MATRIX_AXPY_TYPE:
          switch(op.type_family){
            HANDLE_VECTOR_REDUCTION(true);
            HANDLE_MATRIX_PRODUCT(!is_first);
            default: break;
          }
          break;

        // Inside a MATRIX PRODUCT:
        //  - AXPY are temporaries
        //  - MATRIX PRODUCTS are temporaries
        case MATRIX_PRODUCT_NN_TYPE:
        case MATRIX_PRODUCT_NT_TYPE:
        case MATRIX_PRODUCT_TN_TYPE:
        case MATRIX_PRODUCT_TT_TYPE:
          switch(op.type_family){
            HANDLE_MATRIX_AXPY(true);
            HANDLE_MATRIX_PRODUCT(true);
            HANDLE_VECTOR_REDUCTION(true);
            default: break;
          }

        default:
          break;
      }

#undef HANDLE_VECTOR_AXPY
#undef HANDLE_VECTOR_REDUCTION
#undef HANDLE_MATRIX_AXPY
#undef HANDLE_ROWS_REDUCTION
#undef HANDLE_COLUMN_REDUCTION
#undef HANDLE_MATRIX_PRODUCT

      return make_pair(false, current_type);
    }

    /** @brief Parses the breakpoints for a given expression tree */
    static void parse(array_expression::container_type&array, size_t idx,
               expression_type current_type,
               breakpoints_t & breakpoints,
               expression_type & final_type,
               bool is_first = true)
    {
      array_expression::node & node = array[idx];
      if (node.lhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      {
        std::pair<bool, expression_type> breakpoint = is_breakpoint(current_type, array[node.lhs.node_index].op, is_first);
        expression_type next_type = breakpoint.second;
        if(breakpoint.first)
            breakpoints.push_back(std::make_pair(next_type, &node.lhs));
        else
            final_type = next_type;
        parse(array, node.lhs.node_index, next_type, breakpoints, final_type, false);
      }
      if (node.rhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      {
        std::pair<bool, expression_type> breakpoint = is_breakpoint(current_type, array[node.rhs.node_index].op, is_first);
        expression_type next_type = breakpoint.second;
        if(breakpoint.first)
            breakpoints.push_back(std::make_pair(next_type, &node.rhs));
        else
            final_type = next_type;
        parse(array, node.rhs.node_index, next_type, breakpoints, final_type, false);
      }
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
    expression_type final_type = current_type;

    /*----Parse required temporaries-----*/
    detail::parse(tree, rootidx, current_type, breakpoints, final_type);
    std::vector<tools::shared_ptr<array> > temporaries_;

    /*----Compute required temporaries----*/
    for(detail::breakpoints_t::reverse_iterator rit = breakpoints.rbegin() ; rit != breakpoints.rend() ; ++rit)
    {
      tools::shared_ptr<model> const & pmodel = models[std::make_pair(rit->first, dtype)];
      array_expression::node const & node = tree[rit->second->node_index];
      array_expression::node const & lmost = lhs_most(tree, node);

      //Creates temporary
      tools::shared_ptr<array> tmp;
      switch(rit->first){
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
      tree[rootidx].rhs = *rit->second;
      tree[rootidx].rhs.type_family = rit->second->type_family;

      //Execute
      pmodel->execute(controller<expressions_tuple>(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
      tree[rootidx] = root_save;

      //Incorporates the temporary within the array_expression
      fill(*rit->second, (array&)*tmp);
    }

    /*-----Compute final expression-----*/
    models[std::make_pair(final_type, dtype)]->execute(controller<expressions_tuple>(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
  }

}
