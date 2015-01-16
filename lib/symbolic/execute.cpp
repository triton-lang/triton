#include <cassert>
#include <list>
#include <vector>
#include "atidlas/types.h"
#include "atidlas/array.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/model/model.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, lhs_rhs_element*> > breakpoints_t;

    /** @brief Determine if a particular operation requires a breakpoint
    *
    *  @return std::pair<bool, expression_type> the first element is weather or not a breakpoint is required
    *          The second element is the type of the new operation
    */
    static std::pair<bool, expression_type> is_breakpoint(expression_type current_type, op_element op)
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
                                              case OPERATOR_MATRIX_PRODUCT_NN_TYPE:      return make_pair(true, MATRIX_PRODUCT_NN_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_TN_TYPE:      return make_pair(true, MATRIX_PRODUCT_TN_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_NT_TYPE:      return make_pair(true, MATRIX_PRODUCT_NT_TYPE);\
                                              case OPERATOR_MATRIX_PRODUCT_TT_TYPE:      return make_pair(true, MATRIX_PRODUCT_TT_TYPE);\
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
            HANDLE_MATRIX_PRODUCT(true);
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
    static void parse(symbolic_expression::container_type&array, size_t idx,
               expression_type current_type,
               breakpoints_t & breakpoints,
               expression_type & final_type)
    {
      symbolic_expression_node & node = array[idx];
      if (node.lhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      {
        std::pair<bool, expression_type> breakpoint = is_breakpoint(current_type, array[node.lhs.node_index].op);
        expression_type next_type = breakpoint.second;
        if(breakpoint.first)
            breakpoints.push_back(std::make_pair(next_type, &node.lhs));
        else
            final_type = next_type;
        parse(array, node.lhs.node_index, next_type, breakpoints, final_type);
      }
      if (node.rhs.type_family == COMPOSITE_OPERATOR_FAMILY)
      {
        std::pair<bool, expression_type> breakpoint = is_breakpoint(current_type, array[node.rhs.node_index].op);
        expression_type next_type = breakpoint.second;
        if(breakpoint.first)
            breakpoints.push_back(std::make_pair(next_type, &node.rhs));
        else
            final_type = next_type;
        parse(array, node.rhs.node_index, next_type, breakpoints, final_type);
      }
    }

  }

  /** @brief Executes a symbolic_expression on the given models map*/
  void execute(atidlas::symbolic_expression & symbolic_expression, model_map_t & models)
  {
    cl::Context const & context = symbolic_expression.context();
    size_t rootidx = symbolic_expression.root();
    symbolic_expression::container_type & tree = const_cast<symbolic_expression::container_type &>(symbolic_expression.tree());
    symbolic_expression_node root_save = tree[rootidx];

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
    std::vector<tools::shared_ptr<obj_base> > temporaries_;

    /*----Compute required temporaries----*/
    for(detail::breakpoints_t::reverse_iterator rit = breakpoints.rbegin() ; rit != breakpoints.rend() ; ++rit)
    {
      tools::shared_ptr<model> const & pmodel = models[std::make_pair(rit->first, dtype)];
      symbolic_expression_node const & node = tree[rit->second->node_index];
      symbolic_expression_node const & lmost = lhs_most(tree, node);

      //Creates temporary
      tools::shared_ptr<obj_base> tmp;
      switch(rit->first){
        case SCALAR_AXPY_TYPE:
        case REDUCTION_TYPE:           tmp = tools::shared_ptr<obj_base>(new array(1, dtype, context));                                                        break;

        case VECTOR_AXPY_TYPE:         tmp = tools::shared_ptr<obj_base>(new array(lmost.lhs.array->shape()._1, dtype, context));                              break;
        case ROW_WISE_REDUCTION_TYPE:  tmp = tools::shared_ptr<obj_base>(new array(lmost.lhs.array->shape()._1, dtype, context));                              break;
        case COL_WISE_REDUCTION_TYPE:  tmp = tools::shared_ptr<obj_base>(new array(lmost.lhs.array->shape()._2, dtype, context));                              break;

        case MATRIX_AXPY_TYPE:         tmp = tools::shared_ptr<obj_base>(new array(lmost.lhs.array->shape()._1, lmost.lhs.array->shape()._2, dtype, context)); break;
        case MATRIX_PRODUCT_NN_TYPE:   tmp = tools::shared_ptr<obj_base>(new array(node.lhs.array->shape()._1, node.rhs.array->shape()._2, dtype, context));   break;
        case MATRIX_PRODUCT_NT_TYPE:   tmp = tools::shared_ptr<obj_base>(new array(node.lhs.array->shape()._1, node.rhs.array->shape()._1, dtype, context));   break;
        case MATRIX_PRODUCT_TN_TYPE:   tmp = tools::shared_ptr<obj_base>(new array(node.lhs.array->shape()._2, node.rhs.array->shape()._2, dtype, context));   break;
        case MATRIX_PRODUCT_TT_TYPE:   tmp = tools::shared_ptr<obj_base>(new array(node.lhs.array->shape()._2, node.rhs.array->shape()._1, dtype, context));   break;

        default: throw "This shouldn't happen. Ever.";
      }
      temporaries_.push_back(tmp);

      tree[rootidx].op.type = OPERATOR_ASSIGN_TYPE;
      tree[rootidx].lhs = lhs_rhs_element((array const &)*tmp);
      tree[rootidx].rhs = *rit->second;
      tree[rootidx].rhs.type_family = rit->second->type_family;

      //Execute
      pmodel->execute(symbolic_expression);
      tree[rootidx] = root_save;

      //Incorporates the temporary within the symbolic_expression
      rit->second->dtype = dtype;
      rit->second->type_family = ARRAY_TYPE_FAMILY;
      rit->second->subtype = DENSE_ARRAY_TYPE;
      rit->second->array = (array*)tmp.get();
    }

    /*-----Compute final expression-----*/
    models[std::make_pair(final_type, dtype)]->execute(symbolic_expression);
  }

}
