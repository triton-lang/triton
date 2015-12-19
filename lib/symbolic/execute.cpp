#include <assert.h>
#include <list>
#include <vector>
#include <stdexcept>
#include "isaac/types.h"
#include "isaac/array.h"
#include "isaac/profiles/profiles.h"
#include "isaac/symbolic/expression.h"
#include "isaac/symbolic/preset.h"

namespace isaac
{

  namespace detail
  {
    typedef std::vector<std::pair<expression_type, tree_node*> > breakpoints_t;


    inline bool is_mmprod(expression_type x)
    {
        return x==MATRIX_PRODUCT_NN || x==MATRIX_PRODUCT_NT ||
               x==MATRIX_PRODUCT_TN || x==MATRIX_PRODUCT_TT;
    }

    inline bool is_mvprod(expression_type x)
    {
        return x==REDUCE_2D_ROWS || x==REDUCE_2D_COLS;
    }

    inline bool has_temporary_impl(op_element op, expression_type expression, expression_type other, bool is_first)
    {
        bool result = false;
        switch(op.type_family)
        {
            case UNARY_TYPE_FAMILY:
            case BINARY_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || (result |= expression==REDUCE_2D_ROWS && other==REDUCE_2D_COLS)
                          || (result |= expression==REDUCE_2D_COLS && other==REDUCE_2D_ROWS);
                break;
            case VECTOR_DOT_TYPE_FAMILY:
                result |= is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case ROWS_DOT_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case COLUMNS_DOT_TYPE_FAMILY:
                result |= is_mmprod(expression)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
                break;
            case MATRIX_PRODUCT_TYPE_FAMILY:
                result |= (is_mmprod(expression) && !is_first)
                          || is_mvprod(expression)
                          || expression==REDUCE_1D;
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
            case UNARY_TYPE_FAMILY:
                if(is_mmprod(left))
                    return ELEMENTWISE_2D;
                return left;
            case BINARY_TYPE_FAMILY:
                if(left == REDUCE_2D_ROWS || right == REDUCE_2D_ROWS) return REDUCE_2D_ROWS;
                else if(left == REDUCE_2D_COLS || right == REDUCE_2D_COLS) return REDUCE_2D_COLS;
                else if(left == REDUCE_1D || right == REDUCE_1D) return REDUCE_1D;
                else if(left == ELEMENTWISE_2D || right == ELEMENTWISE_2D) return ELEMENTWISE_2D;
                else if(left == ELEMENTWISE_1D || right == ELEMENTWISE_1D) return op.type==OUTER_PROD_TYPE?ELEMENTWISE_2D:ELEMENTWISE_1D;
                else if(is_mmprod(left) || is_mmprod(right)) return ELEMENTWISE_2D;
                else if(right == INVALID_EXPRESSION_TYPE) return left;
                else if(left == INVALID_EXPRESSION_TYPE) return right;
                throw;
            case VECTOR_DOT_TYPE_FAMILY:
                return REDUCE_1D;
            case ROWS_DOT_TYPE_FAMILY:
                return REDUCE_2D_ROWS;
            case COLUMNS_DOT_TYPE_FAMILY:
                return REDUCE_2D_COLS;
            case MATRIX_PRODUCT_TYPE_FAMILY:
                if(op.type==MATRIX_PRODUCT_NN_TYPE) return MATRIX_PRODUCT_NN;
                else if(op.type==MATRIX_PRODUCT_TN_TYPE) return MATRIX_PRODUCT_TN;
                else if(op.type==MATRIX_PRODUCT_NT_TYPE) return MATRIX_PRODUCT_NT;
                else return MATRIX_PRODUCT_TT;
            default:
                throw;
        }
    }

    /** @brief Parses the breakpoints for a given expression tree */
    static void parse(expression_tree::container_type&array, size_t idx,
               breakpoints_t & breakpoints,
               expression_type & final_type,
               bool is_first = true)
    {
      expression_tree::node & node = array[idx];

      auto ng1 = [](shape_t const & shape){ size_t res = 0 ; for(size_t i = 0 ; i < shape.size() ; ++i) res += (shape[i] > 1); return res;};
      //Left
      expression_type type_left = INVALID_EXPRESSION_TYPE;
      if (node.lhs.subtype == COMPOSITE_OPERATOR_TYPE)
          parse(array, node.lhs.node_index, breakpoints, type_left, false);
      else if(node.lhs.subtype == DENSE_ARRAY_TYPE)
      {
          if(node.op.type==MATRIX_ROW_TYPE || node.op.type==MATRIX_COLUMN_TYPE || ng1(node.lhs.array->shape())<=1)
              type_left = ELEMENTWISE_1D;
          else
              type_left = ELEMENTWISE_2D;
      }

      //Right
      expression_type type_right = INVALID_EXPRESSION_TYPE;
      if (node.rhs.subtype == COMPOSITE_OPERATOR_TYPE)
          parse(array, node.rhs.node_index, breakpoints, type_right, false);
      else if(node.rhs.subtype == DENSE_ARRAY_TYPE)
      {
          if(node.op.type==MATRIX_ROW_TYPE || node.op.type==MATRIX_COLUMN_TYPE || ng1(node.rhs.array->shape())<=1)
              type_right = ELEMENTWISE_1D;
          else
              type_right = ELEMENTWISE_2D;
      }

      final_type = merge(array[idx].op, type_left, type_right);
      std::pair<bool, bool> tmp = has_temporary(array[idx].op, type_left, type_right, is_first);
      if(tmp.first)
          breakpoints.push_back(std::make_pair(type_left, &array[idx].lhs));
      if(tmp.second)
          breakpoints.push_back(std::make_pair(type_right, &array[idx].rhs));
    }
  }

  /** @brief Executes a expression_tree on the given models map*/
  void execute(execution_handler const & c, profiles::map_type & profiles)
  {
    expression_tree expression = c.x();
    driver::Context const & context = expression.context();
    size_t rootidx = expression.root();
    expression_tree::container_type & tree = const_cast<expression_tree::container_type &>(expression.tree());
    expression_tree::node root_save = tree[rootidx];

    //Todo: technically the datatype should be per temporary
    numeric_type dtype = expression.dtype();
    std::vector<std::shared_ptr<array> > temporaries_;

    expression_type final_type;
    //MATRIX_PRODUCT
    if(symbolic::preset::matrix_product::args args = symbolic::preset::matrix_product::check(tree, rootidx)){
        final_type = args.type;
    }
    //Default
    else
    {
        detail::breakpoints_t breakpoints;
        breakpoints.reserve(8);

        //Init
        expression_type current_type;
        auto ng1 = [](shape_t const & shape){ size_t res = 0 ; for(size_t i = 0 ; i < shape.size() ; ++i) res += (shape[i] > 1); return res;};
        if(ng1(expression.shape())<=1)
          current_type=ELEMENTWISE_1D;
        else
          current_type=ELEMENTWISE_2D;
        final_type = current_type;

        /*----Parse required temporaries-----*/
        detail::parse(tree, rootidx, breakpoints, final_type);

        /*----Compute required temporaries----*/
        for(detail::breakpoints_t::iterator it = breakpoints.begin() ; it != breakpoints.end() ; ++it)
        {
          std::shared_ptr<profiles::value_type> const & profile = profiles[std::make_pair(it->first, dtype)];
          expression_tree::node const & node = tree[it->second->node_index];
          expression_tree::node const & lmost = lhs_most(tree, node);

          //Creates temporary
          std::shared_ptr<array> tmp;
          switch(it->first){
            case REDUCE_1D:           tmp = std::shared_ptr<array>(new array(1, dtype, context));                                                        break;

            case ELEMENTWISE_1D:         tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case REDUCE_2D_ROWS:  tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], dtype, context));                              break;
            case REDUCE_2D_COLS:  tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[1], dtype, context));                              break;

            case ELEMENTWISE_2D:         tmp = std::shared_ptr<array>(new array(lmost.lhs.array->shape()[0], lmost.lhs.array->shape()[1], dtype, context)); break;
            case MATRIX_PRODUCT_NN:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[1], dtype, context));   break;
            case MATRIX_PRODUCT_NT:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[0], node.rhs.array->shape()[0], dtype, context));   break;
            case MATRIX_PRODUCT_TN:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[1], dtype, context));   break;
            case MATRIX_PRODUCT_TT:   tmp = std::shared_ptr<array>(new array(node.lhs.array->shape()[1], node.rhs.array->shape()[0], dtype, context));   break;

            default: throw std::invalid_argument("Unrecognized operation");
          }
          temporaries_.push_back(tmp);

          tree[rootidx].op.type = ASSIGN_TYPE;
          fill(tree[rootidx].lhs, (array&)*tmp);
          tree[rootidx].rhs = *it->second;
          tree[rootidx].rhs.subtype = it->second->subtype;

          //Execute
          profile->execute(execution_handler(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
          tree[rootidx] = root_save;

          //Incorporates the temporary within, the expression_tree
          fill(*it->second, (array&)*tmp);
        }
    }

    /*-----Compute final expression-----*/
    profiles[std::make_pair(final_type, dtype)]->execute(execution_handler(expression, c.execution_options(), c.dispatcher_options(), c.compilation_options()));
  }

  void execute(execution_handler const & c)
  {
    execute(c, isaac::profiles::get(c.execution_options().queue(c.x().context())));
  }

}
