#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include "isaac/kernels/mapped_object.h"
#include "isaac/kernels/parse.h"
#include "isaac/array.h"

namespace isaac
{
namespace templates
{

//Generate
inline std::string generate_arguments(std::string const &, driver::Device const & device, mapping_type const & mappings, math_expression const & expressions)
{
    std::string kwglobal = Global(device.backend()).get();
    std::string _size_t = size_type(device);

    kernel_generation_stream stream;

    process(stream, PARENT_NODE_TYPE, {  {"array11", kwglobal + " #scalartype* #pointer, " + _size_t + " #start,"},
                                         {"array1", kwglobal + " #scalartype* #pointer, " + _size_t + " #start,"},
                                        {"host_scalar", "#scalartype #name,"},
                                        {"arrayn", kwglobal + " #scalartype* #pointer, " + _size_t + " #start, " + _size_t + " #stride,"},
                                        {"array1n", kwglobal + " #scalartype* #pointer, " + _size_t + " #start, " + _size_t + " #stride,"},
                                        {"arrayn1", kwglobal + " #scalartype* #pointer, " + _size_t + " #start, " + _size_t + " #stride,"},
                                        {"arraynn", kwglobal + " #scalartype* #pointer, " + _size_t + " #start, " + _size_t + " #stride," +  _size_t + " #ld,"},
                                        {"tuple4", "#scalartype #name0, #scalartype #name1, #scalartype #name2, #scalartype #name3,"}}
            , expressions, mappings);

    std::string res = stream.str();
    res.erase(res.rfind(','));
    return res;
}

//Enqueue
class set_arguments_functor : public traversal_functor
{
public:
    typedef void result_type;

    set_arguments_functor(symbolic_binder & binder, unsigned int & current_arg, driver::Kernel & kernel)
        : binder_(binder), current_arg_(current_arg), kernel_(kernel)
    {
    }

    void set_arguments(numeric_type dtype, values_holder const & scal) const
    {
        switch(dtype)
        {
        //    case BOOL_TYPE: kernel_.setArg(current_arg_++, scal.bool8); break;
        case CHAR_TYPE: kernel_.setArg(current_arg_++, scal.int8); break;
        case UCHAR_TYPE: kernel_.setArg(current_arg_++, scal.uint8); break;
        case SHORT_TYPE: kernel_.setArg(current_arg_++, scal.int16); break;
        case USHORT_TYPE: kernel_.setArg(current_arg_++, scal.uint16); break;
        case INT_TYPE: kernel_.setArg(current_arg_++, scal.int32); break;
        case UINT_TYPE: kernel_.setArg(current_arg_++, scal.uint32); break;
        case LONG_TYPE: kernel_.setArg(current_arg_++, scal.int64); break;
        case ULONG_TYPE: kernel_.setArg(current_arg_++, scal.uint64); break;
            //    case HALF_TYPE: kernel_.setArg(current_arg_++, scal.float16); break;
        case FLOAT_TYPE: kernel_.setArg(current_arg_++, scal.float32); break;
        case DOUBLE_TYPE: kernel_.setArg(current_arg_++, scal.float64); break;
        default: throw unknown_datatype(dtype);
        }
    }

    void set_arguments(array_base const * a, bool is_assigned) const
    {
        bool is_bound = binder_.bind(a, is_assigned);
        if (is_bound)
        {
            kernel_.setArg(current_arg_++, a->data());
            kernel_.setSizeArg(current_arg_++, a->start());
            for(int_t i = 0 ; i < a->dim() ; i++)
              if(a->shape()[i] > 1)
                kernel_.setSizeArg(current_arg_++, a->stride()[i]);
        }
    }

    void set_arguments(lhs_rhs_element const & lhs_rhs, bool is_assigned) const
    {
        switch(lhs_rhs.type_family)
        {
        case VALUE_TYPE_FAMILY: return set_arguments(lhs_rhs.dtype, lhs_rhs.vscalar);
        case ARRAY_TYPE_FAMILY: return set_arguments(lhs_rhs.array, is_assigned);
        case PLACEHOLDER_TYPE_FAMILY: return;
        default: throw std::runtime_error("Unrecognized type family");
        }
    }

    void operator()(isaac::math_expression const & math_expression, size_t root_idx, leaf_t leaf_t) const
    {
        math_expression::node const & root_node = math_expression.tree()[root_idx];
        if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
            set_arguments(root_node.lhs, detail::is_assignment(root_node.op));
        else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
            set_arguments(root_node.rhs, false);
    }


private:
    symbolic_binder & binder_;
    unsigned int & current_arg_;
    driver::Kernel & kernel_;
};

inline void set_arguments(math_expression const & expression, driver::Kernel & kernel, unsigned int & current_arg, binding_policy_t binding_policy)
{
    std::unique_ptr<symbolic_binder> binder;
    if (binding_policy==BIND_SEQUENTIAL)
        binder.reset(new bind_sequential());
    else
        binder.reset(new bind_independent());
    traverse(expression, expression.root(), set_arguments_functor(*binder, current_arg, kernel), true);

}

}
}
