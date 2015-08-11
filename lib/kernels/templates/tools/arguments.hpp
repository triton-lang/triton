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
inline std::string generate_arguments(std::string const & data_type, driver::Device const & device, std::vector<mapping_type> const & mappings, expressions_tuple const & expressions)
{
    std::string kwglobal = Global(device.backend()).get();
    std::string _size_t = size_type(device);

    kernel_generation_stream stream;

    process(stream, PARENT_NODE_TYPE, { {"array0", kwglobal + " #scalartype* #pointer, " + _size_t + " #start,"},
                                        {"host_scalar", "#scalartype #name,"},
                                        {"array1", kwglobal + " " + data_type + "* #pointer, " + _size_t + " #start, " + _size_t + " #stride,"},
                                        {"array2", kwglobal + " " + data_type + "* #pointer, " + _size_t + " #ld, " + _size_t + " #start, " + _size_t + " #stride, "},
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

    void set_arguments(array const * a) const
    {
        bool is_bound = binder_.bind(a->data());
        if (is_bound)
        {
            kernel_.setArg(current_arg_++, a->data());
            //scalar
            if(a->shape()[0]==1 && a->shape()[1]==1)
            {
                kernel_.setSizeArg(current_arg_++, a->start()[0]);
            }
            //array
            else if(a->shape()[0]==1 || a->shape()[1]==1)
            {
                kernel_.setSizeArg(current_arg_++, std::max(a->start()[0], a->start()[1]));
                kernel_.setSizeArg(current_arg_++, std::max(a->stride()[0], a->stride()[1]));
            }
            else
            {
                kernel_.setSizeArg(current_arg_++, a->ld()*a->stride()[1]);
                kernel_.setSizeArg(current_arg_++, a->start()[0] + a->start()[1]*a->ld());
                kernel_.setSizeArg(current_arg_++, a->stride()[0]);
            }
        }
    }

    void set_arguments(repeat_infos const & i) const
    {
        kernel_.setSizeArg(current_arg_++, i.sub1);
        kernel_.setSizeArg(current_arg_++, i.sub2);
        kernel_.setSizeArg(current_arg_++, i.rep1);
        kernel_.setSizeArg(current_arg_++, i.rep2);
    }


    void set_arguments(lhs_rhs_element const & lhs_rhs) const
    {
        switch(lhs_rhs.type_family)
        {
        case VALUE_TYPE_FAMILY: return set_arguments(lhs_rhs.dtype, lhs_rhs.vscalar);
        case ARRAY_TYPE_FAMILY: return set_arguments(lhs_rhs.array);
        case INFOS_TYPE_FAMILY: return set_arguments(lhs_rhs.tuple);
        default: throw std::runtime_error("Unrecognized type family");
        }
    }

    void operator()(isaac::array_expression const & array_expression, int_t root_idx, leaf_t leaf_t) const
    {
        array_expression::node const & root_node = array_expression.tree()[root_idx];
        if (leaf_t==LHS_NODE_TYPE && root_node.lhs.type_family != COMPOSITE_OPERATOR_FAMILY)
            set_arguments(root_node.lhs);
        else if (leaf_t==RHS_NODE_TYPE && root_node.rhs.type_family != COMPOSITE_OPERATOR_FAMILY)
            set_arguments(root_node.rhs);
    }


private:
    symbolic_binder & binder_;
    unsigned int & current_arg_;
    driver::Kernel & kernel_;
};

inline void set_arguments(expressions_tuple const & expressions, driver::Kernel & kernel, unsigned int & current_arg, binding_policy_t binding_policy)
{
    std::unique_ptr<symbolic_binder> binder;
    if (binding_policy==BIND_TO_HANDLE)
        binder.reset(new bind_to_handle());
    else
        binder.reset(new bind_all_unique());
    for (const auto & elem : expressions.data())
        traverse(*elem, (elem)->root(), set_arguments_functor(*binder, current_arg, kernel), true);
}

}
}
