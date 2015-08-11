#include <set>
#include <string>

#include <iostream>

#include "isaac/kernels/mapped_object.h"
#include "isaac/kernels/parse.h"
#include "isaac/kernels/stream.h"
#include "isaac/symbolic/expression.h"

#include <string>
#include "find_and_replace.hpp"
#include "to_string.hpp"

namespace isaac
{

void mapped_object::preprocess(std::string &) const { }

void mapped_object::postprocess(std::string &) const { }

void mapped_object::replace_macro(std::string & str, std::string const & macro, MorphBase const & morph)
{
  size_t pos = 0;
  while ((pos=str.find(macro, pos))!=std::string::npos)
  {
    std::string postprocessed;
    size_t pos_po = str.find('{', pos);
    size_t pos_pe = str.find('}', pos_po);

    size_t pos_comma = str.find(',', pos_po);
    if(pos_comma > pos_pe)
    {
        std::string i = str.substr(pos_po + 1, pos_pe - pos_po - 1);
        postprocessed = morph(i);
    }
    else
    {
        std::string i = str.substr(pos_po + 1, pos_comma - pos_po - 1);
        std::string j = str.substr(pos_comma + 1, pos_pe - pos_comma - 1);
        postprocessed = morph(i, j);
    }
    str.replace(pos, pos_pe + 1 - pos, postprocessed);
    pos = pos_pe;
  }
}

void mapped_object::register_attribute(std::string & attribute, std::string const & key, std::string const & value)
{
  attribute = value;
  keywords_[key] = attribute;
}

mapped_object::node_info::node_info(mapping_type const * _mapping, isaac::array_expression const * _array_expression, int_t _root_idx) :
    mapping(_mapping), array_expression(_array_expression), root_idx(_root_idx) { }

mapped_object::mapped_object(std::string const & scalartype, unsigned int id, std::string const & type_key) : type_key_(type_key)
{
  register_attribute(scalartype_, "#scalartype", scalartype);
  register_attribute(name_, "#name", "obj" + tools::to_string(id));
}

mapped_object::~mapped_object()
{ }

std::string mapped_object::type_key() const
{ return type_key_; }

std::string const & mapped_object::name() const
{ return name_; }

std::map<std::string, std::string> const & mapped_object::keywords() const
{ return keywords_; }

std::string mapped_object::process(std::string const & in) const
{
  std::string res(in);
  preprocess(res);
  for (const auto & elem : keywords_)
    tools::find_and_replace(res, elem.first, elem.second);
  postprocess(res);
  return res;
}

std::string mapped_object::evaluate(std::map<std::string, std::string> const & accessors) const
{
  if (accessors.find(type_key_)==accessors.end())
    return name_;
  return process(accessors.at(type_key_));
}


binary_leaf::binary_leaf(mapped_object::node_info info) : info_(info){ }

void binary_leaf::process_recursive(kernel_generation_stream & stream, leaf_t leaf, std::map<std::string, std::string> const & accessors)
{
  std::set<std::string> already_fetched;
  process(stream, leaf, accessors, *info_.array_expression, info_.root_idx, *info_.mapping, already_fetched);
}

std::string binary_leaf::evaluate_recursive(leaf_t leaf, std::map<std::string, std::string> const & accessors)
{
  return evaluate(leaf, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}


mapped_gemm::mapped_gemm(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "gemm"), binary_leaf(info) { }

//
mapped_dot::mapped_dot(std::string const & scalartype, unsigned int id, node_info info, std::string const & type_key) :
  mapped_object(scalartype, id, type_key), binary_leaf(info)
{ }

int_t mapped_dot::root_idx() const
{ return info_.root_idx; }

isaac::array_expression const & mapped_dot::array_expression() const
{ return *info_.array_expression; }

array_expression::node mapped_dot::root_node() const
{ return array_expression().tree()[root_idx()]; }

bool mapped_dot::is_index_dot() const
{
  op_element const & op = root_op();
  return op.type==OPERATOR_ELEMENT_ARGFMAX_TYPE
      || op.type==OPERATOR_ELEMENT_ARGMAX_TYPE
      || op.type==OPERATOR_ELEMENT_ARGFMIN_TYPE
      || op.type==OPERATOR_ELEMENT_ARGMIN_TYPE;
}

op_element mapped_dot::root_op() const
{
    return info_.array_expression->tree()[info_.root_idx].op;
}


//
mapped_scalar_dot::mapped_scalar_dot(std::string const & scalartype, unsigned int id, node_info info) : mapped_dot(scalartype, id, info, "scalar_dot"){ }

//
mapped_gemv::mapped_gemv(std::string const & scalartype, unsigned int id, node_info info) : mapped_dot(scalartype, id, info, "gemv") { }

//
void mapped_host_scalar::preprocess(std::string & str) const
{
  struct Morph : public MorphBase
  {
    std::string operator()(std::string const &) const
    { return "#name"; }

    std::string operator()(std::string const &, std::string const &) const
    { return "#name"; }
  };
  replace_macro(str, "$VALUE", Morph());
}


mapped_host_scalar::mapped_host_scalar(std::string const & scalartype, unsigned int id) : mapped_object(scalartype, id, "host_scalar"){ }

//
mapped_tuple::mapped_tuple(std::string const & scalartype, unsigned int id, size_t size) : mapped_object(scalartype, id, "tuple"+tools::to_string(size)), size_(size), names_(size)
{
  for(size_t i = 0 ; i < size_ ; ++i)
    register_attribute(names_[i], "#tuplearg"+tools::to_string(i), name_ + tools::to_string(i));
}

//
mapped_handle::mapped_handle(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_object(scalartype, id, type_key)
{ register_attribute(pointer_, "#pointer", name_ + "_pointer"); }

//
mapped_buffer::mapped_buffer(std::string const & scalartype, unsigned int id, std::string const & type_key) : mapped_handle(scalartype, id, type_key){ }

//
void mapped_array::preprocess(std::string & str) const
{

  struct MorphValue : public MorphBase
  {
    MorphValue(std::string const & _ld) : ld(_ld){ }

    std::string operator()(std::string const & i) const
    { return "#pointer[" + i + "]"; }

    std::string operator()(std::string const & i, std::string const & j) const
    { return "#pointer[(" + i + ") +  (" + j + ") * " + ld + "]"; }
  private:
    std::string const & ld;
  };

  struct MorphOffset : public MorphBase
  {
    MorphOffset(std::string const & _ld) : ld(_ld){ }

    std::string operator()(std::string const & i) const
    { return i; }

    std::string operator()(std::string const & i, std::string const & j) const
    {return "(" + i + ") +  (" + j + ") * " + ld; }
  private:
    std::string const & ld;
  };

  replace_macro(str, "$VALUE", MorphValue(ld_));
  replace_macro(str, "$OFFSET", MorphOffset(ld_));
}

mapped_array::mapped_array(std::string const & scalartype, unsigned int id, char type) : mapped_buffer(scalartype, id, type=='s'?"array0":(type=='m'?"array2":"array1")), type_(type)
{
  if(type_ == 's')
  {
    register_attribute(start_, "#start", name_ + "_start");
  }
  else if(type_=='m')
  {
    register_attribute(start_, "#start", name_ + "_start");
    register_attribute(stride_, "#stride", name_ + "_stride");
    register_attribute(ld_, "#ld", name_ + "_ld");
  }
  else
  {
    register_attribute(start_, "#start", name_ + "_start");
    register_attribute(stride_, "#stride", name_ + "_stride");
  }

}

//
void mapped_vdiag::postprocess(std::string &res) const
{
  std::map<std::string, std::string> accessors;
  tools::find_and_replace(res, "#diag_offset", isaac::evaluate(RHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping));
  accessors["array1"] = res;
  accessors["host_scalar"] = res;
  res = isaac::evaluate(LHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}

mapped_vdiag::mapped_vdiag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "vdiag"), binary_leaf(info){}


//
void mapped_matrix_row::postprocess(std::string &res) const
{
  std::map<std::string, std::string> accessors;
  tools::find_and_replace(res, "#row", isaac::evaluate(RHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping));
  accessors["array2"] = res;
  res = isaac::evaluate(LHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}

mapped_matrix_row::mapped_matrix_row(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_row"), binary_leaf(info)
{ }

//
void mapped_matrix_column::postprocess(std::string &res) const
{
  std::map<std::string, std::string> accessors;
  tools::find_and_replace(res, "#column", isaac::evaluate(RHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping));
  accessors["array2"] = res;
  res = isaac::evaluate(LHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}

mapped_matrix_column::mapped_matrix_column(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_column"), binary_leaf(info)
{ }

//
char mapped_repeat::get_type(node_info const & info)
{
  repeat_infos const & infos = info.array_expression->tree()[info.root_idx].rhs.tuple;
  if(infos.sub1>1 && infos.sub2==1)
    return 'c';
  else if(infos.sub1==1 && infos.sub2>1)
    return 'r';
  else
    return 'm';
}

void mapped_repeat::postprocess(std::string &res) const
{
  std::map<std::string, std::string> accessors;
  mapped_object& args = *(info_.mapping->at(std::make_pair(info_.root_idx,RHS_NODE_TYPE)));
  tools::find_and_replace(res, "#tuplearg0", args.process("#tuplearg0"));
  tools::find_and_replace(res, "#tuplearg1", args.process("#tuplearg1"));
  tools::find_and_replace(res, "#tuplearg2", args.process("#tuplearg2"));
  tools::find_and_replace(res, "#tuplearg3", args.process("#tuplearg3"));

  struct MorphValue : public MorphBase
  {
    MorphValue(char _type): type(_type){ }

    std::string operator()(std::string const &) const { return "";}

    std::string operator()(std::string const & i, std::string const & j) const
    {
      if(type=='c') return "$VALUE{" + i + "}";
      else if(type=='r') return "$VALUE{" + j + "}";
      else return "$VALUE{" + i + "," + j + "}";
    }
  private:
    char type;
  };

  replace_macro(res, "$VALUE", MorphValue(type_));
  accessors["array1"] = res;
  accessors["array2"] = res;
  res = isaac::evaluate(LHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}

mapped_repeat::mapped_repeat(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "repeat"), binary_leaf(info), type_(get_type(info))
{
}


//
void mapped_matrix_diag::postprocess(std::string &res) const
{
  std::map<std::string, std::string> accessors;
  tools::find_and_replace(res, "#diag_offset", isaac::evaluate(RHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping));
  accessors["array2"] = res;
  res = isaac::evaluate(LHS_NODE_TYPE, accessors, *info_.array_expression, info_.root_idx, *info_.mapping);
}

mapped_matrix_diag::mapped_matrix_diag(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "matrix_diag"), binary_leaf(info)
{ }

//
void mapped_outer::postprocess(std::string &res) const
{
    struct Morph : public MorphBase
    {
      Morph(leaf_t const & leaf, node_info const & i) : leaf_(leaf), i_(i){}
      std::string operator()(std::string const & i) const
      {
        std::map<std::string, std::string> accessors;
        accessors["array1"] = "$VALUE{"+i+"}";
        return isaac::evaluate(leaf_, accessors, *i_.array_expression, i_.root_idx, *i_.mapping);
      }
      std::string operator()(std::string const &, std::string const &) const{return "";}
    private:
      leaf_t leaf_;
      node_info i_;
    };

    replace_macro(res, "$LVALUE", Morph(LHS_NODE_TYPE, info_));
    replace_macro(res, "$RVALUE", Morph(RHS_NODE_TYPE, info_));
}

mapped_outer::mapped_outer(std::string const & scalartype, unsigned int id, node_info info) : mapped_object(scalartype, id, "outer"), binary_leaf(info)
{ }

std::string mapped_cast::operator_to_str(operation_node_type type)
{
  switch(type)
  {
    case OPERATOR_CAST_BOOL_TYPE : return "bool";
    case OPERATOR_CAST_CHAR_TYPE : return "char";
    case OPERATOR_CAST_UCHAR_TYPE : return "uchar";
    case OPERATOR_CAST_SHORT_TYPE : return "short";
    case OPERATOR_CAST_USHORT_TYPE : return "ushort";
    case OPERATOR_CAST_INT_TYPE : return "int";
    case OPERATOR_CAST_UINT_TYPE : return "uint";
    case OPERATOR_CAST_LONG_TYPE : return "long";
    case OPERATOR_CAST_ULONG_TYPE : return "ulong";
    case OPERATOR_CAST_HALF_TYPE : return "half";
    case OPERATOR_CAST_FLOAT_TYPE : return "float";
    case OPERATOR_CAST_DOUBLE_TYPE : return "double";
    default : return "invalid";
  }
}

mapped_cast::mapped_cast(operation_node_type type, unsigned int id) : mapped_object(operator_to_str(type), id, "cast")
{ }


}
