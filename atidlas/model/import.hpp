#ifndef ATIDLAS_MODEL_IMPORT_HPP
#define ATIDLAS_MODEL_IMPORT_HPP

#include "atidlas/templates/vector_axpy.hpp"
#include "atidlas/templates/reduction.hpp"
#include "atidlas/templates/matrix_axpy.hpp"
#include "atidlas/templates/row_wise_reduction.hpp"
#include "atidlas/templates/matrix_product.hpp"

#include "atidlas/model/tools.hpp"
#include "atidlas/model/rapidjson/document.h"
#include "atidlas/model/model.hpp"

namespace atidlas
{

  namespace detail
  {
    tools::shared_ptr<template_base> create(std::string const & template_name, std::vector<int> const & a)
    {
      fetching_policy_type fetch[] = {FETCH_FROM_LOCAL, FETCH_FROM_GLOBAL_CONTIGUOUS, FETCH_FROM_GLOBAL_STRIDED};
      if(template_name=="vector-axpy")
        return tools::shared_ptr<template_base>(new vector_axpy_template( vector_axpy_parameters(a[0], a[1], a[2], fetch[a[3]])));
      if(template_name=="reduction")
        return tools::shared_ptr<template_base>(new reduction_template( reduction_parameters(a[0], a[1], a[2], fetch[a[3]])));
      if(template_name=="matrix-axpy")
        return tools::shared_ptr<template_base>(new matrix_axpy_template( matrix_axpy_parameters(a[0], a[1], a[2], a[3], a[4], fetch[a[5]])));
      if(template_name.find("row-wise-reduction")!=std::string::npos)
        return tools::shared_ptr<template_base>(new row_wise_reduction_template( row_wise_reduction_parameters(a[0], a[1], a[2], a[3], fetch[a[5]])));
      if(template_name.find("matrix-product")!=std::string::npos)
      {
        char A_trans = template_name[15];
        char B_trans = template_name[16];
        return tools::shared_ptr<template_base>(new matrix_product_template( matrix_product_parameters(a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                                                                                  fetch[a[7]], fetch[a[8]], a[9], a[10]), A_trans, B_trans));
      }
      else
        throw generator_not_supported_exception("The provided operation is not supported");
    }
  }

  std::map<std::string, tools::shared_ptr<model> > import(std::string const & fname)
  {
    namespace js = rapidjson;

    std::map<std::string, tools::shared_ptr<model> > result;

    viennacl::ocl::context & context = viennacl::ocl::current_context();
    viennacl::ocl::device const & device = viennacl::ocl::current_device();

    //Parse the JSON document
    js::Document document;
    std::ifstream t(fname.c_str());
    std::string str;
    t.seekg(0, std::ios::end);
    str.reserve(t.tellg());
    t.seekg(0, std::ios::beg);
    str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    document.Parse<0>(str.c_str());

    //Deserialize
    std::vector<std::string> operations = tools::make_vector<std::string>() << "vector-axpy" << "reduction"
                                                                            << "matrix-axpy" << "row-wise-reduction" << "matrix-product";
    std::vector<std::string> dtype = tools::make_vector<std::string>() << "float32" << "float64";
    for(std::vector<std::string>::iterator op = operations.begin() ; op != operations.end() ; ++op)
    {
      const char * opcstr = op->c_str();
      if(document.HasMember(opcstr))
        for(std::vector<std::string>::iterator dt = dtype.begin() ; dt != dtype.end() ; ++dt)
        {
          const char * dtcstr = dt->c_str();
          if(document[opcstr].HasMember(dtcstr))
          {
            // Get profiles
            std::vector<tools::shared_ptr<template_base> > templates;
            js::Value const & profiles = document[opcstr][dtcstr]["profiles"];
            for (js::SizeType id = 0 ; id < profiles.Size() ; ++id)
              templates.push_back(detail::create(*op, tools::to_int_array<int>(profiles[id])));
            // Get predictor
            random_forest predictor(document[opcstr][dtcstr]["predictor"]);

            result[*op + "-" + *dt] = tools::shared_ptr<model>(new model(predictor, templates, context, device));
          }
        }
    }


    return result;
  }

}

#endif
