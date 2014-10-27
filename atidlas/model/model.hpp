#ifndef ATIDLAS_MODEL_MODEL_HPP
#define ATIDLAS_MODEL_MODEL_HPP

#include "rapidjson/document.h"

#include "viennacl/ocl/program.hpp"

#include "atidlas/model/tools.hpp"
#include "atidlas/tools/shared_ptr.hpp"
#include "atidlas/tools/lazy_program_compiler.hpp"
#include "atidlas/templates/template_base.hpp"

namespace atidlas
{

  class random_forest
  {
  public:
    class tree
    {
    public:
      tree(rapidjson::Value const & treerep)
      {
        children_left_ = tools::to_int_array<int>(treerep["children_left"]);
        children_right_ = tools::to_int_array<int>(treerep["children_right"]);
        threshold_ = tools::to_float_array<float>(treerep["threshold"]);
        feature_ = tools::to_float_array<float>(treerep["feature"]);
        for(rapidjson::SizeType i = 0 ; i < treerep["value"].Size() ; i++)
          value_.push_back(tools::to_float_array<float>(treerep["value"][i]));
        D_ = value_[0].size();
      }

      std::vector<float> const & predict(std::vector<atidlas_int_t> const & x) const
      {
        atidlas_int_t idx = 0;
        while(children_left_[idx]!=-1)
        {
          if(x[feature_[idx]] <= threshold_[idx])
            idx = children_left_[idx];
          else
            idx = children_right_[idx];
        }
        return value_[idx];
      }

      atidlas_int_t D() const { return D_; }

    private:
      std::vector<int> children_left_;
      std::vector<int> children_right_;
      std::vector<float> threshold_;
      std::vector<float> feature_;
      std::vector<std::vector<float> > value_;
      atidlas_int_t D_;
    };

    random_forest(rapidjson::Value const & estimators)
    {
      for(rapidjson::SizeType i = 0 ; i < estimators.Size() ; ++i)
        estimators_.push_back(tree(estimators[i]));
    }

    std::vector<float> predict(std::vector<atidlas_int_t> const & x) const
    {
      atidlas_int_t D = estimators_.front().D();
      std::vector<float> res(D, 0);
      for(std::vector<tree>::const_iterator it = estimators_.begin() ; it != estimators_.end() ; ++it)
      {
        std::vector<float> const & subres = it->predict(x);
        for(atidlas_int_t i = 0 ; i < D ; ++i)
          res[i] += subres[i];
      }
      for(atidlas_int_t i = 0 ; i < D ; ++i)
        res[i] /= estimators_.size();
      return res;
    }

  private:
    std::vector<tree> estimators_;
  };

  class model
  {
    typedef std::vector< tools::shared_ptr<template_base> > templates_container;
  private:
    std::string define_extension(std::string const & ext)
    {
      // Note: On devices without double precision support, 'ext' is an empty string.
      return (ext.length() > 1) ? std::string("#pragma OPENCL EXTENSION " + ext + " : enable\n") : std::string("\n");
    }

    void init_program_compiler(std::string const & name, bool force_recompilation)
    {
      lazy_programs_.push_back(lazy_program_compiler(&context_, name, force_recompilation));
      lazy_programs_.back().add(define_extension(device_.double_support_extension()));
    }

  public:
    model(random_forest const & predictor, std::vector< tools::shared_ptr<template_base> > const & templates,
          viennacl::ocl::context & context, viennacl::ocl::device const & device) : predictor_(predictor), templates_(templates), context_(context), device_(device)
    {  }

    void execute(statements_container const & statements)
    {
      if(lazy_programs_.empty())
      {
        std::string pname = tools::statements_representation(statements, BIND_TO_HANDLE);

        init_program_compiler(pname, false);
        init_program_compiler(pname + "_fb", false);

        for(size_t i = 0 ; i < templates_.size() ; ++i)
        {
          std::vector<std::string> cur = templates_[i]->generate("k" + tools::to_string(i), statements, device_);
          for(size_t j = 0 ; j < cur.size() ; ++j)
            lazy_programs_[j].add(cur[j]);
        }
      }

      //Prediction
      std::vector<atidlas_int_t> x = templates_[0]->input_sizes(statements);
      std::vector<float> predictions = predictor_.predict(x);
      atidlas_int_t label = std::distance(predictions.begin(),std::min_element(predictions.begin(), predictions.end()));
      std::cout << label << std::endl;
      //Execution
      templates_[label]->enqueue("k" + tools::to_string(label), lazy_programs_, statements);
    }

  private:
    random_forest predictor_;

    templates_container templates_;

    viennacl::ocl::context & context_;
    viennacl::ocl::device const & device_;
    std::vector<lazy_program_compiler> lazy_programs_;
  };

}

#endif
