#ifndef ATIDLAS_MODEL_MODEL_H
#define ATIDLAS_MODEL_MODEL_H

#include <string>
#include <vector>
#include <map>

#include "atidlas/backend/templates/base.h"
#include "atidlas/cl_ext/compare.hpp"
#include "atidlas/cl_ext/lazy_compiler.h"
#include "atidlas/model/predictors/random_forest.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

  class model
  {
    typedef std::vector< std::shared_ptr<base> > templates_container;
  public:
    struct runtime_options
    {
      runtime_options() : label(-1), recompile(false){}
      runtime_options(std::string const & p) : program_name(p), label(-1), recompile(false){}

      std::string program_name;
      int label;
      bool recompile;
    };

  private:
    std::string define_extension(std::string const & extensions, std::string const & ext);
    inline void fill_program_name(char* program_name, expressions_tuple const & expressions, binding_policy_t binding_policy);
    std::vector<cl_ext::lazy_compiler>& init(expressions_tuple const & expressions, runtime_options const & opt = runtime_options());

  public:
    model(predictors::random_forest const &, std::vector< std::shared_ptr<base> > const &, cl::CommandQueue &);
    model(std::vector< std::shared_ptr<base> > const &, cl::CommandQueue &);
    model(base const &, cl::CommandQueue &);

    void execute(expressions_tuple const &, operation_cache * cache = NULL, runtime_options const & opt = runtime_options());
    void tune(expressions_tuple const &);

    templates_container const & templates() const;
  private:
    templates_container templates_;
    std::shared_ptr<predictors::random_forest> predictor_;
    std::map<std::vector<int_t>, int> hardcoded_;
    std::map<cl_context, std::map<std::string, std::vector<cl_ext::lazy_compiler> > > lazy_programs_;
    cl::CommandQueue & queue_;
  };

  typedef std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<model> > model_map_t;

  model_map_t init_models(cl::CommandQueue const & queue);
  model_map_t& get_model_map(cl::CommandQueue & queue);
  model& get_model(cl::CommandQueue & queue, expression_type, numeric_type);

  extern std::map<cl::CommandQueue, model_map_t, cl_ext::compare> models;

}

#endif
