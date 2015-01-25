#ifndef ATIDLAS_MODEL_MODEL_H
#define ATIDLAS_MODEL_MODEL_H

#include <string>
#include <vector>
#include <map>

#include "atidlas/backend/templates/base.h"
#include "atidlas/cl/compare.hpp"
#include "atidlas/cl/lazy_compiler.h"
#include "atidlas/model/predictors/random_forest.h"
#include "atidlas/symbolic/expression.h"

namespace atidlas
{

  class model
  {
    typedef std::vector< tools::shared_ptr<base> > templates_container;

  private:
    std::string define_extension(std::string const & extensions, std::string const & ext);
    inline void fill_program_name(char* program_name, symbolic_expressions_container const & symbolic_expressions, binding_policy_t binding_policy);
    std::vector<cl::lazy_compiler>& init(symbolic_expressions_container const & symbolic_expressions, cl::Context const & context, cl::Device const & device, bool force_recompilation);

  public:
    model(predictors::random_forest const &, std::vector< tools::shared_ptr<base> > const &, cl::CommandQueue &);
    model(std::vector< tools::shared_ptr<base> > const &, cl::CommandQueue &);
    model(base const &, cl::CommandQueue &);

    void execute(symbolic_expressions_container const &, bool bypass_predictor = false, bool force_recompilation = false);
    void tune(symbolic_expressions_container const &);

    templates_container const & templates() const;
  private:
    templates_container templates_;
    tools::shared_ptr<predictors::random_forest> predictor_;
    std::map<std::vector<int_t>, int> hardcoded_;
    std::map<cl_context, std::map<std::string, std::vector<cl::lazy_compiler> > > lazy_programs_;
    cl::CommandQueue & queue_;
  };

  typedef std::map<std::pair<expression_type, numeric_type>, tools::shared_ptr<model> > model_map_t;

  model_map_t init_models(cl::CommandQueue const & queue);
  model_map_t& get_model_map(cl::CommandQueue & queue);
  model& get_model(cl::CommandQueue & queue, expression_type, numeric_type);

  extern std::map<cl::CommandQueue, model_map_t, cl::compare> models;

}

#endif
