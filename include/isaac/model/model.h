#ifndef ISAAC_MODEL_MODEL_H
#define ISAAC_MODEL_MODEL_H

#include <string>
#include <vector>
#include <map>

#include "isaac/backend/templates/base.h"
#include "isaac/model/predictors/random_forest.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

  class model
  {
    typedef tools::shared_ptr<templates::base> template_pointer;
    typedef std::vector< template_pointer > templates_container;

  private:
    std::string define_extension(std::string const & extensions, std::string const & ext);
    inline void fill_program_name(char* program_name, expressions_tuple const & expressions, binding_policy_t binding_policy);
    driver::Program& init(controller<expressions_tuple> const &);

  public:
    model(expression_type, numeric_type, predictors::random_forest const &, std::vector< tools::shared_ptr<templates::base> > const &, driver::CommandQueue const &);
    model(expression_type, numeric_type, templates::base const &, driver::CommandQueue const &);

    void execute(controller<expressions_tuple> const &);
    templates_container const & templates() const;

    void test() const
    { std::cout << queue_.device().backend() << std::endl;}

  private:
    templates_container templates_;
    template_pointer fallback_;
    tools::shared_ptr<predictors::random_forest> predictor_;
    std::map<std::vector<int_t>, int> hardcoded_;
    std::map<driver::Context, std::map<std::string, std::shared_ptr<driver::Program> > > programs_;
    driver::CommandQueue queue_;
  };

  typedef std::map<std::pair<expression_type, numeric_type>, tools::shared_ptr<model> > model_map_t;

  model_map_t init_models(driver::CommandQueue const & queue);
  model_map_t& models(driver::CommandQueue & queue);

  extern std::map<std::pair<expression_type, numeric_type>, tools::shared_ptr<templates::base> > fallbacks;
  extern std::map<driver::CommandQueue, model_map_t> models_;

}

#endif
