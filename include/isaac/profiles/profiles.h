#ifndef ISAAC_MODEL_DATABASE_H
#define ISAAC_MODEL_DATABASE_H

#include <map>
#include <memory>

#include "isaac/driver/command_queue.h"
#include "isaac/driver/device.h"
#include "isaac/common/expression_type.h"
#include "isaac/common/numeric_type.h"
#include "isaac/kernels/templates/base.h"
#include "isaac/profiles/predictors/random_forest.h"
#include "isaac/symbolic/expression.h"

namespace isaac
{

struct profiles
{
    typedef std::map<std::tuple<driver::Device::Type, driver::Device::Vendor, driver::Device::Architecture> , const char *> presets_type;
public:
    class value_type
    {
      typedef std::shared_ptr<templates::base> template_pointer;
      typedef std::vector< template_pointer > templates_container;

    private:
      std::string define_extension(std::string const & extensions, std::string const & ext);
      inline void fill_program_name(char* program_name, expressions_tuple const & expressions, binding_policy_t binding_policy);
      driver::Program const & init(controller<expressions_tuple> const &);

    public:
      value_type(expression_type, numeric_type, predictors::random_forest const &, std::vector< std::shared_ptr<templates::base> > const &, driver::CommandQueue const &);
      value_type(expression_type, numeric_type, templates::base const &, driver::CommandQueue const &);

      void execute(controller<expressions_tuple> const &);
      templates_container const & templates() const;

    private:
      templates_container templates_;
      template_pointer fallback_;
      std::shared_ptr<predictors::random_forest> predictor_;
      std::map<std::vector<int_t>, int> hardcoded_;
      driver::CommandQueue queue_;
      driver::ProgramCache & cache_;
    };

    typedef std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<value_type> > map_type;
private:
    static std::shared_ptr<templates::base> create(std::string const & template_name, std::vector<int> const & x);
    static void import(std::string const & fname, driver::CommandQueue const & queue);
    static map_type & init(driver::CommandQueue const & queue);
public:
    static map_type & get(driver::CommandQueue const & queue);
    static void set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<value_type> const & profile);
private:
    static const presets_type presets_;
    static std::map<driver::CommandQueue, map_type> cache_;
};

extern std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<templates::base> > fallbacks;

}

#endif
