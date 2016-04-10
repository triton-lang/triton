/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#ifndef ISAAC_MODEL_DATABASE_H
#define ISAAC_MODEL_DATABASE_H

#include <map>
#include <memory>

#include "isaac/driver/command_queue.h"
#include "isaac/driver/device.h"
#include "isaac/common/expression_type.h"
#include "isaac/common/numeric_type.h"
#include "isaac/jit/generation/base.h"
#include "isaac/runtime/inference/predictors/random_forest.h"
#include "isaac/jit/syntax/expression/expression.h"

namespace isaac
{
namespace runtime
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
      driver::Program const & init(runtime::execution_handler const &);

    public:
      value_type(expression_type, numeric_type, predictors::random_forest const &, std::vector< std::shared_ptr<templates::base> > const &, driver::CommandQueue const &);
      value_type(expression_type, numeric_type, templates::base const &, driver::CommandQueue const &);
      void execute(runtime::execution_handler const &);
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
    static void release();
    static map_type & get(driver::CommandQueue const & queue);
    static void set(driver::CommandQueue const & queue, expression_type operation, numeric_type dtype, std::shared_ptr<value_type> const & profile);
private:
    static const presets_type presets_;
    static std::map<driver::CommandQueue, map_type> cache_;
};

extern std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<templates::base> > fallbacks;

}
}

#endif
