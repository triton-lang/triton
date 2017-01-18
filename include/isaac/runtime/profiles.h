/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
#include "isaac/runtime/predictors/random_forest.h"
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
      typedef std::vector<template_pointer> templates_container;

    private:
      std::string define_extension(std::string const & extensions, std::string const & ext);
      driver::Program const & init(runtime::execution_handler const &);

    public:
      value_type(expression_type, numeric_type, predictors::random_forest const &, std::vector< std::shared_ptr<templates::base> > const &, driver::CommandQueue const &);
      value_type(numeric_type, std::shared_ptr<templates::base> const &, driver::CommandQueue const &);
      void execute(runtime::execution_handler const &);
      templates_container const & templates() const;

    private:
      templates_container templates_;
      std::shared_ptr<predictors::random_forest> predictor_;
      std::map<std::vector<int_t>, int> labels_;
      driver::CommandQueue queue_;
      driver::ProgramCache & cache_;
    };

    typedef std::map<std::pair<expression_type, numeric_type>, std::shared_ptr<value_type> > map_type;
private:
    static std::shared_ptr<templates::base> create(std::string const & template_name, std::vector<int> const & x);
    static std::shared_ptr<templates::base> create(std::string const & op, std::string const & x);
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

}
}

#endif
