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

#ifndef TDL_INCLUDE_DRIVER_BACKEND_H
#define TDL_INCLUDE_DRIVER_BACKEND_H

#include <map>
#include <list>
#include <vector>
#include "triton/driver/context.h"


namespace triton
{
namespace driver
{

class buffer;
class stream;
class device;
class context;
class platform;
class module;
class kernel;

struct backend
{

  // platforms
  class platforms
  {
    friend class backend;
  private:
    static void init();

  public:
    static void get(std::vector<driver::platform*> &results);

  private:
    static std::vector<driver::platform*> cache_;
  };

  // devices
  class devices
  {
    friend class backend;

  private:
    static void init(const std::vector<platform *> &platforms);

  public:
    static void get(std::vector<driver::device*>& devs);

  private:
    static std::vector<driver::device*> cache_;
  };

  // modules
  class modules
  {
    friend class backend;

  public:
    static void release();
    static driver::module* get(driver::stream* stream, std::string const & name, std::string const &src);

  private:
    static std::map<std::tuple<driver::stream*, std::string>, driver::module*> cache_;
  };

  // kernels
  class kernels
  {
    friend class backend;
  public:
    static void release();
    static driver::kernel* get(driver::module* mod, const std::string & name);
  private:
    static std::map<std::tuple<module*, std::string>, driver::kernel*> cache_;
  };

  // contexts
  class contexts
  {
    friend class backend;
  private:
    static void init(const std::vector<device *> &);
    static void release();
  public:
    static driver::context* get_default();

    static driver::context* import(CUcontext ctx)
    {
      for(driver::context* x: cache_){
        driver::cu_context* cu_x = (driver::cu_context*)x;
        if(*cu_x->cu()==ctx)
          return x;
      }
      cache_.emplace_back(new driver::cu_context(ctx, false));
      return cache_.back();
    }

    static void get(std::list<driver::context*> &);

  private:
    static std::list<driver::context*> cache_;
  };

  // streams
  class streams
  {
    friend class backend;
  private:
    static void init(std::list<context*> const &);
    static void release();
  public:
    static void get(driver::context*, std::vector<driver::stream *> &streams);
    static driver::stream* get(driver::context*, unsigned int id = 0);
    static driver::stream* get_default();
  private:
    static std::map<driver::context*, std::vector<driver::stream*> > cache_;
  };

  static void init();
  static void release();
  static void synchronize(triton::driver::context *);

  static unsigned int default_device;
};

}
}

#endif
