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

  class modules
  {
    friend class backend;
  public:
    static void release();
    static module& get(driver::stream const & stream, std::string const & name, std::string const &src);
  private:
    static std::map<std::tuple<stream, std::string>, module * > cache_;
  };

  class kernels
  {
    friend class backend;
  public:
    static void release();
    static kernel & get(driver::module const & program, std::string const & name);
  private:
    static std::map<std::tuple<module, std::string>, kernel * > cache_;
  };

  class contexts
  {
    friend class backend;
  private:
    static void init(std::vector<platform> const &);
    static void release();
  public:
    static driver::context const & get_default();
    template<class T>
    static driver::context const & import(T ctx)
    {
      for(driver::context const * x: cache_)
        if((T)*x==ctx)
          return *x;
      cache_.emplace_back(new driver::context(ctx, false));
      return *cache_.back();
    }
    static void get(std::list<context const *> &);
  private:
    static std::list<context const *> cache_;
  };

  class streams
  {
    friend class backend;
  private:
    static void init(std::list<context const *> const &);
    static void release();
  public:
    static void get(driver::context const &, std::vector<stream *> &streams);
    static stream & get(driver::context const &, unsigned int id = 0);
    static stream & get_default();
  private:
    static std::map< context, std::vector<stream*> > cache_;
  };

  static void init();
  static void release();

  static std::vector<device> devices();
  static std::vector<platform> platforms();
  static void synchronize(driver::context const &);

  static unsigned int default_device;
};

}
}

#endif
