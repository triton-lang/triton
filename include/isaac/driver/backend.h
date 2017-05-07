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

#ifndef ISAAC_CL_QUEUES_H
#define ISAAC_CL_QUEUES_H

#include <map>
#include <list>
#include <vector>


namespace isaac
{
namespace driver
{

class Buffer;
class Stream;
class Context;
class Platform;
class Module;
class Kernel;

struct backend
{

  class modules
  {
    friend class backend;
  public:
    static void release();
    static Module& get(Stream const & stream, std::string const & name, std::string const &src);
  private:
    static std::map<std::tuple<Stream, std::string>, Module * > cache_;
  };

  class kernels
  {
    friend class backend;
  public:
    static void release();
    static Kernel & get(Module const & program, std::string const & name);
  private:
    static std::map<std::tuple<Module, std::string>, Kernel * > cache_;
  };

  class contexts
  {
    friend class backend;
  private:
    static void init(std::vector<Platform> const &);
    static void release();
  public:
    static Context const & get_default();
    template<class T>
    static Context const & import(T context)
    {
      for(driver::Context const * x: cache_)
        if((T)*x==context)
          return *x;
      cache_.emplace_back(new Context(context, false));
      return *cache_.back();
    }
    static void get(std::list<Context const *> &);
  private:
    static std::list<Context const *> cache_;
  };

  class streams
  {
    friend class backend;
  private:
    static void init(std::list<Context const *> const &);
    static void release();
  public:
    static void get(Context const &, std::vector<Stream *> &streams);
    static Stream & get(Context const &, unsigned int id = 0);
    static Stream & get_default();
  private:
    static std::map< Context, std::vector<Stream*> > cache_;
  };

  static void init();
  static void release();

  static std::vector<Platform> platforms();
  static void synchronize(Context const &);

  static unsigned int default_device;
};

}
}

#endif
