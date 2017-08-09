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

#ifndef ISAAC_TOOLS_CPP_STRING_HPP
#define ISAAC_TOOLS_CPP_STRING_HPP

#include <string>
#include <sstream>
#include <vector>
#include <iostream>

#ifdef _MSC_VER
  #include <algorithm>
#endif

namespace isaac
{
namespace tools
{

template<class T>
inline typename std::enable_if<std::is_fundamental<T>::value, std::string>::type to_string ( T const t )
{
#if defined(ANDROID) || defined(__CYGWIN__)
  std::stringstream ss;
  ss << t;
  return ss.str();
#else
  return  std::to_string(t);
#endif
}

template<class T>
inline typename std::enable_if<!std::is_fundamental<T>::value, std::string>::type to_string ( T const t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

inline std::vector<std::string> &split(const std::string &str, char delim, std::vector<std::string> &res)
{
    std::stringstream ss(str);
    std::string item;
    while (std::getline(ss, item, delim)) {
        res.push_back(item);
    }
    return res;
}


inline std::vector<std::string> split(const std::string &str, char delim)
{
    std::vector<std::string> res;
    split(str, delim, res);
    return res;
}

//

template<class Iterator>
inline std::string join(Iterator begin, Iterator end, std::string delimiter){
  std::string result;
  while(begin!=end){
    result += *begin;
    if(++begin!=end) result += delimiter;
  }
  return result;
}

template<class T>
inline std::string join(T const & x, std::string const & delimiter)
{
  return join(x.begin(), x.end(), delimiter);
}

//

inline int find_and_replace(std::string & source, std::string const & find, std::string const & replace)
{
  int num=0;
  size_t fLen = find.size();
  size_t rLen = replace.size();
  for (size_t pos=0; (pos=source.find(find, pos))!=std::string::npos; pos+=rLen)
  {
    num++;
    source.replace(pos, fLen, replace);
  }
  return num;
}

//

inline std::vector<std::string> tokenize(std::string const & str,std::string const & delimiters)
{
  std::vector<std::string> result;
  size_t to = 0;
  size_t from = str.find_first_not_of(delimiters, to);
  for(size_t i = to ; i < std::max(to+1,from); ++i) result.push_back(str.substr(i,1));
  do{
    to = str.find_first_of(delimiters, from + 1);
    result.push_back(str.substr(from, to - from));
    from = std::min(str.size(), str.find_first_not_of(delimiters, to + 1));
    for(size_t i = to ; i < std::max(to+1,from); ++i) result.push_back(str.substr(i,1));
  }while(to != std::string::npos);
  return result;
}

//
inline void fast_append(char * & ptr, unsigned int val)
{
  if (val==0)
    *ptr++='0';
  else
    while (val>0)
    {
      *ptr++= (char)('0' + (val % 10));
      val /= 10;
    }
}

}
}

#endif
