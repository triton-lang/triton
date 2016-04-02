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

#ifndef ISAAC_TOOLS_CPP_STRING_HPP
#define ISAAC_TOOLS_CPP_STRING_HPP

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
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
