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

#ifndef ISAAC_TOOLS_CPP_TUPLES_HPP
#define ISAAC_TOOLS_CPP_TUPLES_HPP

#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <numeric>
#include "isaac/defines.h"
#include "isaac/types.h"

namespace isaac
{

class tuple
{
    friend ISAACWINAPI std::ostream& operator<<(std::ostream & oss, tuple const &);
public:
    tuple() {}
    tuple(std::vector<int_t> const & list): data_(list){}
    tuple(std::initializer_list<int_t> const & list) : data_(list){}
    tuple(int_t a) : data_{a} {}
    tuple(int_t a, int_t b) : data_{a, b} {}

    tuple(tuple const & other) = default;
    tuple(tuple&& other) = default;
    tuple& operator=(tuple const & other) = default;
    tuple& operator=(tuple && other) = default;

    std::vector<int_t>::iterator begin() { return data_.begin(); }
    std::vector<int_t>::const_iterator begin() const { return data_.begin(); }
    std::vector<int_t>::iterator end() { return data_.end(); }
    std::vector<int_t>::const_iterator end() const { return data_.end(); }

    size_t size() const { return data_.size(); }
    int_t front() const { return data_.front(); }
    int_t back() const { return data_.back(); }

    int_t& operator[](size_t i) { return data_[i]; }
    int_t operator[](size_t i) const { return data_[i]; }

    bool operator==(tuple const & other) const { return data_==other.data_; }
    operator std::vector<int_t>() const { return data_; }
private:
    std::vector<int_t> data_;
};

inline ISAACAPI std::ostream& operator<<(std::ostream & oss, tuple const &tp)
{
  oss << "(";
  std::copy(tp.data_.begin(), tp.data_.end() - 1, std::ostream_iterator<int_t>(oss, ","));
  oss << tp.data_.back();
  if(tp.size()==1)
    oss << ",";
  oss << ")";
  return oss;
}

inline std::string to_string(tuple const & tp)
{
    std::ostringstream oss;
    oss << tp;
    return oss.str();
}


inline ISAACAPI int_t max(tuple const & tp)
{ return std::accumulate(tp.begin(), tp.end(), std::numeric_limits<int_t>::min(), [](int_t a, int_t b){ return std::max(a, b); }); }

inline ISAACAPI int_t min(tuple const & tp)
{ return std::accumulate(tp.begin(), tp.end(), std::numeric_limits<int_t>::max(), [](int_t a, int_t b){ return std::min(a, b); }); }

inline ISAACAPI int_t prod(tuple const & tp)
{ return std::accumulate(tp.begin(), tp.end(), 1, std::multiplies<int>()); }

inline ISAACAPI size_t numgt1(tuple const & tp)
{ return std::accumulate(tp.begin(), tp.end(), 0, [](size_t a, size_t b){ return a + (b>1); }); }

inline ISAACAPI tuple pad(tuple const & tp, size_t numones)
{
  std::vector<int_t> data = tp;
  data.insert(data.begin(), numones, 1);
  return tuple(data);
}

}

#endif
