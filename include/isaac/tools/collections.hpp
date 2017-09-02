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

#ifndef ISAAC_CPP_COLLECTIONS_HPP
#define ISAAC_CPP_COLLECTIONS_HPP

#include <vector>
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <deque>

namespace isaac
{
namespace cpp
{

/* ---- Cached Map ----- */
template<class K, class V>
class CachedMap{
public:
  CachedMap(std::function<V(K const &)> value_maker) : value_maker_(value_maker)
  { }

  V const & get(K const & key){
    auto it = cache_.find(key);
    if(it==cache_.end())
      return cache_.insert(std::make_pair(key, value_maker_(key))).first->second;
    return it->second;
  }
private:
  std::map<K, V> cache_;
  std::function<V(K const &)> value_maker_;
};

/* ---- Cartesian ---- */
std::vector<std::vector<int>> cartesian(const std::vector<std::vector<int>>& v) {
  std::vector<std::vector<int>> res = {{}};
  for (const auto& u : v){
    std::vector<std::vector<int>> current;
    for (const auto& x : res)
      for (const auto y : u){
        current.push_back(x);
        current.back().push_back(y);
      }
    res = std::move(current);
  }
  return res;
}

/* ---- Tuple ----- */

template<class T>
class tuple
{
  template<class U>
  friend std::ostream& operator<<(std::ostream & oss, tuple<U> const &);
public:
  tuple() {}
  tuple(std::vector<T> const & list): data_(list){}
  tuple(std::initializer_list<T> const & list) : data_(list){}
  tuple(T a) : data_{a} {}
  tuple(T a, T b) : data_{a, b} {}

  tuple(tuple const & other) = default;
  tuple(tuple&& other) = default;
  tuple& operator=(tuple const & other) = default;
  tuple& operator=(tuple && other) = default;

  typename std::vector<T>::iterator begin() { return data_.begin(); }
  typename std::vector<T>::const_iterator begin() const { return data_.begin(); }
  typename std::vector<T>::iterator end() { return data_.end(); }
  typename std::vector<T>::const_iterator end() const { return data_.end(); }

  size_t size() const { return data_.size(); }
  T front() const { return data_.front(); }
  T back() const { return data_.back(); }

  void remove_index(size_t i) { data_.erase(std::next(data_.begin(), i)); }

  T& operator[](size_t i) { return data_[i]; }
  T operator[](size_t i) const { return data_[i]; }

  bool operator==(tuple const & other) const { return data_==other.data_; }
  operator std::vector<T>() const { return data_; }
private:
  std::vector<T> data_;
};

template<class T>
inline std::ostream& operator<<(std::ostream & oss, tuple<T> const &tp)
{
  oss << "(";
  std::copy(tp.data_.begin(), tp.data_.end() - 1, std::ostream_iterator<T>(oss, ","));
  oss << tp.data_.back();
  if(tp.size()==1)
    oss << ",";
  oss << ")";
  return oss;
}

template<class T>
inline std::string to_string(tuple<T> const & tp)
{
  std::ostringstream oss;
  oss << tp;
  return oss.str();
}

template<class T>
inline void remove_index(std::vector<T>& tp, size_t i)
{ tp.erase(std::next(tp.begin(), i)); }

template<class T>
inline T max(std::vector<T> const & tp)
{ return std::accumulate(tp.begin(), tp.end(), std::numeric_limits<T>::min(), [](T a, T b){ return std::max(a, b); }); }

template<class T>
inline T min(std::vector<T> const & tp)
{ return std::accumulate(tp.begin(), tp.end(), std::numeric_limits<T>::max(), [](T a, T b){ return std::min(a, b); }); }

template<class T>
inline T prod(std::vector<T> const & tp)
{ return std::accumulate(tp.begin(), tp.end(), 1, std::multiplies<T>()); }

template<class T>
inline size_t numgt1(std::vector<T> const & tp)
{ return std::accumulate(tp.begin(), tp.end(), 0, [](size_t a, size_t b){ return a + (b>1); }); }

/* ----- Set/Map ----- */

template<class T>
struct deref_hash
{ size_t operator()(T const & x) const { return x.hash();} };

template<class T>
struct deref_hash<T*>
{ size_t operator()(T const * x) const { return x->hash();} };

template<class T>
struct deref_hash<std::shared_ptr<T>>
{ size_t operator()(std::shared_ptr<T> const & x) const { return x->hash();} };


template<class T>
struct deref_eq
{ size_t operator()(T const & x, T const & y) const { return x == y;} };

template<class T>
struct deref_eq<T*>
{ size_t operator()(T const * x, T const * y) const { return *x == *y;} };

template<class T>
struct deref_eq<std::shared_ptr<T>>
{ size_t operator()(std::shared_ptr<T> const & x, std::shared_ptr<T> const & y) const { return *x == *y;} };

template<class KEY>
using deref_unordered_set = std::unordered_set<KEY, deref_hash<KEY>, deref_eq<KEY>>;

template<class U>
using set_map = std::map<U, std::set<U>>;

template<class U, class H = std::hash<U>, class E = std::equal_to<U>>
using unordered_set_map = std::unordered_map<U, std::unordered_set<U,H,E>, H, E>;

template<class T>
struct is_set_map
{ static const bool value = false; };
template<class U>
struct is_set_map<set_map<U>> { static const bool value = true; };
template<class U, class H, class E>
struct is_set_map<unordered_set_map<U,H,E>> { static const bool value = true; };


/* ---- Transformations ---- */

//Pairs
template<class T, class Enable = typename std::enable_if<is_set_map<T>::value>::type>
std::deque<std::pair<typename T::key_type, typename T::key_type>> pairs(T const & map)
{
  typedef typename T::key_type K;
  std::deque<std::pair<K,K>> result;
  for(auto const& x: map)
    for(auto const & y: x.second)
      result.push_back({x.first, y});
  return result;
}

//Invert
template<class T, class Enable = typename std::enable_if<is_set_map<T>::value>::type>
static T invert(T const & in)
{
  T result;
  typedef typename T::key_type U;
  typedef typename T::mapped_type V;
  for(auto const & x: in){
    U u = x.first;
    result.insert({u, V()});
    for(U v: x.second)
      result[v].insert(u);
  }
  return result;
}

//Intersect
template<class T, class H, class E>
std::unordered_set<T,H,E> intersection(std::unordered_set<T,H,E> const & x,
                                       std::unordered_set<T,H,E> const & y)
{
  if(y.size() < x.size())
    return intersection(y, x);
  std::unordered_set<T,H,E> result;
  for(auto const & u: x)
    if(y.find(u)!=y.end())
      result.insert(u);
  return result;
}

//Merge
template<class T>
typename std::enable_if<!is_set_map<T>::value, T&>::type merge(T& x, T const & y)
{
  std::merge(x.begin(), x.end(), y.begin(), y.end(), std::inserter(x, x.end()));
  return x;
}

template<class T>
typename std::enable_if<is_set_map<T>::value, T&>::type merge(T& x, T const & y)
{
  for(auto const & p: y) merge(x[p.first], p.second);
  return x;
}

//Transfer
template<class T, class U, class Enable = typename std::enable_if<is_set_map<T>::value>::type>
void transfer(T& map, U u, U v, typename T::mapped_type const & exclude)
{
  for(auto const & x: exclude)
    map[v].erase(x);
  merge(map[u], map[v]);
  for(auto& x: map)
    x.second.erase(v);
  map.erase(v);
}

//subset
template<class T, class Enable = typename std::enable_if<is_set_map<T>::value>::type>
T subset(T& map, typename T::mapped_type const & include)
{
  T result;
  for(auto const & e: map)
    if(include.find(e.first)!=include.end())
      result[e.first] = cpp::intersection(e.second, include);
  return result;
}

}
}

#endif
