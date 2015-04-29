#ifndef ISAAC_TOOLS_MAKE_MAP_HPP
#define ISAAC_TOOLD_MAKE_MAP_HPP

#include <vector>

namespace isaac
{

namespace tools
{

template<typename MapT>
class make_map
{
    typedef typename MapT::key_type T;
    typedef typename MapT::mapped_type U;
public:
    make_map(const T& key, const U& val)
    {
        map_.insert(std::make_pair(key,val));
    }

    make_map<MapT>& operator()(const T& key, const U& val)
    {
        map_.insert(std::make_pair(key,val));
        return *this;
    }

    operator MapT()
    {
        return map_;
    }
private:
    MapT map_;
};

}

}

#endif
