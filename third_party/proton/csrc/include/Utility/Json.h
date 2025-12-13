#ifndef PROTON_UTILITY_JSON_H_
#define PROTON_UTILITY_JSON_H_

#include "nlohmann/json.hpp"
#include <unordered_map>

namespace proton {

using Json = nlohmann::basic_json<std::unordered_map>;

} // namespace proton

#endif // PROTON_UTILITY_JSON_H_
