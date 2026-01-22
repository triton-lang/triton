#ifndef PROTON_UTILITY_TRAITS_H_
#define PROTON_UTILITY_TRAITS_H_

#include <type_traits>
#include <variant>

namespace proton {

namespace details {

template <class T, class Variant> struct variant_index;

template <class T, class... Ts> struct variant_index<T, std::variant<Ts...>> {
  static constexpr std::size_t value = []() constexpr {
    std::size_t i = 0;
    (void)((std::is_same_v<T, Ts> ? true : (++i, false)) || ...);
    return i;
  }();
};

} // namespace details
template <class T, class... Ts>
struct is_one_of : std::disjunction<std::is_same<T, Ts>...> {};

template <class T> struct always_false : std::false_type {};

template <class T, class Variant>
inline constexpr std::size_t variant_index_v =
    details::variant_index<T, Variant>::value;

} // namespace proton

#endif // PROTON_UTILITY_TRAITS_H_
