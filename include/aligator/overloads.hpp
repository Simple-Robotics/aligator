#pragma once

namespace aligator {

/// @brief Utility helper struct for creating visitors from lambdas.
template <typename... Ts> struct overloads : Ts... {
  using Ts::operator()...;
};
template <class... Ts> overloads(Ts...) -> overloads<Ts...>;

} // namespace aligator
