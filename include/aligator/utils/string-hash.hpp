#pragma once

#include <string_view>

namespace aligator {
/// @brief Extended hashing function for strings which supports const char* and
/// std::string_view.
struct ExtendedStringHash : std::hash<std::string_view> {
  // enable transparent lookup e.g. conversion of the key
  using is_transparent = void;
};
} // namespace aligator
