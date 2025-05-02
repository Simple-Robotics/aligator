/// @copyright Copyright (C) 2024 INRIA
#pragma once

namespace aligator {

/// @brief Tag type for e.g. non-allocating constructors.
struct no_alloc_t {
  explicit constexpr no_alloc_t() {}
};
inline constexpr no_alloc_t no_alloc{};

} // namespace aligator
