#pragma once

namespace aligator {

// tag type
struct no_alloc_t {
  explicit constexpr no_alloc_t() {}
};
inline constexpr no_alloc_t no_alloc{};

} // namespace aligator
