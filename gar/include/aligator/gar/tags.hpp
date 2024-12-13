#pragma once

namespace aligator {

// tag type
struct no_alloc_t {
  explicit constexpr no_alloc_t() {}
};
static constexpr no_alloc_t no_alloc{};

} // namespace aligator
