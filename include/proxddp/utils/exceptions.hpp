#pragma once

#include "proxddp/math.hpp"
#include <stdexcept>
#include <fmt/core.h>

#define proxddp_runtime_error(msg)                                             \
  throw std::runtime_error(fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define PROXDDP_RAISE_IF_NAN(value)                                            \
  if (::proxddp::math::checkScalar(value))                                     \
  proxddp_runtime_error("encountered NaN.\n")

namespace proxddp {} // namespace proxddp
