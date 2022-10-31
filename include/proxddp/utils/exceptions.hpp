#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define PROXDDP_RUNTIME_ERROR(msg)                                             \
  throw std::runtime_error(fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

namespace proxddp {} // namespace proxddp
