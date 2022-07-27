#pragma once

#include <stdexcept>
#include <fmt/core.h>

#define proxddp_runtime_error(msg)                                             \
  {                                                                            \
    throw std::runtime_error(                                                  \
        fmt::format("{}({}): {}", __FILE__, __LINE__, msg));                   \
  }

namespace proxddp {} // namespace proxddp
