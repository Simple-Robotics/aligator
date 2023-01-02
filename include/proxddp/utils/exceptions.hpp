#pragma once

#include <stdexcept>
#include <fmt/format.h>

#define PROXDDP_RUNTIME_ERROR(msg)                                             \
  throw proxddp::RuntimeError(                                                 \
      fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

namespace proxddp {

class RuntimeError : public std::runtime_error {
public:
  explicit RuntimeError(const std::string &what = "")
      : std::runtime_error(what) {}
};

} // namespace proxddp
