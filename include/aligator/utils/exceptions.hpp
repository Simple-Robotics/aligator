#pragma once

#include <stdexcept>
#include <fmt/color.h>

#define ALIGATOR_RUNTIME_ERROR(msg)                                            \
  throw aligator::RuntimeError(                                                \
      fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define ALIGATOR_DOMAIN_ERROR(msg)                                             \
  throw std::domain_error(fmt::format("{}({}): {}", __FILE__, __LINE__, msg))

#define ALIGATOR_WARNING(loc, msg)                                             \
  fmt::print(fmt::fg(fmt::color::yellow), "[{}] {}: {}", loc, __FUNCTION__,    \
             msg);

namespace aligator {

class RuntimeError : public std::runtime_error {
public:
  explicit RuntimeError(const std::string &what = "")
      : std::runtime_error(what) {}
};

} // namespace aligator
