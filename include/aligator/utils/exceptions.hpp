#pragma once

#include <stdexcept>
#include <string>
#include <fmt/core.h>

#define ALIGATOR_RUNTIME_ERROR(...)                                            \
  throw ::aligator::RuntimeError(                                              \
      ::aligator::detail::exception_msg(__FILE__, __LINE__, __VA_ARGS__))

#define ALIGATOR_DOMAIN_ERROR(msg)                                             \
  throw std::domain_error(                                                     \
      ::aligator::detail::exception_msg(__FILE__, __LINE__, msg))

#define ALIGATOR_WARNING(loc, ...)                                             \
  ::aligator::detail::warning_call(loc, __FUNCTION__, __VA_ARGS__)

namespace aligator {
namespace detail {
void warning_impl(const char *loc, const char *fun, fmt::string_view fstr,
                  fmt::format_args args);
template <typename... T>
void warning_call(const char *loc, const char *fun,
                  fmt::format_string<T...> fstr, T &&...args) {
  warning_impl(loc, fun, fstr, fmt::make_format_args(args...));
}
template <typename T>
void warning_call(const char *loc, const char *fun, T &&msg) {
  warning_impl(loc, fun, msg, {});
}

std::string exception_msg_impl(const char *filename, int lineno,
                               fmt::string_view fstr, fmt::format_args args);
template <typename... T>
std::string exception_msg(const char *filename, int lineno,
                          fmt::format_string<T...> fstr, T &&...args) {
  return exception_msg_impl(filename, lineno, fstr,
                            fmt::make_format_args(args...));
}
} // namespace detail

class RuntimeError : public std::runtime_error {
public:
  explicit RuntimeError(const std::string &what) : std::runtime_error(what) {}
};

} // namespace aligator
