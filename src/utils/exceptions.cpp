#include "aligator/utils/exceptions.hpp"
#include <fmt/color.h>

namespace aligator {
namespace detail {
void warning_impl(const char *loc, const char *fun, fmt::string_view fstr,
                  fmt::format_args args) {
  const auto ts = fmt::fg(fmt::color::yellow);
  fmt::print(ts, "[Warning] {:s}: {:s}: {}", loc, fun,
             fmt::vformat(fstr, args));
}

std::string exception_msg_impl(const char *filename, int lineno,
                               fmt::string_view fstr, fmt::format_args args) {
  return fmt::format("{:s}({:d}): {}", filename, lineno,
                     fmt::vformat(fstr, args));
}
} // namespace detail
} // namespace aligator
