/// @file
/// @copyright Copyright (C) 2026 INRIA
#pragma once

#include <fmt/core.h>

#if FMT_VERSION < 100000

// Redefine `fmt::println`
// This code come from:
// https://github.com/fmtlib/fmt/pull/3267/changes/94d53d405f1b8332277d5bfea33f1e1c460c5f0d
namespace fmt {

template <typename... T>
FMT_INLINE void println(std::FILE *f, format_string<T...> fmt, T &&...args) {
  return fmt::print(f, "{}\n", fmt::format(fmt, std::forward<T>(args)...));
}

template <typename... T>
FMT_INLINE void println(format_string<T...> fmt, T &&...args) {
  return fmt::println(stdout, fmt, std::forward<T>(args)...);
}

} // namespace fmt
#endif // if FMT_VERSION < 100000
