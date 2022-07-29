#pragma once

#include <vector>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxddp {

const std::vector<std::string> BASIC_KEYS{"iter",     "step_size", "inner_crit",
                                          "prim_err", "dual_err",  "dphi0",
                                          "merit"};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";

struct LogRecord {
  unsigned int iter;
  double step_size;
  double inner_crit;
  double prim_err;
  double dual_err;
  double dphi0;
  double merit;
};

/// @brief  A logging utility.
struct CustomLogger {
  static constexpr unsigned int COL_WIDTH_0 = 6;
  static constexpr unsigned int COL_WIDTH = 10;

  void start() {
    static constexpr char fstr[] = "{: ^{}s}";
    std::vector<std::string> v;
    auto it = BASIC_KEYS.begin();
    v.push_back(fmt::format(fstr, *it, COL_WIDTH_0));
    for (it = BASIC_KEYS.begin() + 1; it != BASIC_KEYS.end(); ++it) {
      v.push_back(fmt::format(fstr, *it, COL_WIDTH));
    }
    fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, " | "));
  }

  template <typename T> void log(const T &values) {
    std::vector<std::string> v;
    int dbl_prec = 3;

    v.push_back(fmt::format(int_format, values.iter, COL_WIDTH_0));
    v.push_back(fmt::format(sci_format, values.step_size, COL_WIDTH, dbl_prec));
    v.push_back(
        fmt::format(sci_format, values.inner_crit, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(sci_format, values.prim_err, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(sci_format, values.dual_err, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(dbl_format, values.dphi0, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(dbl_format, values.merit, COL_WIDTH, dbl_prec));

    fmt::print("{}\n", fmt::join(v, " | "));
  }
};

} // namespace proxddp
