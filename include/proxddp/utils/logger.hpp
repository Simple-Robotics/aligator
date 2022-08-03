#pragma once

#include <vector>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxddp {

const std::vector<std::string> BASIC_KEYS{"iter",     "step_size", "inner_crit",
                                          "prim_err", "dual_err",  "xreg",
                                          "dphi0",    "merit",     "dM"};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";

struct LogRecord {
  unsigned int iter;
  double step_size;
  double inner_crit;
  double prim_err;
  double dual_err;
  double xreg;
  double dphi0;
  double merit;
  double dM;
};

/// @brief  A logging utility.
struct BaseLogger {
  static constexpr unsigned int COL_WIDTH_0 = 6;
  static constexpr unsigned int COL_WIDTH = 10;
  bool active = true;

  void start() {
    if (!active)
      return;
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
    if (!active)
      return;
    std::vector<std::string> v;
    int dbl_prec = 3;

    v.push_back(fmt::format(int_format, values.iter, COL_WIDTH_0));
    v.push_back(fmt::format(sci_format, values.step_size, COL_WIDTH, dbl_prec));
    v.push_back(
        fmt::format(sci_format, values.inner_crit, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(sci_format, values.prim_err, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(sci_format, values.dual_err, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(sci_format, values.xreg, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(dbl_format, values.dphi0, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(dbl_format, values.merit, COL_WIDTH, dbl_prec));
    v.push_back(fmt::format(dbl_format, values.dM, COL_WIDTH, dbl_prec));

    fmt::print("{}\n", fmt::join(v, " | "));
  }

  void finish(bool conv) {
    if (!active)
      return;
    if (conv)
      fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.");
    else
      fmt::print(fmt::fg(fmt::color::red), "Convergence failure.");
    fmt::print("\n");
  }
};

} // namespace proxddp
