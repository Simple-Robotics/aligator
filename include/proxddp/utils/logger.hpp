/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <vector>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxddp {

static const std::vector<std::string> BASIC_KEYS{
    "iter", "alpha", "inner_crit", "prim_err", "dual_err",
    "xreg", "dphi0", "merit",      "dM"};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";
constexpr char flt_format[] = "{: > {}.{}f}";

struct LogRecord {
  unsigned long iter;
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
  unsigned int COL_WIDTH_0 = 4;
  unsigned int COL_WIDTH_1 = 6;
  unsigned int COL_WIDTH = 10;
  bool active = true;
  const std::string join_str = "｜";

  void start() {
    if (!active)
      return;
    static constexpr char fstr[] = "{:^{}s}";
    std::vector<std::string> v;
    auto it = BASIC_KEYS.begin();
    v.push_back(fmt::format(fstr, *it, COL_WIDTH_0));
    it++;
    v.push_back(fmt::format(fstr, *it, COL_WIDTH_1));
    for (it = BASIC_KEYS.begin() + 2; it != BASIC_KEYS.end(); ++it) {
      v.push_back(fmt::format(fstr, *it, COL_WIDTH));
    }
    fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, join_str));
  }

  template <typename T> void log(const T &values) {
    if (!active)
      return;
    std::vector<std::string> v;
    int sci_prec = 3;
    int dbl_prec = 3;
    using fmt::format;
    if (values.iter % 25 == 0)
      start();
    v.push_back(format(int_format, values.iter, COL_WIDTH_0));
    v.push_back(format(flt_format, values.step_size, COL_WIDTH, dbl_prec));
    v.push_back(format(sci_format, values.inner_crit, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.prim_err, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.dual_err, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.xreg, COL_WIDTH, sci_prec));
    v.push_back(format(sci_format, values.dphi0, COL_WIDTH, dbl_prec));
    v.push_back(format(sci_format, values.merit, COL_WIDTH, sci_prec));
    v.push_back(format(dbl_format, values.dM, COL_WIDTH, dbl_prec));

    fmt::print("{}\n", fmt::join(v, join_str));
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
