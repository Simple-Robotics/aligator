#pragma once

#include "proxddp/fwd.hpp"

#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxddp {

const std::vector<std::string> BASIC_KEYS{"iter",     "step_size", "inner_crit",
                                          "prim_err", "dual_err",  "dphi0",
                                          "merit"};

struct Log {
  unsigned int iter;
  double step_size;
  double inner_crit;
  double prim_err;
  double dual_err;
  double dphi0;
  double merit;
};

struct CustomLogger {
  unsigned int COL_WIDTH = 10;
  std::string fstr = "{: ^{}s}";

  void start() {
    std::vector<std::string> v;
    for (auto &key : BASIC_KEYS) {
      v.push_back(fmt::format(fstr, key, COL_WIDTH));
    }
    fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, " | "));
  }

  void log(Log &values) {
    std::vector<std::string> v;
    std::string log_format = "{: >{}}";
    int dbl_prec = 3;
    std::string sci_format = "{: > {}.{}e}";
    std::string dbl_format = "{: > {}.{}g}";

    v.push_back(fmt::format(log_format, values.iter, COL_WIDTH));
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
