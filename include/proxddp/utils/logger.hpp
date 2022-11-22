/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <array>
#include <vector>
#include <utility>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace proxddp {

using log_pair_t = std::pair<fmt::string_view, unsigned int>;
static const std::array<log_pair_t, 9> BASIC_KEYS = {{{"iter", 4U},
                                                      {"alpha", 7U},
                                                      {"inner_crit", 10U},
                                                      {"prim_err", 10U},
                                                      {"dual_err", 10U},
                                                      {"xreg", 10U},
                                                      {"dphi0", 10U},
                                                      {"merit", 10U},
                                                      {"delta_M", 10U}}};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";
constexpr char flt_format[] = "{: > {}.{}f}";

template <typename T> struct LogRecordTpl {
  unsigned long iter;
  T step_size;
  T inner_crit;
  T prim_err;
  T dual_err;
  T xreg;
  T dphi0;
  T merit;
  T dM;
};

using LogRecord = LogRecordTpl<double>;

/// @brief  A logging utility.
struct BaseLogger {
  bool active = true;
  const bool inner_crit;
  const std::size_t print_outline_every = 25;
  const char *join_str = "｜";
  std::vector<std::string> v;

  BaseLogger(bool inner_crit = true) : inner_crit(inner_crit) {
    v.reserve(BASIC_KEYS.size());
  }

  void start();
  template <typename T> void log(const LogRecordTpl<T> &values);
  void finish(bool conv);
};

template <typename T> void BaseLogger::log(const LogRecordTpl<T> &values) {
  if (!active)
    return;
  int sci_prec = 3;
  int dbl_prec = 3;
  using fmt::format;
  if (values.iter % print_outline_every == 0)
    start();
  decltype(BASIC_KEYS)::const_iterator it = BASIC_KEYS.cbegin();
  v.push_back(format(int_format, values.iter, it->second));
  ++it;
  v.push_back(format(flt_format, values.step_size, it->second, 4));
  ++it;
  if (inner_crit)
    v.push_back(format(sci_format, values.inner_crit, it->second, sci_prec));
  ++it;
  v.push_back(format(sci_format, values.prim_err, it->second, sci_prec));
  ++it;
  v.push_back(format(sci_format, values.dual_err, it->second, sci_prec));
  ++it;
  v.push_back(format(sci_format, values.xreg, it->second, sci_prec));
  ++it;
  v.push_back(format(sci_format, values.dphi0, it->second, dbl_prec));
  ++it;
  v.push_back(format(sci_format, values.merit, it->second, sci_prec));
  ++it;
  v.push_back(format(dbl_format, values.dM, it->second, dbl_prec));

  fmt::print("{}\n", fmt::join(v, join_str));
  v.clear();
}

} // namespace proxddp
