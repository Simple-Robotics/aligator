/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <array>
#include <vector>
#include <utility>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace aligator {

using log_pair_t = std::pair<fmt::string_view, unsigned int>;
static constexpr int NKEYS = 11;
// log columns names and widths
static const std::array<log_pair_t, NKEYS> BASIC_KEYS = {{{"iter", 4U},
                                                          {"alpha", 10U},
                                                          {"inner_crit", 10U},
                                                          {"prim_err", 10U},
                                                          {"dual_err", 10U},
                                                          {"xreg", 10U},
                                                          {"dphi0", 10U},
                                                          {"merit", 10U},
                                                          {"delta_M", 10U},
                                                          {"aliter", 6U},
                                                          {"mu", 8U}}};
constexpr char int_format[] = "{: >{}d}";
constexpr char sci_format[] = "{: > {}.{}e}";
constexpr char dbl_format[] = "{: > {}.{}g}";

template <typename T> struct LogRecordTpl {
  unsigned long iter;
  T step_size = 0.;
  T inner_crit = 0.;
  T prim_err = 0.;
  T dual_err = 0.;
  T xreg = 0.;
  T dphi0 = 0.;
  T merit = 0.;
  T dM = 0.;
  unsigned long al_iter = 0;
  T mu;
};

using LogRecord = LogRecordTpl<double>;

/// @brief  A logging utility.
struct BaseLogger {
  bool active = true;
  bool is_prox_ = false;
  std::size_t print_outline_every = 25;
  using key_it_t = decltype(BASIC_KEYS)::const_iterator;
  const char *join_str = "｜";
  std::vector<std::string> cols;

  BaseLogger();

  void printHeadline();
  void log(const LogRecord &values) {
    if (!active)
      return;
    if (values.iter % print_outline_every == 0)
      printHeadline();
    log_impl(values);
    fmt::print("{}\n", fmt::join(cols, join_str));
    cols.clear();
  }
  void finish(bool conv);

protected:
  virtual auto log_impl(const LogRecord &values) -> key_it_t {
    constexpr int sci_prec = 3;
    constexpr int dbl_prec = 3;
    using fmt::format;
    auto it = BASIC_KEYS.cbegin();
    cols.push_back(format(int_format, values.iter, it->second));
    ++it;
    cols.push_back(format(sci_format, values.step_size, it->second, sci_prec));
    ++it;
    if (is_prox_)
      cols.push_back(
          format(sci_format, values.inner_crit, it->second, sci_prec));
    ++it;
    cols.push_back(format(sci_format, values.prim_err, it->second, sci_prec));
    ++it;
    cols.push_back(format(sci_format, values.dual_err, it->second, sci_prec));
    ++it;
    cols.push_back(format(sci_format, values.xreg, it->second, sci_prec));
    ++it;
    cols.push_back(format(sci_format, values.dphi0, it->second, dbl_prec));
    ++it;
    cols.push_back(format(sci_format, values.merit, it->second, sci_prec));
    ++it;
    cols.push_back(format(dbl_format, values.dM, it->second, dbl_prec));
    ++it;
    return it;
  }
};

struct ALMLogger : BaseLogger {

  ALMLogger() : BaseLogger() { is_prox_ = true; }
  auto log_impl(const LogRecord &values) -> key_it_t {
    using fmt::format;
    auto it = BaseLogger::log_impl(values);
    cols.push_back(format(int_format, values.al_iter, it->second));
    ++it;
    cols.push_back(format(sci_format, values.mu, it->second, 2));
    ++it;
    return it;
  }
};

} // namespace aligator
