/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <array>
#include <vector>
#include <utility>
#include <fmt/color.h>
#include <fmt/ranges.h>

namespace aligator {

constexpr fmt::string_view int_format = "{: >{}d}";
constexpr fmt::string_view sci_format = "{: > {}.{}e}";
constexpr fmt::string_view dbl_format = "{: > {}.{}g}";
struct LogColumn {
  fmt::string_view name;
  fmt::string_view format;
  uint colSize;
};

// log columns names and widths
static const std::array<LogColumn, 11> BASIC_KEYS = {
    {{"iter", int_format, 4U},
     {"alpha", sci_format, 10U},
     {"inner_crit", sci_format, 10U},
     {"prim_err", sci_format, 10U},
     {"dual_err", sci_format, 10U},
     {"xreg", sci_format, 10U},
     {"dphi0", sci_format, 10U},
     {"merit", sci_format, 10U},
     {"delta_M", sci_format, 10U},
     {"aliter", int_format, 6U},
     {"mu", dbl_format, 8U}}};

/// @brief  A table logging utility to log the trace of the numerical solvers.
struct BaseLogger {
  bool active = true;
  bool is_prox = false;
  using iterator = decltype(BASIC_KEYS)::const_iterator;
  static constexpr fmt::string_view join_str = "｜";
  std::vector<std::string> cols;

  BaseLogger();

  void printHeadline();
  void log();
  void finish(bool conv);
  inline void reset() {
    cols.clear();
    m_iter = BASIC_KEYS.cbegin();
  }

  void addEntry(double val) {
    checkIter();
    constexpr int prec = 3;
    cols.push_back(fmt::format(m_iter->format, val, m_iter->colSize, prec));
    m_iter++;
  }

  void addEntry(size_t val) {
    checkIter();
    cols.push_back(fmt::format(m_iter->format, val, m_iter->colSize));
    m_iter++;
  }

  inline void advance() { m_iter++; }

protected:
  iterator m_iter;
  void checkIter() const;
};

} // namespace aligator
