/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <array>
#include <string_view>
#include <vector>
#include <map>

namespace aligator {
using uint = unsigned int;

constexpr std::string_view int_format = "{: >{}d}";
constexpr std::string_view sci_format = "{: >{}.4e}";
constexpr std::string_view dbl_format = "{: >{}.4g}";
struct LogColumn {
  std::string_view name;
  std::string_view format;
  uint width;
};

// log columns names and widths
static const std::array<LogColumn, 11> BASIC_KEYS = {
    {{"iter", int_format, 5U},
     {"alpha", sci_format, 10U},
     {"inner_crit", sci_format, 10U},
     {"prim_err", sci_format, 10U},
     {"dual_err", sci_format, 10U},
     {"preg", sci_format, 10U},
     {"dphi0", sci_format, 11U},
     {"merit", sci_format, 11U},
     {"ΔM", sci_format, 11U},
     {"aliter", int_format, 7U},
     {"mu", dbl_format, 8U}}};

/// @brief  A table logging utility to log the trace of the numerical solvers.
struct Logger {
  bool active = true;
  static constexpr std::string_view join_str = "｜";

  Logger();

  void printHeadline();
  void log();
  void finish(bool conv);
  void reset();

  void addColumn(std::string_view name, uint width, std::string_view format);
  void addColumn(LogColumn col) { addColumn(col.name, col.width, col.format); }

  void addEntry(std::string_view name, double val);
  void addEntry(std::string_view name, size_t val);

protected:
  // sizes and formats
  std::vector<std::string_view> m_colNames;
  std::map<std::string_view, std::pair<uint, std::string>> m_colSpecs;
  std::map<std::string_view, std::string> m_currentLine;
};

} // namespace aligator
