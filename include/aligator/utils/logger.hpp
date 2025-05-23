/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include <string_view>
#include <vector>
#include <boost/unordered_map.hpp>

namespace aligator {
using uint = unsigned int;

constexpr std::string_view int_format = "{: >{}d} ";
constexpr std::string_view sci_format = "{: >{}.3e} ";
constexpr std::string_view dbl_format = "{: >{}.3g} ";
struct LogColumn {
  std::string_view name;
  std::string_view format;
  uint width;
};

// log columns names and widths
static const LogColumn BASIC_KEYS[12] = {
    {"iter", int_format, 5U},        {"alpha", sci_format, 10U},
    {"inner_crit", sci_format, 11U}, {"prim_err", sci_format, 10U},
    {"dual_err", sci_format, 10U},   {"preg", sci_format, 10U},
    {"cost", sci_format, 10U},       {"dphi0", sci_format, 11U},
    {"merit", sci_format, 10U},      {"ΔM", sci_format, 11U},
    {"aliter", int_format, 7U},      {"mu", dbl_format, 7U}};

/// @brief  A table logging utility to log the trace of the numerical solvers.
struct Logger {
  bool active = true;

  explicit Logger() = default;

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
  boost::unordered_map<std::string_view, std::pair<uint, std::string>>
      m_colSpecs;
  boost::unordered_map<std::string_view, std::string> m_currentLine;
};

} // namespace aligator
