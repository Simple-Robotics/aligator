/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/utils/logger.hpp"

#include <fmt/color.h>
#include <fmt/ranges.h>
#include <cassert>
#include <vector>

namespace aligator {
static constexpr char fstr[] = "{:─^{}s}";

void Logger::printHeadline() {
  if (!active)
    return;
  std::vector<std::string> formattedCols;
  formattedCols.reserve(m_colSpecs.size());
  for (const auto &name : m_columnNames) {
    const auto &spec = m_colSpecs[name];
    formattedCols.push_back(fmt::format(fstr, name, spec.first + 1));
  }
  fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(formattedCols, "┬"));
}

void Logger::log() {
  if (!active)
    return;
  std::vector<std::string_view> cols;
  cols.reserve(m_colSpecs.size());
  for (const auto &name : m_columnNames) {
    cols.emplace_back(m_currentLine[name]);
  }
  fmt::print("{}\n", fmt::join(cols, "│"));
}

void Logger::reset() {
  m_columnNames.clear();
  m_colSpecs.clear();
  m_currentLine.clear();
}

void Logger::finish(bool conv) {
  if (!active)
    return;

  fmt::text_style ts =
      fmt::fg(conv ? fmt::color::dodger_blue : fmt::color::red);
  if (conv) {
    fmt::print(ts, "Successfully converged\n");
  } else {
    fmt::print(ts, "Convergence failure\n");
  }
}

void Logger::addColumn(std::string_view name, uint width,
                       std::string_view format) {
  m_columnNames.emplace_back(name);
  m_colSpecs.emplace(name, std::pair(width, std::string(format)));
}

void Logger::addEntry(std::string_view name, double val) {
  const auto spec = m_colSpecs[name];
  m_currentLine.insert_or_assign(
      name, fmt::format(fmt::runtime(spec.second), val, spec.first));
}

void Logger::addEntry(std::string_view name, size_t val) {
  const auto spec = m_colSpecs[name];
  m_currentLine.insert_or_assign(
      name, fmt::format(fmt::runtime(spec.second), val, spec.first));
}

} // namespace aligator
