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
  for (const auto &[name, spec] : m_colSpecs) {
    formattedCols.push_back(fmt::format(fstr, name, spec.first + 1));
  }
  fmt::print(fmt::emphasis::bold, "{}", fmt::join(formattedCols, "┬"));
  fmt::print("\n");
}

void Logger::log() {
  if (!active)
    return;
  std::vector<std::string_view> cols;
  cols.reserve(m_colSpecs.size());
  for (const auto &[name, spec] : m_colSpecs) {
    cols.emplace_back(m_currentLine[name]);
  }
  fmt::print("{}\n", fmt::join(cols, "│"));
}

void Logger::reset() {
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
  m_colSpecs[name] = {width, std::string(format)};
}

void Logger::addEntry(std::string_view name, double val) {
  const auto spec = m_colSpecs[name];
  m_currentLine[name] = fmt::format(fmt::runtime(spec.second), val, spec.first);
}

void Logger::addEntry(std::string_view name, size_t val) {
  const auto spec = m_colSpecs[name];
  m_currentLine[name] = fmt::format(fmt::runtime(spec.second), val, spec.first);
}

} // namespace aligator
