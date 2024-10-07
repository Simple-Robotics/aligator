#include "aligator/utils/logger.hpp"

#include <fmt/color.h>
#include <fmt/ranges.h>
#include <cassert>

namespace aligator {
static constexpr char fstr[] = "{:─^{}s}";
Logger::Logger() {}

void Logger::printHeadline() {
  if (!active)
    return;
  std::vector<std::string> formattedCols;
  for (const auto name : m_colNames) {
    const auto spec = m_colSpecs[name];
    formattedCols.push_back(fmt::format(fstr, name, spec.first + 1));
  }
  fmt::print(fmt::emphasis::bold, "{}", fmt::join(formattedCols, "┬"));
  fmt::print("\n");
}

void Logger::log() {
  if (!active)
    return;
  std::vector<std::string> cols;
  for (const auto name : m_colNames) {
    auto line = m_currentLine[name];
    cols.push_back(line);
  }
  fmt::print("{}\n", fmt::join(cols, "│"));
}

void Logger::reset() {
  m_colNames.clear();
  m_colSpecs.clear();
  m_currentLine.clear();
}

void Logger::finish(bool conv) {
  if (!active)
    return;

  auto ts = fmt::fg(conv ? fmt::color::dodger_blue : fmt::color::red);
  const char *msg = conv ? "Successfully converged." : "Convergence failure.";
  fmt::print(ts, msg);
  fmt::print("\n");
}

void Logger::addColumn(std::string_view name, uint width,
                       std::string_view format) {
  m_colNames.push_back(name);
  m_colSpecs[name] = {width, std::string(format)};
}

void Logger::addEntry(std::string_view name, double val) {
  const auto spec = m_colSpecs[name];
  m_currentLine[name] = fmt::format(spec.second, val, spec.first);
}

void Logger::addEntry(std::string_view name, size_t val) {
  const auto spec = m_colSpecs[name];
  m_currentLine[name] = fmt::format(spec.second, val, spec.first);
}

} // namespace aligator
