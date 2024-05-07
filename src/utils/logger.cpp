#include "aligator/utils/logger.hpp"

#include <cassert>

namespace aligator {
BaseLogger::BaseLogger() : m_iter(BASIC_KEYS.cbegin()) {
  cols.reserve(BASIC_KEYS.size());
}

void BaseLogger::printHeadline() {
  if (!active)
    return;
  static constexpr char fstr[] = "{:^{}s}";
  std::array<std::string, BASIC_KEYS.size()> columns;
  for (size_t j = 0; j < BASIC_KEYS.size(); j++) {
    auto key = BASIC_KEYS[j];
    if (!is_prox) {
      if (key.name == "inner_crit" || key.name == "al_iter")
        continue;
    }
    columns[j] = fmt::format(fstr, key.name, key.colSize);
  }
  fmt::print(fmt::emphasis::bold, "{}", fmt::join(columns, join_str));
  fmt::print("\n");
}

void BaseLogger::log() {
  if (!active)
    return;
  fmt::print("{}\n", fmt::join(cols, join_str));
  reset();
}

void BaseLogger::finish(bool conv) {
  if (!active)
    return;
  if (conv)
    fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.");
  else
    fmt::print(fmt::fg(fmt::color::red), "Convergence failure.");
  fmt::print("\n");
}

bool BaseLogger::checkIter() const {
  return active && (m_iter != BASIC_KEYS.cend());
}

} // namespace aligator
