#include "proxddp/utils/logger.hpp"

namespace proxddp {
BaseLogger::BaseLogger() { cols.reserve(BASIC_KEYS.size()); }

void BaseLogger::printHeadline() {
  if (!active)
    return;
  static constexpr char fstr[] = "{:^{}s}";
  for (auto it = BASIC_KEYS.cbegin(); it != BASIC_KEYS.cend(); ++it) {
    if (!is_prox_) {
      if (it->first == "inner_crit" || it->first == "al_iter")
        continue;
    }
    cols.push_back(fmt::format(fstr, it->first, it->second));
  }
  fmt::print(fmt::emphasis::bold, "{}", fmt::join(cols, join_str));
  fmt::print("\n");
  cols.clear();
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

// instantiate

template struct LogRecordTpl<double>;

} // namespace proxddp
