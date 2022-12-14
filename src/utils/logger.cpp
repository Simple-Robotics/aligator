#include "proxddp/utils/logger.hpp"

namespace proxddp {
BaseLogger::BaseLogger(bool solver_is_prox) : is_prox_(solver_is_prox) {
  v.reserve(BASIC_KEYS.size());
}

void BaseLogger::start() {
  if (!active)
    return;
  static constexpr char fstr[] = "{:^{}s}";
  for (auto it = BASIC_KEYS.cbegin(); it != BASIC_KEYS.cend(); ++it) {
    if (it->first == "inner_crit" && (!is_prox_))
      continue;
    v.push_back(fmt::format(fstr, it->first, it->second));
  }
  fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, join_str));
  v.clear();
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

template void BaseLogger::log(const LogRecord &);

} // namespace proxddp
