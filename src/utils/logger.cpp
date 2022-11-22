#include "proxddp/utils/logger.hpp"

namespace proxddp {

void BaseLogger::start() {
  if (!active)
    return;
  static constexpr char fstr[] = "{:^{}s}";
  std::vector<std::string> v;
  v.reserve(BASIC_KEYS.size());
  for (auto it = BASIC_KEYS.cbegin(); it != BASIC_KEYS.cend(); ++it) {
    if (it->first == "inner_crit" && (!inner_crit))
      continue;
    v.push_back(fmt::format(fstr, it->first, it->second));
  }
  fmt::print(fmt::emphasis::bold, "{}\n", fmt::join(v, join_str));
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
