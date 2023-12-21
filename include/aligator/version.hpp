#pragma once

#include "aligator/config.hpp"

#include <string>
#include <sstream>

namespace aligator {

/// @brief    Pretty-print the package version number.
/// @param    delimiter   The delimiter between the major/minor/patch version
/// components.
inline std::string printVersion(const std::string &delimiter = ".") {
  std::ostringstream oss;
  oss << PROXDDP_MAJOR_VERSION << delimiter << PROXDDP_MINOR_VERSION
      << delimiter << PROXDDP_PATCH_VERSION;
  return oss.str();
}
} // namespace aligator
