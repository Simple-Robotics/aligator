#pragma once

#include "proxddp/config.hpp"

#include <string>
#include <sstream>

namespace aligator {

/// @brief    Pretty-print the package version number.
/// @param    delimiter   The delimiter between the major/minor/patch version
/// components.
inline std::string printVersion(const std::string &delimiter = ".") {
  std::ostringstream oss;
  oss << ALIGATOR_MAJOR_VERSION << delimiter << ALIGATOR_MINOR_VERSION
      << delimiter << ALIGATOR_PATCH_VERSION;
  return oss.str();
}
} // namespace aligator
