#pragma once

#include "proxddp/core/linesearch.hpp"

namespace proxddp {

/// @brief  Filter data structure to use with the linesearch algorithm.
template <typename T> struct LSFilter {};

/// @brief A filter linesearch algorithm.
template <typename T> struct FilterLinesearch {
  template <typename Fn>
  void run(Fn phi, const T phi0, const T dphi0, const VerboseLevel,
           Scalar &atry) {
    atry = 1.;
  }
};

} // namespace proxddp
