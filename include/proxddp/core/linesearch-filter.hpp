#pragma once

#include "proxddp/core/linesearch.hpp"

namespace proxddp {

/// @brief A filter linesearch algorithm.
template <typename T> struct FilterLinesearch {

  struct LSFilter {};

  template <typename Fn>
  void run(Fn phi, const T phi0, const T dphi0, const VerboseLevel,
           Scalar &atry) {
    atry = 1.;
  }
};

} // namespace proxddp
