#pragma once

#include <proxnlp/linesearch-base.hpp>

namespace proxddp {

using proxnlp::LinesearchStrategy;

/// @brief  Linesearch parameters.
template <typename Scalar> struct LinesearchOptions {
  Scalar alpha_min = 1e-7;
  Scalar directional_derivative_thresh = 1e-13;
  Scalar armijo_c1 = 1e-4;
  Scalar ls_beta = 0.5;
};

} // namespace proxddp
