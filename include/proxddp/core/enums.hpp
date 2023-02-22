#pragma once

namespace proxddp {

enum struct RolloutType {
  /// Linear rollout
  LINEAR,
  /// Nonlinear rollout, using the full dynamics
  NONLINEAR
};

enum struct HessianApprox {
  /// Use exact Hessian.
  EXACT,
  /// Use the Gauss-Newton approximation.
  GAUSS_NEWTON
};

enum struct MultiplierUpdateMode { NEWTON, PRIMAL, PRIMAL_DUAL };

/// Whether to use merit functions in primal or primal-dual mode.
enum struct LinesearchMode { PRIMAL = 0, PRIMAL_DUAL = 1 };

} // namespace proxddp
