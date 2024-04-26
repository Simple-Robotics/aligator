#pragma once

namespace aligator {

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
  GAUSS_NEWTON,
  /// Use a BFGS-type approximation.
  BFGS
};

enum struct MultiplierUpdateMode { NEWTON, PRIMAL, PRIMAL_DUAL };

/// Whether to use merit functions in primal or primal-dual mode.
enum struct LinesearchMode { PRIMAL = 0, PRIMAL_DUAL = 1 };

/// Whether to use linesearch or filter during step acceptance phase
enum struct StepAcceptanceStrategy { LINESEARCH = 0, FILTER = 1 };

} // namespace aligator
