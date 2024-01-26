/// @file merit-function.hpp
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/core/enums.hpp"
#include "aligator/core/alm-weights.hpp"

namespace aligator {

template <typename Scalar>
Scalar costDirectionalDerivative(const WorkspaceTpl<Scalar> &workspace,
                                 const TrajOptDataTpl<Scalar> &prob_data);

/** @brief Primal-dual augmented Lagrangian merit function.
 *
 * @details The standard Powell-Hestenes-Rockafellar (PHR) augmented Lagrangian
 * evaluates as: \f[
 *    \calL_{\mu_k}(\bfx, \bfu, \bmlam^k) = J(\bfx, \bfu) +
 * \mathscr{P}(\bfx,\bfu,\bmlam^k; \mu_k), \\ \mathscr{P} = \sum_{i=0}^{N-1}
 *      \frac{1}{2\mu_k}
 *      \left\|
 *      \Bigg[\begin{array}{c}
 *        \phi(x_i, u_i, x_{i+1}) + \mu p^k_i  \\
 *        \Pi_{N_\calC}(h(x_i, u_i) + \mu \nu^k_i)
 *      \end{array}\Bigg]
 *    \right\|^2
 * \f]
 * where \f$\lambda_i^k = (p_i^k, \nu_i^k)\f$ are the multipliers at time node
 * \f$i\f$, \f$J\f$ is the trajectory cost functional, and \f$\mathscr{P}\f$ is
 * the penalty functional.
 * The Gill-Robinson primal-dual variant
 * \f[
 *    \calM_{\mu_k}(\bfx, \bfu, \bmlam; \bmlam^k) = \calL_{\mu_k}(\bfx, \bfu,
 * \bmlam^k) + \mathscr{P}_2 \f] also adds another penalty term \f[
 *    \mathscr{P}_2 = \sum_{i=0}^{N-1}
 *    \frac{1}{2\mu_k}\left\|
 *    \Bigg[ (*) - \lambda_i \Bigg]
 *    \right\|^2
 * \f]
 * where \f$(*)\f$ is the expression within the norm in \f$\mathscr{P}\f$ above.
 *
 * Some of the parameters of this function are obtained from the linked the
 * SolverProxDDP<Scalar> instance.
 */
template <typename _Scalar> struct PDALFunction {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using TrajOptProblem = TrajOptProblemTpl<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using CstrProximalScaler = ConstraintProximalScalerTpl<Scalar>;

  /// @brief    Compute the merit function at the trial point.
  /// @warning  Evaluate the problem and proximal terms first!
  static Scalar evaluate(const Scalar mu, const TrajOptProblem &problem,
                         const std::vector<VectorXs> &lams,
                         const std::vector<VectorXs> &vs, Workspace &workspace);

  static Scalar directionalDerivative(const Scalar mu,
                                      const TrajOptProblem &problem,
                                      const std::vector<VectorXs> &lams,
                                      const std::vector<VectorXs> &vs,
                                      Workspace &workspace);
};

} // namespace aligator

#include "merit-function.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "merit-function.txx"
#endif
