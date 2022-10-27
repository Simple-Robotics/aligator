/// @file merit-function.hpp
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"

namespace proxddp {

/// Whether to use merit functions in primal or primal-dual mode.
enum class LinesearchMode : unsigned int { PRIMAL = 0, PRIMAL_DUAL = 1 };

/**
 * @brief   Compute the proximal penalty in the state-control trajectory.
 * @warning Compute the proximal penalty for each timestep first.
 */
template <typename Scalar>
Scalar computeProxPenalty(const WorkspaceTpl<Scalar> &workspace,
                          const Scalar rho) {
  Scalar res = 0.;
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i <= nsteps; i++) {
    res += rho * workspace.prox_datas[i]->value_;
  }
  return res;
}

template <typename Scalar>
Scalar costDirectionalDerivative(const WorkspaceTpl<Scalar> &workspace,
                                 const TrajOptDataTpl<Scalar> &prob_data) {
  Scalar d1 = 0.;
  const std::size_t nsteps = workspace.nsteps;
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageDataTpl<Scalar> &sd = prob_data.getStageData(i);
    const CostDataAbstractTpl<Scalar> &cd = *sd.cost_data;
    d1 += cd.Lx_.dot(workspace.dxs[i]);
    d1 += cd.Lu_.dot(workspace.dus[i]);
  }

  const CostDataAbstractTpl<Scalar> &tcd = *prob_data.term_cost_data;
  d1 += tcd.Lx_.dot(workspace.dxs[nsteps]);
  return d1;
}

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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using CstrSet = ConstraintSetBase<Scalar>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;

  const SolverProxDDP<Scalar> const *solver;
  Scalar traj_cost_ = 0.;
  Scalar penalty_value_ = 0.;
  Scalar value_ = 0.;
  /// Weight of dual penalty. Values different from 1 not supported yet.

  Scalar mu_min = 1e-7;
  Scalar mu_max = 1. / mu_min;

  Scalar dual_weight() { return solver->dual_weight; }

  Scalar mu() const { return std::max(mu_min, solver->mu()); }

  Scalar mu_inv() const { return std::max(mu_max, solver->mu_inv()); }

  PDALFunction(SolverProxDDP<Scalar> const *solver);

  Scalar mu_scaled(std::size_t j) const {
    return std::max(mu_min, solver->mu_scaled(j));
  }

  Scalar mu_inv_scaled(std::size_t j) const {
    return std::min(mu_max, solver->mu_inv_scaled(j));
  }

  /// @brief    Compute the merit function at the trial point.
  /// @warning  Evaluate the problem and proximal terms first!
  Scalar evaluate(const TrajOptProblemTpl<Scalar> &problem,
                  const std::vector<VectorXs> &lams,
                  WorkspaceTpl<Scalar> &workspace,
                  TrajOptDataTpl<Scalar> &prob_data);

  Scalar directionalDerivative(const TrajOptProblemTpl<Scalar> &problem,
                               const std::vector<VectorXs> &lams,
                               WorkspaceTpl<Scalar> &workspace,
                               TrajOptDataTpl<Scalar> &prob_data);
};

} // namespace proxddp

#include "proxddp/core/merit-function.hxx"
