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
  for (std::size_t i = 0; i < nsteps + 1; i++) {
    res += rho * workspace.prox_datas[i]->value_;
  }
  return res;
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
 */
template <typename _Scalar> struct PDALFunction {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using FunctionData = FunctionDataTpl<Scalar>;
  using CstrSet = ConstraintSetBase<Scalar>;

  LinesearchMode ls_mode;

  SolverProxDDP<Scalar> const *solver;
  Scalar traj_cost = 0.;
  Scalar penalty_value = 0.;
  Scalar prox_value = 0.;
  Scalar value_ = 0.;
  /// Weight of dual penalty. Values different from 1 not supported yet.
  Scalar dual_weight_ = 1.;

  PDALFunction(SolverProxDDP<Scalar> const *solver, const LinesearchMode mode)
      : ls_mode(mode), solver(solver) {}

  /// @brief    Compute the merit function at the trial point.
  /// @warning  Evaluate the problem and proximal terms first!
  Scalar evaluate(const TrajOptProblemTpl<Scalar> &problem,
                  const std::vector<VectorXs> &lams,
                  WorkspaceTpl<Scalar> &workspace,
                  TrajOptDataTpl<Scalar> &prob_data) {
    traj_cost = computeTrajectoryCost(problem, prob_data);
    penalty_value = prox_value;

    bool with_primal_dual_terms = ls_mode == LinesearchMode::PRIMAL_DUAL;

    // initial constraint
    workspace.lams_plus[0] =
        workspace.prev_lams[0] + solver->mu_inv() * prob_data.init_data->value_;
    workspace.lams_pdal[0] = 2 * workspace.lams_plus[0] - lams[0];
    penalty_value += .5 * solver->mu() * workspace.lams_plus[0].squaredNorm();
    if (with_primal_dual_terms) {
      penalty_value += .5 * dual_weight_ * solver->mu() *
                       (workspace.lams_plus[0] - lams[0]).squaredNorm();
    }

    // stage-per-stage
    std::size_t num_c;
    const std::size_t nsteps = problem.numSteps();
    for (std::size_t step = 0; step < nsteps; step++) {
      const StageModel &stage = *problem.stages_[step];
      const StageData &stage_data = prob_data.getStageData(step);

      num_c = stage.numConstraints();
      // loop over constraints
      // get corresponding multipliers from allocated memory
      for (std::size_t j = 0; j < num_c; j++) {
        const ConstraintContainer<Scalar> &cstr_mgr = stage.constraints_;
        const CstrSet &cstr_set = cstr_mgr.getConstraintSet(j);
        const FunctionData &cstr_data = *stage_data.constraint_data[j];
        auto lamplus_j =
            cstr_mgr.getSegmentByConstraint(workspace.lams_plus[step + 1], j);
        auto lamprev_j = cstr_mgr.getConstSegmentByConstraint(
            workspace.prev_lams[step + 1], j);
        auto c_s_expr = cstr_data.value_ + solver->mu_scaled() * lamprev_j;
        penalty_value += proxnlp::computeMoreauEnvelope(
            cstr_set, c_s_expr, solver->mu_inv_scaled(), lamplus_j);
        lamplus_j *= solver->mu_inv_scaled();
      }
      if (with_primal_dual_terms) {
        penalty_value +=
            .5 * dual_weight_ * solver->mu_scaled() *
            (workspace.lams_plus[step + 1] - lams[step + 1]).squaredNorm();
      }
    }

    if (problem.term_constraint_) {
      const StageConstraintTpl<Scalar> &tc = *problem.term_constraint_;
      const FunctionData &cstr_data = *prob_data.term_cstr_data;
      VectorXs &lamplus = workspace.lams_plus[nsteps + 1];
      auto c_s_expr =
          cstr_data.value_ + solver->mu() * workspace.prev_lams[nsteps + 1];
      penalty_value += proxnlp::computeMoreauEnvelope(
          *tc.set_, c_s_expr, solver->mu_inv(), lamplus);
      lamplus *= solver->mu_inv();
      if (with_primal_dual_terms) {
        penalty_value +=
            .5 * dual_weight_ * solver->mu() *
            (workspace.lams_plus[nsteps + 1] - lams[nsteps + 1]).squaredNorm();
      }
    }

    value_ = traj_cost + penalty_value;
    return value_;
  }
};

} // namespace proxddp
