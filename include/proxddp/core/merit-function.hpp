/// @file merit-function.hpp
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"


namespace proxddp
{

  /// Whether to use merit functions in primal or primal-dual mode.
  enum class LinesearchMode : unsigned int
  {
    PRIMAL = 0,
    PRIMAL_DUAL = 1
  };

  /** @brief Primal-dual augmented Lagrangian merit function.
   * 
   * @details The standard Powell-Hestenes-Rockafellar (PHR) augmented Lagrangian evaluates as:
   * \f[
   *    \calL_{\mu_k}(\bfx, \bfu, \bmlam^k) = J(\bfx, \bfu) + \mathscr{P}(\bfx,\bfu,\bmlam^k; \mu_k), \\
   *    \mathscr{P} = \sum_{i=0}^{N-1}
   *      \frac{1}{2\mu_k}
   *      \left\|
   *      \Bigg[\begin{array}{c}
   *        \phi(x_i, u_i, x_{i+1}) + \mu p^k_i  \\
   *        \Pi_{N_\calC}(h(x_i, u_i) + \mu \nu^k_i)
   *      \end{array}\Bigg]
   *    \right\|^2
   * \f]
   * where \f$\lambda_i^k = (p_i^k, \nu_i^k)\f$ are the multipliers at time node \f$i\f$,
   * \f$J\f$ is the trajectory cost functional, and \f$\mathscr{P}\f$ is
   * the penalty functional.
   * The Gill-Robinson primal-dual variant
   * \f[
   *    \calM_{\mu_k}(\bfx, \bfu, \bmlam; \bmlam^k) = \calL_{\mu_k}(\bfx, \bfu, \bmlam^k) + \mathscr{P}_2
   * \f]
   * also adds another penalty term
   * \f[
   *    \mathscr{P}_2 = \sum_{i=0}^{N-1}
   *    \frac{1}{2\mu_k}\left\|
   *    \Bigg[ (*) - \lambda_i \Bigg]
   *    \right\|^2
   * \f]
   * where \f$(*)\f$ is the expression within the norm in \f$\mathscr{P}\f$ above.
   */
  template<typename _Scalar>
  struct PDAL_Function
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    Scalar mu_penal_;
    LinesearchMode ls_mode;

    Scalar mu_penal_inv_ = 1. / mu_penal_;
    Scalar traj_cost = 0.;
    Scalar penalty_value = 0.;
    Scalar value_ = 0.;

    /// Weight of dual penalty. Values different from 1 not supported yet.
    static constexpr Scalar dual_weight_ = 1.;


    /// @brief Evaluate the merit function at the trial point.
    Scalar evaluate(
      const TrajOptProblemTpl<Scalar>& problem,
      const std::vector<VectorXs>& xs,
      const std::vector<VectorXs>& us,
      const std::vector<VectorXs>& lams,
      WorkspaceTpl<Scalar>& workspace,
      TrajOptDataTpl<Scalar>& prob_data)
    {
      using StageModel = StageModelTpl<Scalar>;
      problem.evaluate(xs, us, prob_data);

      traj_cost = computeTrajectoryCost(problem, prob_data);
      penalty_value = 0.;

      workspace.lams_plus_[0] = workspace.prev_lams_[0] + mu_penal_inv_ * prob_data.init_data->value_;
      workspace.lams_pdal_[0] = 2 * workspace.lams_plus_[0] - lams[0];
      penalty_value += .5 * mu_penal_ * workspace.lams_plus_[0].squaredNorm();
      if (this->ls_mode == LinesearchMode::PRIMAL_DUAL)
      {
        penalty_value += .5 * dual_weight_ * mu_penal_ * (workspace.lams_plus_[0] - lams[0]).squaredNorm();
      }

      std::size_t num_c;
      const std::size_t nsteps = problem.numSteps();
      for (std::size_t step = 0; step < nsteps; step++)
      {
        const StageModel& sm = problem.stages_[step];
        const StageDataTpl<Scalar>& sd = *prob_data.stage_data[step];

        num_c = sm.numConstraints();
        // loop over constraints
        // get corresponding multipliers from allocated memory
        for (std::size_t j = 0; j < num_c; j++)
        {
          const ConstraintSetBase<Scalar>& cstr_set = sm.constraints_manager[j]->getConstraintSet();
          const FunctionDataTpl<Scalar>& cstr_data = *sd.constraint_data[j];
          VectorRef lamplus_j = sm.constraints_manager.getSegmentByConstraint(workspace.lams_plus_[step + 1], j);
          VectorRef lamprev_j = sm.constraints_manager.getSegmentByConstraint(workspace.prev_lams_[step + 1], j);
          lamplus_j = lamprev_j + mu_penal_inv_ * cstr_data.value_;
          lamplus_j.noalias() = cstr_set.normalConeProjection(lamplus_j);
        }
        penalty_value += .5 * mu_penal_ * workspace.lams_plus_[step + 1].squaredNorm();
        if (this->ls_mode == LinesearchMode::PRIMAL_DUAL)
        {
          penalty_value += .5 * dual_weight_ * mu_penal_ * (workspace.lams_plus_[step + 1] - lams[step + 1]).squaredNorm();
        }
      }
      value_ = traj_cost + penalty_value;
      return value_;
    }

  };
  
} // namespace proxddp
