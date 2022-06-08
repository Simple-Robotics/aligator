#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/shooting-problem.hpp"
#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"


namespace proxddp
{
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
    Scalar mu_penal_inv_ = 1. / mu_penal_;

    /// @brief Evaluate the merit function at the trial point.
    Scalar evaluate(
      const ShootingProblemTpl<Scalar>& problem,
      WorkspaceTpl<Scalar>& workspace,
      const ResultsTpl<Scalar>& results,
      ShootingProblemDataTpl<Scalar>& prob_data) const
    {
      problem.evaluate(workspace.trial_xs_, workspace.trial_us_, prob_data);
      Scalar traj_cost = 0.;
      Scalar penalty_value = 0.;
      const std::size_t nsteps = problem.numSteps();
      std::size_t num_c;
      for (std::size_t i = 0; i < nsteps; i++)
      {
        StageDataTpl<Scalar>& sd = *prob_data.stage_data[i];
        traj_cost += sd.cost_data->value_;

        num_c = sd.constraint_data.size();
        for (std::size_t j = 0; j < num_c; j++)
        {
          FunctionDataTpl<Scalar>& cstr_data = *sd.constraint_data[j];
        }
      }
      traj_cost += prob_data.term_cost_data->value_;
      Scalar result = traj_cost + penalty_value;
      return result;
    }

  };
  
} // namespace proxddp
