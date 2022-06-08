/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/solver-workspace.hpp"
#include "proxddp/core/solver-results.hpp"

#include "proxddp/core/merit-function.hpp"

#include <proxnlp/constraint-base.hpp>
#include <proxnlp/linesearch-base.hpp>

#include <fmt/color.h>
#include <fmt/ostream.h>

#include <Eigen/Cholesky>

namespace proxddp
{
  template<typename S>
  struct LinesearchParams
  {
    S alpha_min = 1e-7;
    S directional_derivative_thresh = 1e-13;
    S armijo_c1 = 1e-4;
    S ls_beta = 0.5;
  };

  /// @brief Solver.
  template<typename _Scalar>
  struct SolverProxDDP
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using Problem = ShootingProblemTpl<Scalar>;
    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
    using StageModel = StageModelTpl<Scalar>;
    using value_store_t = internal::value_storage<Scalar>;
    using q_store_t = internal::q_function_storage<Scalar>;

    /// Solver tolerance \f$\epsilon > 0\f$.
    Scalar target_tolerance = 1e-6;

    const Scalar mu_init = 0.01;
    const Scalar rho_init = 0.;

    /// Maximum number \f$N_{\mathrm{max}}\f$ of Newton iterations.
    const std::size_t MAX_STEPS = 20;
    const std::size_t MAX_AL_ITERS = 20;

    /// Dual proximal/constraint penalty parameter \f$\mu\f$
    Scalar mu_ = mu_init;
    Scalar mu_inverse_ = 1. / mu_;

    /// Primal proximal parameter \f$\rho > 0\f$
    Scalar rho_ = rho_init;

    const Scalar inner_tol0 = 1.;
    const Scalar prim_tol0 = 1.;

    /// Log-factor \f$\alpha_\eta\f$ for primal tolerance (failure)
    const Scalar prim_alpha;
    /// Log-factor \f$\beta_\eta\f$ for primal tolerance (success)
    const Scalar prim_beta;
    /// Log-factor \f$\alpha_\eta\f$ for dual tolerance (failure)
    const Scalar dual_alpha;
    /// Log-factor \f$\beta_\eta\f$ for dual tolerance (success)
    const Scalar dual_beta;

    Scalar mu_update_factor_ = 0.1;
    Scalar rho_update_factor_ = 1.;

    /// Subproblem tolerance
    Scalar inner_tol_;
    /// Desired primal feasibility
    Scalar prim_tol;

    LinesearchParams<Scalar> ls_params;

    /// Minimum possible tolerance asked from the solver.
    const Scalar TOL_MIN = 1e-10;

    SolverProxDDP(const Scalar tol=1e-6,
                  const Scalar mu_init=0.01,
                  const Scalar rho_init=0.,
                  const Scalar prim_alpha=0.1,
                  const Scalar prim_beta=0.9,
                  const Scalar dual_alpha=1.,
                  const Scalar dual_beta=1.
                  )
      : target_tolerance(tol)
      , mu_init(mu_init)
      , rho_init(rho_init)
      , prim_alpha(prim_alpha)
      , prim_beta(prim_beta)
      , dual_alpha(dual_alpha)
      , dual_beta(dual_beta)
      {}

    /// @brief Compute the search direction.
    ///
    /// @todo Compute real search direction in x0
    /// @warning This function assumes \f$\delta x_0\f$ has already been computed!
    /// @returns This computes the primal-dual step \f$(\delta \bfx,\delta \bfu,\delta\bmlam)\f$
    void computeDirection(const Problem& problem, Workspace& workspace) const;

    /// @brief    Try a step of size \f$\alpha\f$.
    /// @returns  A primal-dual trial point
    ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu, \bmlam+\alpha\delta\bmlam)\f$
    void tryStep(const Problem& problem,
                 Workspace& workspace,
                 const Results& results,
                 const Scalar alpha) const;

    /// Compute the active sets at each node and multiplier estimates, and projector Jacobian matrices.
    void computeActiveSetsAndMultipliers(const Problem& problem, Workspace& workspace, Results& results) const;

    /// @brief    Perform the Riccati backward pass.
    ///
    /// @warning  Compute the derivatives first!
    void backwardPass(const Problem& problem, Workspace& workspace, Results& results) const;

    bool run(
      const Problem& problem,
      Workspace& workspace,
      Results& results,
      const std::vector<VectorXs>& xs_init,
      const std::vector<VectorXs>& us_init)
    {
      const std::size_t nsteps = problem.numSteps();
      assert(xs_init.size() == nsteps + 1);
      assert(us_init.size() == nsteps);

      results.xs_ = xs_init;
      results.us_ = us_init;
      results.xs_[0] = problem.x0_init;

      workspace.prev_xs_ = results.xs_;
      workspace.prev_us_ = results.us_;
      workspace.prev_lams_ = results.lams_;

      inner_tol_ = inner_tol0;
      prim_tol = prim_tol0;
      this->updateTolerancesOnFailure();

      inner_tol_ = std::max(inner_tol_, target_tolerance);

      bool conv = false;

      std::size_t al_iter;
      for (al_iter = 0; al_iter < MAX_AL_ITERS; al_iter++)
      {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::medium_orchid),
                   "[AL iter {:>02d}]", al_iter);
        fmt::print("\n");
        fmt::print("inner_tol={:.3e} | dual_tol={:.3e} | mu={:.3e} | rho={:.3e}\n",
                   inner_tol_, prim_tol, mu_, rho_);
        solverInnerLoop(problem, workspace, results);

        if (workspace.primal_infeasibility <= prim_tol)
        {
          this->updateTolerancesOnSuccess();
          workspace.prev_lams_ = workspace.lams_plus_;
          if (workspace.primal_infeasibility <= target_tolerance)
          {
            conv = true;
            break;
          }
        } else {
          this->updateALPenalty();
          this->updateTolerancesOnFailure();
        }

        inner_tol_ = std::max(inner_tol_, TOL_MIN);
        prim_tol = std::max(prim_tol, TOL_MIN);
      }

      if (conv)
      {
        fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.\n");
      }

      return conv;
    }

    /** @brief    Perform the inner loop of the algorithm (augmented Lagrangian minimization).
     */
    void solverInnerLoop(
      const Problem& problem,
      Workspace& workspace,
      Results& results)
    {

      const std::size_t nsteps = problem.numSteps();
      results.xs_ = workspace.prev_xs_;
      results.us_ = workspace.prev_us_;
      results.lams_ = workspace.prev_lams_;

      assert(results.xs_.size() == nsteps + 1);
      assert(results.us_.size() == nsteps);
      assert(results.lams_.size() == nsteps);

      // instantiate the subproblem merit function
      PDAL_Function<Scalar> fun { mu_ };

      auto merit_eval_fun = [&](Scalar a0) {
        tryStep(problem, workspace, results, a0);
        return fun.evaluate(
          problem,
          workspace.trial_xs_,
          workspace.trial_us_,
          workspace.trial_lams_,
          workspace,
          *workspace.problem_data);
      };

      std::size_t k = 0;
      while (k < MAX_STEPS)
      {
        fmt::print(fmt::fg(fmt::color::yellow_green), "iter {:>3d}", k);
        fmt::print("\n");
        problem.evaluate(results.xs_, results.us_, *workspace.problem_data);
        problem.computeDerivatives(results.xs_, results.us_, *workspace.problem_data);

        backwardPass(problem, workspace, results);

        fmt::print(" | inner_crit: {:.3e}\n", workspace.inner_criterion);

        if ((workspace.inner_criterion < inner_tol_))
        {
          break;
        }

        computeDirection(problem, workspace);

        Scalar phi0 = merit_eval_fun(0.);
        Scalar eps = 1e-9;
        Scalar veps = merit_eval_fun(eps);
        Scalar dphi0 = (veps - phi0) / eps;

        Scalar alpha_opt = 1;

        proxnlp::ArmijoLinesearch<Scalar>::run(
          merit_eval_fun, phi0, dphi0,
          ls_params.ls_beta, ls_params.armijo_c1, ls_params.alpha_min,
          alpha_opt);

        results.traj_cost_ = fun.traj_cost;
        fmt::print(" | New merit fun. val: {:.3e}\n", phi0);
        fmt::print(" | New traj. cost: {:.3e}\n", results.traj_cost_);

        // accept the damn step
        results.xs_ = workspace.trial_xs_;
        results.us_ = workspace.trial_us_;
        results.lams_ = workspace.trial_lams_;

        k++;
      }
    }


  protected:

    /// @brief  Put together the Q-function parameters and compute the Riccati gains.
    inline void computeGains(
      const Problem& problem,
      Workspace& workspace,
      Results& results,
      const std::size_t step) const;

    void updateTolerancesOnFailure()
    {
      prim_tol = prim_tol0 * std::pow(mu_, prim_alpha);
      inner_tol_ = inner_tol0 * std::pow(mu_, dual_alpha);
    }

    void updateTolerancesOnSuccess()
    {
      prim_tol = prim_tol * std::pow(mu_, prim_beta);
      inner_tol_ = inner_tol_ * std::pow(mu_, dual_beta);
    }

    void setPenalty(Scalar new_mu)
    {
      mu_ = new_mu;
      mu_inverse_ = 1. / new_mu;
    }

    /// Update the dual proximal penalty.
    void updateALPenalty()
    {
      setPenalty(mu_ * mu_update_factor_);
    }

  };
 
} // namespace proxddp

#include "proxddp/solver-proxddp.hxx"
