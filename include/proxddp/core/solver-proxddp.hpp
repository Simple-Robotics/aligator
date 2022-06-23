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

namespace proxddp
{
  using proxnlp::VerboseLevel;

  template<typename Scalar>
  struct LinesearchParams
  {
    Scalar alpha_min = 1e-7;
    Scalar directional_derivative_thresh = 1e-13;
    Scalar armijo_c1 = 1e-4;
    Scalar ls_beta = 0.5;
    LinesearchMode ls_mode = LinesearchMode::PRIMAL_DUAL;
  };

  enum class MultiplierUpdateMode : unsigned int
  {
    NEWTON = 0,
    PRIMAL = 1,
    PRIMAL_DUAL = 2
  };

  template<typename Scalar>
  struct BCLParams
  {
    Scalar prim_alpha = 0.1;
    Scalar prim_beta = 0.9;
    Scalar dual_alpha = 1.;
    Scalar dual_beta = 1.;
    Scalar mu_update_factor = 0.01;
    Scalar rho_update_factor = 0.1;
  };

  /// @brief Solver.
  template<typename _Scalar>
  struct SolverProxDDP
  {
  public:
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

    using Problem = ShootingProblemTpl<Scalar>;
    using Workspace = WorkspaceTpl<Scalar>;
    using Results = ResultsTpl<Scalar>;
    using StageModel = StageModelTpl<Scalar>;
    using value_store_t = internal::value_storage<Scalar>;
    using q_store_t = internal::q_function_storage<Scalar>;

    /// Subproblem tolerance
    Scalar inner_tol_;
    /// Desired primal feasibility
    Scalar prim_tol;

    /// Solver tolerance \f$\epsilon > 0\f$.
    Scalar target_tolerance = 1e-6;

    const Scalar mu_init = 0.01;
    const Scalar rho_init = 0.;

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

    Scalar mu_update_factor_ = 0.01;
    Scalar rho_update_factor_ = 0.1;

    VerboseLevel verbose_;
    LinesearchParams<Scalar> ls_params;
    MultiplierUpdateMode mul_update_mode = MultiplierUpdateMode::NEWTON;

    /// Maximum number \f$N_{\mathrm{max}}\f$ of Newton iterations.
    const std::size_t MAX_ITERS;
    const std::size_t MAX_AL_ITERS = 50;

    /// Minimum possible tolerance asked from the solver.
    const Scalar TOL_MIN = 1e-8;
    const Scalar MU_MIN = 1e-8;

    std::unique_ptr<Workspace> workspace_;
    std::unique_ptr<Results> results_;

    Results& getResults() { return *results_; }
    Workspace& getWorkspace() { return *workspace_; }

    SolverProxDDP(const Scalar tol=1e-6,
                  const Scalar mu_init=0.01,
                  const Scalar rho_init=0.,
                  const Scalar prim_alpha=0.1,
                  const Scalar prim_beta=0.9,
                  const Scalar dual_alpha=1.,
                  const Scalar dual_beta=1.,
                  const std::size_t max_iters=1000,
                  const VerboseLevel verbose=VerboseLevel::QUIET
                  )
      : target_tolerance(tol)
      , mu_init(mu_init)
      , rho_init(rho_init)
      , prim_alpha(prim_alpha)
      , prim_beta(prim_beta)
      , dual_alpha(dual_alpha)
      , dual_beta(dual_beta)
      , verbose_(verbose)
      , MAX_ITERS(max_iters)
      {}

    /// @brief Compute the search direction.
    ///
    /// @pre This function assumes \f$\delta x_0\f$ has already been computed!
    /// @returns This computes the primal-dual step \f$(\delta \bfx,\delta \bfu,\delta\bmlam)\f$
    void computeDirection(const Problem& problem, Workspace& workspace, const Results& results) const;

    /// @brief    Try a step of size \f$\alpha\f$.
    /// @returns  A primal-dual trial point
    ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu, \bmlam+\alpha\delta\bmlam)\f$
    void tryStep(const Problem& problem, Workspace& workspace, const Results& results, const Scalar alpha) const;

    /// @brief    Perform the Riccati backward pass.
    ///
    /// @pre  Compute the derivatives first!
    void backwardPass(const Problem& problem, Workspace& workspace, Results& results) const;

    bool run(const Problem& problem,
             const std::vector<VectorXs>& xs_init,
             const std::vector<VectorXs>& us_init)
    {
      workspace_  = std::unique_ptr<Workspace>(new Workspace(problem));
      results_    = std::unique_ptr<Results>(new Results(problem));
      Workspace& workspace = *workspace_;
      Results& results = *results_;

      const std::size_t nsteps = problem.numSteps();
      assert(xs_init.size() == nsteps + 1);
      assert(us_init.size() == nsteps);

      results.xs_ = xs_init;
      results.us_ = us_init;

      workspace.prev_xs_ = results.xs_;
      workspace.prev_us_ = results.us_;
      workspace.prev_lams_ = results.lams_;

      inner_tol_ = inner_tol0;
      prim_tol = prim_tol0;
      updateTolerancesOnFailure();

      inner_tol_ = std::max(inner_tol_, target_tolerance);

      bool& conv = results.conv;

      std::size_t al_iter = 0;
      while ((al_iter < MAX_AL_ITERS) && (results.num_iters < MAX_ITERS))
      {
        if (verbose_ >= 1)
        {
          auto colout = fmt::fg(fmt::color::medium_orchid);
          fmt::print(fmt::emphasis::bold | colout, "[AL iter {:>2d}]", al_iter + 1);
          fmt::print("\n");
          fmt::print(" | inner_tol={:.3e} | dual_tol={:.3e} | mu={:.3e} | rho={:.3e}\n",
                     inner_tol_, prim_tol, mu_, rho_);
        }
        solverInnerLoop(problem, workspace, results);
        computeInfeasibilities(problem, workspace, results);

        if (verbose_ >= 1)
          fmt::print(" | prim. infeas: {:.3e}\n", results.primal_infeasibility);

        // accept primal updates
        workspace.prev_xs_ = results.xs_;
        workspace.prev_us_ = results.us_;

        if (results.primal_infeasibility <= prim_tol)
        {
          updateTolerancesOnSuccess();

          switch (mul_update_mode)
          {
          case MultiplierUpdateMode::NEWTON:
            workspace.prev_lams_ = results.lams_;
            break;
          case MultiplierUpdateMode::PRIMAL:
            workspace.prev_lams_ = workspace.lams_plus_;
            break;
          case MultiplierUpdateMode::PRIMAL_DUAL:
            workspace.prev_lams_ = workspace.lams_pdal_;
            break;
          default:
            break;
          }

          if (results.primal_infeasibility <= target_tolerance)
          {
            conv = true;
            break;
          }
        } else {
          updateALPenalty();
          updateTolerancesOnFailure();
        }
        rho_ *= rho_update_factor_;

        inner_tol_ = std::max(inner_tol_, std::min(TOL_MIN, target_tolerance));
        prim_tol = std::max(prim_tol, target_tolerance);

        al_iter++;
      }

      if (verbose_ >= 1)
      {
        if (conv)
          fmt::print(fmt::fg(fmt::color::dodger_blue), "Successfully converged.");
        else {
          fmt::print(fmt::fg(fmt::color::red), "Convergence failure.");
        }
        fmt::print("\n");
      }

      return conv;
    }

    /// @brief    Perform the inner loop of the algorithm (augmented Lagrangian minimization).
    void solverInnerLoop(const Problem& problem, Workspace& workspace, Results& results);
    /// @brief    Compute the infeasibility measures.
    void computeInfeasibilities(const Problem& problem, Workspace& workspace, Results& results) const;

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
      mu_ = std::max(new_mu, MU_MIN);
      mu_inverse_ = 1. / new_mu;
    }

    /// Update the dual proximal penalty.
    void updateALPenalty()
    {
      setPenalty(mu_ * mu_update_factor_);
    }

  };
 
} // namespace proxddp

#include "proxddp/core/solver-proxddp.hxx"
