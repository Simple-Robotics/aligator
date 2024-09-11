/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/linesearch.hpp"
#include "aligator/core/filter.hpp"
#include "aligator/core/callback-base.hpp"
#include "aligator/core/enums.hpp"
#include "aligator/threads.hpp"
#include "aligator/utils/logger.hpp"
#include "aligator/gar/riccati-base.hpp"

#include "workspace.hpp"
#include "results.hpp"

#include <proxsuite-nlp/bcl-params.hpp>

#include <unordered_map>

namespace aligator {
namespace gar {
template <typename Scalar> class RiccatiSolverBase;
} // namespace gar

/// TODO: NEW G.A.R. BACKEND CAN'T HANDLE DIFFERENT WEIGHTS, PLS FIX
template <typename Scalar> struct DefaultScaling {
  void operator()(ConstraintProximalScalerTpl<Scalar> &scaler) {
    for (std::size_t j = 0; j < scaler.size(); j++)
      scaler.setWeight(scale, j);
  }
  static constexpr Scalar scale = 10.;
};

enum class LQSolverChoice { SERIAL, PARALLEL, STAGEDENSE };

/// @brief A proximal, augmented Lagrangian-type solver for trajectory
/// optimization.
///
/// @details This class implements the Proximal Differential Dynamic Programming
/// algorithm, a variant of the augmented Lagrangian method for trajectory
/// optimization. The paper "PROXDDP: Proximal Constrained Trajectory
/// Optimization" by Jallet et al (2023) is the reference [1] for this
/// implementation.
template <typename _Scalar> struct SolverProxDDPTpl {
public:
  // typedefs

  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using ConstraintType = StageConstraintTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<CallbackBaseTpl<Scalar>>;
  using CallbackMap = std::unordered_map<std::string, CallbackPtr>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using CstrSet = ConstraintSetBase<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using LinesearchOptions = typename Linesearch<Scalar>::Options;
  using CstrProximalScaler = ConstraintProximalScalerTpl<Scalar>;
  using LinesearchType = proxsuite::nlp::ArmijoLinesearch<Scalar>;
  using LQProblem = gar::LQRProblemTpl<Scalar>;
  using Filter = FilterTpl<Scalar>;

  /// Subproblem tolerance
  Scalar inner_tol_;
  /// Desired primal feasibility
  Scalar prim_tol_;
  /// Solver tolerance \f$\epsilon > 0\f$.
  Scalar target_tol_ = 1e-6;

  Scalar mu_init = 0.01; //< Initial AL parameter
  Scalar rho_init = 0.;

  //// Inertia-correcting heuristic

  Scalar reg_min = 1e-10;         //< Minimal nonzero regularization
  Scalar reg_max = 1e9;           //< Maximum regularization value
  Scalar reg_init = 1e-9;         //< Initial regularization value (can be zero)
  Scalar reg_inc_k_ = 10.;        //< Regularization increase factor
  Scalar reg_inc_first_k_ = 100.; //< Regularization increase (critical)
  Scalar reg_dec_k_ = 1. / 3.;    //< Regularization decrease factor
  Scalar preg_ = reg_init;        //< Primal regularization value
  Scalar preg_last_ = 0.;         //< Last "good" regularization value

  //// Initial BCL tolerances

  Scalar inner_tol0 = 1.;
  Scalar prim_tol0 = 1.;

  /// Logger
  Logger logger{};

  /// Solver verbosity level.
  VerboseLevel verbose_;
  /// Choice of linear solver
  LQSolverChoice linear_solver_choice = LQSolverChoice::SERIAL;
  bool lq_print_detailed = false;
  /// Type of Hessian approximation. Default is Gauss-Newton.
  HessianApprox hess_approx_ = HessianApprox::GAUSS_NEWTON;
  /// Linesearch options, as in proxsuite-nlp.
  LinesearchOptions ls_params;
  /// Type of linesearch strategy. Default is Armijo.
  LinesearchStrategy ls_strat = LinesearchStrategy::ARMIJO;
  /// Type of Lagrange multiplier update.
  MultiplierUpdateMode multiplier_update_mode = MultiplierUpdateMode::NEWTON;
  /// Linesearch mode.
  LinesearchMode ls_mode = LinesearchMode::PRIMAL;
  /// Weight of the dual variables in the primal-dual linesearch.
  Scalar dual_weight = 1.0;
  /// Type of rollout for the forward pass.
  RolloutType rollout_type_ = RolloutType::NONLINEAR;
  /// Parameters for the BCL outer loop of the augmented Lagrangian algorithm.
  BCLParamsTpl<Scalar> bcl_params;
  /// Step acceptance mode.
  StepAcceptanceStrategy sa_strategy = StepAcceptanceStrategy::LINESEARCH;

  /// Force the initial state @f$ x_0 @f$ to be fixed to the problem initial
  /// condition.
  bool force_initial_condition_ = true;

  std::size_t maxRefinementSteps_ =
      0; //< Max number of KKT system refinement iters
  Scalar refinementThreshold_ = 1e-13; //< Target tol. for the KKT system.
  std::size_t max_iters;               //< Max number of Newton iterations.
  std::size_t max_al_iters = 100;      //< Maximum number of ALM iterations.
  Scalar mu_lower_bound = 1e-8;        //< Minimum possible penalty parameter.
  uint rollout_max_iters;              //< Nonlinear rollout options

  /// Callbacks
  CallbackMap callbacks_;
  Workspace workspace_;
  Results results_;
  /// LQR subproblem solver
  unique_ptr<gar::RiccatiSolverBase<Scalar>> linearSolver_;
  Filter filter_;

private:
  /// Number of threads
  std::size_t num_threads_ = 1;
  /// Dual proximal/ALM penalty parameter \f$\mu\f$
  /// This is the global parameter: scales may be applied for stagewise
  /// constraints, dynamicals...
  Scalar mu_penal_ = mu_init;
  /// Primal proximal parameter \f$\rho > 0\f$
  Scalar rho_penal_ = rho_init;
  /// Linesearch function
  LinesearchType linesearch_;

public:
  SolverProxDDPTpl(const Scalar tol = 1e-6, const Scalar mu_init = 0.01,
                   const Scalar rho_init = 0.,
                   const std::size_t max_iters = 1000,
                   VerboseLevel verbose = VerboseLevel::QUIET,
                   HessianApprox hess_approx = HessianApprox::GAUSS_NEWTON);

  void setNumThreads(const std::size_t num_threads) {
    if (linearSolver_) {
      ALIGATOR_WARNING(
          "SolverProxDDP",
          "Linear solver already set: setNumThreads() should be called before "
          "you call setup() if you want to use the parallel linear solver.\n");
    }
    num_threads_ = num_threads;
    omp::set_default_options(num_threads);
  }
  std::size_t getNumThreads() const { return num_threads_; }

  ALIGATOR_DEPRECATED const Results &getResults() { return results_; }
  ALIGATOR_DEPRECATED const Workspace &getWorkspace() { return workspace_; }

  /// @brief    Try a step of size \f$\alpha\f$.
  /// @returns  A primal-dual trial point
  ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu,
  ///           \bmlam+\alpha\delta\bmlam)\f$
  /// @returns  The trajectory cost.
  static Scalar tryLinearStep(const Problem &problem, Workspace &workspace,
                              const Results &results, const Scalar alpha);

  /// @brief    Policy rollout using the full nonlinear dynamics. The feedback
  /// gains need to be computed first. This will evaluate all the terms in the
  /// problem into the problem data, similar to TrajOptProblemTpl::evaluate().
  /// @returns  The trajectory cost.
  Scalar tryNonlinearRollout(const Problem &problem, const Scalar alpha);

  Scalar forwardPass(const Problem &problem, const Scalar alpha);

  void updateLQSubproblem();

  /// @brief Allocate new workspace and results instances according to the
  /// specifications of @p problem.
  /// @param problem  The problem instance with respect to which memory will be
  /// allocated.
  void setup(const Problem &problem);

  /// @brief Run the numerical solver.
  /// @param problem  The trajectory optimization problem to solve.
  /// @param xs_init  Initial trajectory guess.
  /// @param us_init  Initial control sequence guess.
  /// @param lams_init  Initial multiplier guess.
  /// @pre  You must call SolverProxDDP::setup beforehand to allocate a
  /// workspace and results.
  bool run(const Problem &problem, const std::vector<VectorXs> &xs_init = {},
           const std::vector<VectorXs> &us_init = {},
           const std::vector<VectorXs> &lams_init = {});

  /// @brief    Perform the inner loop of the algorithm (augmented Lagrangian
  /// minimization).
  bool innerLoop(const Problem &problem);

  /// @brief    Compute the primal infeasibility measures.
  /// @warning  This will alter the constraint values (by projecting on the
  /// normal cone in-place).
  ///           Compute anything which accesses these before!
  void computeInfeasibilities(const Problem &problem);
  /// @brief Compute stationarity criterion (dual infeasibility).
  void computeCriterion();

  /// @name callbacks
  /// \{

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const std::string &name, CallbackPtr cb) {
    callbacks_[name] = cb;
  }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() noexcept { callbacks_.clear(); }

  const CallbackMap &getCallbacks() const { return callbacks_; }
  void removeCallback(const std::string &name) { callbacks_.erase(name); }
  auto getCallback(const std::string &name) -> CallbackPtr {
    auto cb = callbacks_.find(name);
    if (cb != end(callbacks_)) {
      return cb->second;
    }
    return nullptr;
  }

  /// @brief    Invoke callbacks.
  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (const auto &cb : callbacks_) {
      cb.second->call(workspace, results);
    }
  }
  /// \}

  /// Compute the merit function and stopping criterion dual terms:
  /// first-order Lagrange multiplier estimates, shifted and
  /// projected constraints.
  void computeMultipliers(const Problem &problem,
                          const std::vector<VectorXs> &lams,
                          const std::vector<VectorXs> &vs);

  /// @copydoc mu_penal_
  ALIGATOR_INLINE Scalar mu() const { return mu_penal_; }

  /// @copydoc mu_inverse_
  ALIGATOR_INLINE Scalar mu_inv() const { return 1. / mu_penal_; }

  /// @copydoc rho_penal_
  ALIGATOR_INLINE Scalar rho() const { return rho_penal_; }

  /// @brief Update primal-dual feedback gains (control, costate, path
  /// multiplier)
  inline void updateGains();

protected:
  void updateTolsOnFailure() noexcept {
    prim_tol_ = prim_tol0 * std::pow(mu_penal_, bcl_params.prim_alpha);
    inner_tol_ = inner_tol0 * std::pow(mu_penal_, bcl_params.dual_alpha);
  }

  void updateTolsOnSuccess() noexcept {
    prim_tol_ = prim_tol_ * std::pow(mu_penal_, bcl_params.prim_beta);
    inner_tol_ = inner_tol_ * std::pow(mu_penal_, bcl_params.dual_beta);
  }

  /// Set dual proximal/ALM penalty parameter.
  ALIGATOR_INLINE void setAlmPenalty(Scalar new_mu) noexcept {
    mu_penal_ = std::max(new_mu, mu_lower_bound);
  }

  ALIGATOR_INLINE void setRho(Scalar new_rho) noexcept { rho_penal_ = new_rho; }

  // See sec. 3.1 of the IPOPT paper [Wächter, Biegler 2006]
  // called before first bwd pass attempt
  inline void initializeRegularization() noexcept {
    if (preg_last_ == 0.) {
      // this is the 1st iteration
      preg_ = reg_init;
    } else {
      // attempt decrease from last "good" value
      preg_ = std::max(reg_min, preg_last_ * reg_dec_k_);
    }
  }

  inline void increaseRegularization() noexcept {
    if (preg_last_ == 0.)
      preg_ *= reg_inc_first_k_;
    else
      preg_ *= reg_inc_k_;
  }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "solver-proxddp.txx"
#endif
