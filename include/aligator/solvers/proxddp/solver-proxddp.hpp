/// @file solver-proxddp.hpp
/// @brief  Definitions for the proximal trajectory optimization algorithm.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/overloads.hpp"
#include "aligator/core/linesearch-armijo.hpp"
#include "aligator/core/linesearch-nonmonotone.hpp"
#include "aligator/core/filter.hpp"
#include "aligator/core/callback-base.hpp"
#include "aligator/core/enums.hpp"
#include "aligator/utils/logger.hpp"
#include "aligator/gar/riccati-base.hpp"

#include "workspace.hpp"
#include "results.hpp"

#include <boost/unordered_map.hpp>
#include <variant>

namespace aligator {
namespace gar {
template <typename Scalar> class RiccatiSolverBase;
} // namespace gar

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
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using Workspace = WorkspaceTpl<Scalar>;
  using Results = ResultsTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<CallbackBaseTpl<Scalar>>;
  using CallbackMap = boost::unordered_map<std::string, CallbackPtr>;
  using ConstraintStack = ConstraintStackTpl<Scalar>;
  using CstrSet = ConstraintSetTpl<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using LinearSolverPtr = std::unique_ptr<gar::RiccatiSolverBase<Scalar>>;

  struct LinesearchVariant {
    using VariantType = std::variant<std::monostate, ArmijoLinesearch<Scalar>,
                                     NonmonotoneLinesearch<Scalar>>;

    Scalar run(const std::function<Scalar(Scalar)> &fun, const Scalar phi0,
               const Scalar dphi0, Scalar &alpha_try) {
      return std::visit(
          overloads{[](std::monostate &) {
                      return std::numeric_limits<Scalar>::quiet_NaN();
                    },
                    [&](auto &method) {
                      return method.run(fun, phi0, dphi0, alpha_try);
                    }},
          impl_);
    }

    void reset() {
      std::visit(overloads{[](std::monostate &) {},
                           [&](auto &method) { method.reset(); }},
                 impl_);
    }

    bool isValid() const { return impl_.index() > 0ul; }

    operator const VariantType &() const { return impl_; }

  private:
    explicit LinesearchVariant() {}
    void init(StepAcceptanceStrategy strat,
              const LinesearchOptions<Scalar> &options) {
      switch (strat) {
      case StepAcceptanceStrategy::LINESEARCH_ARMIJO:
        impl_ = ArmijoLinesearch(options);
        break;
      case StepAcceptanceStrategy::LINESEARCH_NONMONOTONE:
        impl_ = NonmonotoneLinesearch(options);
        break;
      default:
        ALIGATOR_RUNTIME_ERROR(
            "Provided StepAcceptanceStrategy value is invalid.");
        break;
      }
    }
    friend SolverProxDDPTpl;
    VariantType impl_;
  };

  struct AlmParams {
    /// Log-factor \f$\alpha_\eta\f$ for primal tolerance (failure)
    Scalar prim_alpha = 0.1;
    /// Log-factor \f$\beta_\eta\f$ for primal tolerance (success)
    Scalar prim_beta = 0.9;
    /// Log-factor \f$\alpha_\eta\f$ for dual tolerance (failure)
    Scalar dual_alpha = 1.;
    /// Log-factor \f$\beta_\eta\f$ for dual tolerance (success)
    Scalar dual_beta = 1.;
    /// Scale factor for the dual proximal penalty.
    Scalar mu_update_factor = 0.01;
    /// Lower bound on AL parameter
    Scalar mu_lower_bound = 1e-8; //< Minimum possible penalty parameter.
  };

  /// Subproblem tolerance
  Scalar inner_tol_;
  /// Desired primal feasibility (for each outer loop)
  Scalar prim_tol_;
  /// Solver tolerance \f$\epsilon > 0\f$. When sync_dual_tol is false, this
  /// will be the desired primal feasibility, where the dual feasibility
  /// tolerance is controlled by SolverProxDDPTpl::target_tol_dual.
  Scalar target_tol_ = 1e-6;

private:
  /// Solver desired dual feasibility (by default, same as
  /// SolverProxDDPTpl::target_tol_)
  Scalar target_dual_tol_;
  /// When this is true, dual tolerance will be set to
  /// SolverProxDDPTpl::target_tol_ when SolverProxDDPTpl::run() is called.
  bool sync_dual_tol_;

public:
  Scalar mu_init = 0.01; //< Initial AL parameter

  /// @name Inertia-correcting heuristic
  /// @{

  Scalar reg_min = 1e-10;         //< Minimal nonzero regularization
  Scalar reg_max = 1e9;           //< Maximum regularization value
  Scalar reg_init = 1e-9;         //< Initial regularization value (can be zero)
  Scalar reg_inc_k_ = 10.;        //< Regularization increase factor
  Scalar reg_inc_first_k_ = 100.; //< Regularization increase (critical)
  Scalar reg_dec_k_ = 1. / 3.;    //< Regularization decrease factor
  Scalar preg_ = reg_init;        //< Primal regularization value
  Scalar preg_last_ = 0.;         //< Last "good" regularization value

  /// @}

  Scalar inner_tol0 = 1.; //< Initial BCL inner subproblem tolerance
  Scalar prim_tol0 = 1.;  //< Initial BCL constraint infeasibility tolerance

  /// Logger
  Logger logger{};

  /// Solver verbosity level.
  VerboseLevel verbose_;
  /// Choice of linear solver
  LQSolverChoice linear_solver_choice = LQSolverChoice::SERIAL;
  /// Type of Hessian approximation. Default is Gauss-Newton.
  HessianApprox hess_approx_ = HessianApprox::GAUSS_NEWTON;
  /// Linesearch options.
  LinesearchOptions<Scalar> ls_params;
  /// Type of Lagrange multiplier update.
  MultiplierUpdateMode multiplier_update_mode = MultiplierUpdateMode::NEWTON;
  /// Linesearch mode.
  LinesearchMode ls_mode = LinesearchMode::PRIMAL;
  /// Weight of the dual variables in the primal-dual linesearch.
  Scalar dual_weight = 1.0;
  /// Type of rollout for the forward pass.
  RolloutType rollout_type_ = RolloutType::NONLINEAR;
  /// Parameters for the BCL outer loop of the augmented Lagrangian algorithm.
  AlmParams bcl_params;
  /// Step acceptance mode.
  StepAcceptanceStrategy sa_strategy_;

  /// Force the initial state @f$ x_0 @f$ to be fixed to the problem initial
  /// condition.
  bool force_initial_condition_ = true;

  size_t max_refinement_steps_ = 0;     //< Max KKT system refinement iters.
  Scalar refinement_threshold_ = 1e-13; //< Target tol. for the KKT system.
  size_t max_iters;                     //< Max number of Newton iterations.
  size_t max_al_iters = 100;            //< Maximum number of ALM iterations.

  Workspace workspace_;
  Results results_;
  /// LQR subproblem solver
  LinearSolverPtr linear_solver_;
  /// Filter linesearch
  FilterTpl<Scalar> filter_;
  /// Linesearch function
  LinesearchVariant linesearch_;

private:
  /// Callbacks
  CallbackMap callbacks_;
  /// Number of threads
  size_t num_threads_ = 1;
  /// Dual proximal/ALM penalty parameter \f$\mu\f$
  /// This is the global parameter: scales may be applied for each stagewise
  /// constraint.
  Scalar mu_penal_ = mu_init;

public:
  SolverProxDDPTpl(const Scalar tol = 1e-6, const Scalar mu_init = 0.01,
                   const size_t max_iters = 1000,
                   VerboseLevel verbose = VerboseLevel::QUIET,
                   StepAcceptanceStrategy sa_strategy =
                       StepAcceptanceStrategy::LINESEARCH_NONMONOTONE,
                   HessianApprox hess_approx = HessianApprox::GAUSS_NEWTON);

  inline size_t getNumThreads() const { return num_threads_; }
  void setNumThreads(const size_t num_threads);

  Scalar getDualTolerance() const { return target_dual_tol_; }
  /// Manually set desired dual feasibility tolerance.
  void setDualTolerance(const Scalar tol) {
    target_dual_tol_ = tol;
    sync_dual_tol_ = false;
  }

  /// @brief    Try a step of size \f$\alpha\f$.
  /// @returns  A primal-dual trial point
  ///           \f$(\bfx \oplus\alpha\delta\bfx, \bfu+\alpha\delta\bfu,
  ///           \bmlam+\alpha\delta\bmlam)\f$
  /// @returns  The trajectory cost.
  Scalar tryLinearStep(const Problem &problem, const Scalar alpha);

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
  void cycleProblem(const Problem &problem, const shared_ptr<StageData> &data);

  /// @brief Run the numerical solver.
  /// @param problem  The trajectory optimization problem to solve.
  /// @param xs_init  Initial trajectory guess.
  /// @param us_init  Initial control sequence guess.
  /// @param vs_init  Initial path multiplier guess.
  /// @param lams_init  Initial co-state guess.
  /// @pre  You must call SolverProxDDP::setup beforehand to allocate a
  /// workspace and results.
  bool run(const Problem &problem, const std::vector<VectorXs> &xs_init = {},
           const std::vector<VectorXs> &us_init = {},
           const std::vector<VectorXs> &vs_init = {},
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
  void registerCallback(const std::string &name, CallbackPtr cb);

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() noexcept { callbacks_.clear(); }

  const CallbackMap &getCallbacks() const { return callbacks_; }

  bool removeCallback(const std::string &name) {
    return callbacks_.erase(name);
  }

  auto getCallbackNames() const {
    std::vector<std::string> keys;
    for (const auto &item : callbacks_) {
      keys.push_back(item.first);
    }
    return keys;
  }

  CallbackPtr getCallback(const std::string &name) const {
    auto cb = callbacks_.find(name);
    if (cb != end(callbacks_)) {
      return cb->second;
    }
    return nullptr;
  }

  /// @brief    Invoke callbacks.
  void invokeCallbacks() {
    for (const auto &cb : callbacks_) {
      cb.second->call(workspace_, results_);
    }
  }
  /// \}

  /// Compute the merit function and stopping criterion dual terms:
  /// first-order Lagrange multiplier estimates, shifted and
  /// projected constraints.
  /// @return bool: whether the op succeeded.
  bool computeMultipliers(const Problem &problem,
                          const std::vector<VectorXs> &lams,
                          const std::vector<VectorXs> &vs);

  ALIGATOR_INLINE Scalar mu() const { return mu_penal_; }
  ALIGATOR_INLINE Scalar mu_inv() const { return 1. / mu_penal_; }

  /// @brief Update primal-dual feedback gains (control, costate, path
  /// multiplier)
  inline void updateGains();

protected:
  void updateTolsOnFailure() noexcept {
    const Scalar arg = std::min(mu_penal_, 0.99);
    prim_tol_ = prim_tol0 * std::pow(arg, bcl_params.prim_alpha);
    inner_tol_ = inner_tol0 * std::pow(arg, bcl_params.dual_alpha);
  }

  void updateTolsOnSuccess() noexcept {
    const Scalar arg = std::min(mu_penal_, 0.99);
    prim_tol_ = prim_tol_ * std::pow(arg, bcl_params.prim_beta);
    inner_tol_ = inner_tol_ * std::pow(arg, bcl_params.dual_beta);
  }

  /// Set dual proximal/ALM penalty parameter.
  ALIGATOR_INLINE void setAlmPenalty(Scalar new_mu) noexcept {
    mu_penal_ = std::max(new_mu, bcl_params.mu_lower_bound);
  }

  // See sec. 3.1 of the IPOPT paper [Wächter, Biegler 2006]
  // called before first bwd pass attempt
  inline void initializeRegularization() noexcept {
    if (preg_last_ == 0.) {
      // this is the 1st iteration
      preg_ = std::max(reg_init, reg_min);
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

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct SolverProxDDPTpl<context::Scalar>;
#endif
} // namespace aligator
