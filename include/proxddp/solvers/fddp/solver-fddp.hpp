/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
///
/// @page fddp_intro The FDDP algorithm
/// @htmlinclude fddp.html
#pragma once

#include "proxddp/core/callback-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "./results.hpp"
#include "./workspace.hpp"
#include "./linesearch.hpp"

#include "proxddp/utils/logger.hpp"

#include <fmt/ostream.h>
#include <unordered_map>

/// @brief  A warning for the FDDP module.
#define PROXDDP_FDDP_WARNING(msg) PROXDDP_WARNING("SolverFDDP", msg)

namespace proxddp {

/**
 * @brief   The feasible DDP (FDDP) algorithm, from Mastalli et al. (2020).
 * @details The implementation very similar to Crocoddyl's SolverFDDP.
 *
 */
template <typename Scalar> struct SolverFDDP {
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using ProblemData = TrajOptDataTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VParams = ValueFunctionTpl<Scalar>;
  using QParams = QFunctionTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using ExpModel = ExplicitDynamicsModelTpl<Scalar>;
  using ExpData = ExplicitDynamicsDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<CallbackBaseTpl<Scalar>>;
  using CallbackMap = std::unordered_map<std::string, CallbackPtr>;

  Scalar target_tol_;

  /// @name Regularization parameters
  /// \{
  Scalar reg_init;
  Scalar xreg_ = reg_init;
  Scalar ureg_ = xreg_;
  Scalar reg_min_ = 1e-9;
  Scalar reg_max_ = 1e9;
  /// Regularization decrease factor
  Scalar reg_dec_factor_ = 0.1;
  /// Regularization increase factor
  Scalar reg_inc_factor_ = 10.;
  /// \}

  Scalar th_grad_ = 1e-12;
  Scalar th_step_dec_ = 0.5;
  Scalar th_step_inc_ = 0.01;

  typename Linesearch<Scalar>::Options ls_params;

  VerboseLevel verbose_;
  /// Maximum number of iterations for the solver.
  std::size_t max_iters;
  /// Crocoddyl's FDDP implementation forces the initial state in linesearch to
  /// satisfy the initial condition. This flag switches that behaviour on or
  /// off.
  bool force_initial_condition_;

  BaseLogger logger{};

private:
  /// Callbacks
  CallbackMap callbacks_;

public:
  Results results_;
  Workspace workspace_;

  SolverFDDP(const Scalar tol = 1e-6,
             VerboseLevel verbose = VerboseLevel::QUIET,
             const Scalar reg_init = 1e-9, const std::size_t max_iters = 200);

  /// @brief  Get the solver results.
  PROXDDP_DEPRECATED const Results &getResults() const { return results_; }
  /// @brief  Get a const reference to the solver's workspace.
  PROXDDP_DEPRECATED const Workspace &getWorkspace() const {
    return workspace_;
  }

  /// @brief Allocate workspace and results structs.
  void setup(const Problem &problem);

  /**
   * @brief   Perform a nonlinear rollout, keeping an infeasibility gap.
   * @details Perform a nonlinear rollout using the computed sensitivity gains
   * from the backward pass, while keeping the dynamical feasibility gaps open
   * proportionally to the step-size @p alpha.
   * @param[in]   problem
   * @param[in]   results
   * @param[out]  workspace
   * @param[in]   alpha step-size.
   */
  static Scalar forwardPass(const Problem &problem, const Results &results,
                            Workspace &workspace, const Scalar alpha);

  /**
   * @brief     Pre-compute parts of the directional derivatives -- this is done
   * before linesearch.
   * @details   Inspired from Crocoddyl's own function,
   * crocoddyl::SolverFDDP::updateExpectedImprovement
   */
  void updateExpectedImprovement(Workspace &workspace, Results &results) const;

  /**
   * @brief    Finish computing the directional derivatives -- this is done
   * *within* linesearch.
   * @details  Inspired from Crocoddyl's own function,
   * crocoddyl::SolverFDDP::expectedImprovement
   */
  void expectedImprovement(Workspace &workspace, Scalar &d1, Scalar &d2) const;

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus f(x_i, u_i)$, as
   * well as the residual of initial condition. This function will compute the
   * forward dynamics at every step to compute the forward map $f(x_i, u_i)$.
   */
  inline Scalar computeInfeasibility(const Problem &problem);

  /// @brief   Perform the backward pass and compute Riccati gains.
  void backwardPass(const Problem &problem, Workspace &workspace) const;

  /// @brief   Accept the gains computed in the last backwardPass().
  /// @details This is called if the convergence check after computeCriterion()
  /// did not exit.
  PROXDDP_INLINE void acceptGains(const Workspace &workspace,
                                  Results &results) const {
    assert(workspace.kkt_rhs_bufs.size() == results.gains_.size());
    PROXDDP_NOMALLOC_BEGIN;
    results.gains_ = workspace.kkt_rhs_bufs;
    PROXDDP_NOMALLOC_END;
  }

  inline void increaseRegularization() {
    xreg_ *= reg_inc_factor_;
    xreg_ = std::min(xreg_, reg_max_);
    ureg_ = xreg_;
  }

  inline void decreaseRegularization() {
    xreg_ *= reg_dec_factor_;
    xreg_ = std::max(xreg_, reg_min_);
    ureg_ = xreg_;
  }

  /// @brief Compute the dual feasibility of the problem.
  inline Scalar computeCriterion(Workspace &workspace);

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const std::string &name, CallbackPtr cb) {
    callbacks_[name] = cb;
  }

  const CallbackMap &getCallbacks() const { return callbacks_; }
  void removeCallback(const std::string &name) { callbacks_.erase(name); }
  auto getCallback(const std::string &name) -> CallbackPtr {
    auto cb = callbacks_.find(name);
    if (cb != end(callbacks_)) {
      return cb->second;
    }
    return nullptr;
  }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() { callbacks_.clear(); }

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (const auto &cb : callbacks_) {
      cb.second->call(workspace, results);
    }
  }

  bool run(const Problem &problem, const std::vector<VectorXs> &xs_init = {},
           const std::vector<VectorXs> &us_init = {});

  static const ExpData &
  stage_get_dynamics_data(const StageDataTpl<Scalar> &data) {
    const DynamicsDataTpl<Scalar> &dd = data.dyn_data();
    return static_cast<const ExpData &>(dd);
  }
};

} // namespace proxddp

#include "./solver-fddp.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./solver-fddp.txx"
#endif
