/**
 * @page fddp_intro The FDDP algorithm
 * @htmlinclude fddp.html
 */
#pragma once

#include "proxddp/core/solver-util.hpp"
#include "proxddp/core/solver-results.hpp"
#include "proxddp/core/linesearch.hpp"
#include "proxddp/core/helpers-base.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include "proxddp/fddp/results.hpp"
#include "proxddp/fddp/workspace.hpp"
#include "proxddp/fddp/linesearch.hpp"

#include "proxddp/utils/exceptions.hpp"
#include "proxddp/utils/logger.hpp"
#include "proxddp/utils/rollout.hpp"

#include <fmt/ostream.h>

#include <Eigen/Cholesky>

#define proxddp_fddp_warning(msg)                                              \
  fmt::print(fmt::fg(fmt::color::yellow), "[SolverFDDP] ({}) warning: {}",     \
             __FUNCTION__, msg)

namespace proxddp {

/**
 * @brief The feasible DDP (FDDP) algorithm
 *
 */
template <typename Scalar> struct SolverFDDP {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Problem = TrajOptProblemTpl<Scalar>;
  using StageModel = StageModelTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using ProblemData = TrajOptDataTpl<Scalar>;
  using Results = ResultsFDDPTpl<Scalar>;
  using Workspace = WorkspaceFDDPTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VParams = internal::value_storage<Scalar>;
  using QParams = internal::q_storage<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using ExpModel = ExplicitDynamicsModelTpl<Scalar>;
  using ExpData = ExplicitDynamicsDataTpl<Scalar>;
  using CallbackPtr = shared_ptr<helpers::base_callback<Scalar>>;

  const Scalar tol_;

  /// @name Regularization parameters
  /// \{
  Scalar xreg_;
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

  LinesearchParams<Scalar> ls_params;
  enum ls_types { GOLDSTEIN, ARMIJO };
  ls_types ls_type = GOLDSTEIN;

  VerboseLevel verbose_;
  /// Maximum number of iterations for the solver.
  std::size_t MAX_ITERS = 200;

  /// Callbacks
  std::vector<CallbackPtr> callbacks_;

  std::unique_ptr<Results> results_;
  std::unique_ptr<Workspace> workspace_;

  SolverFDDP(const Scalar tol = 1e-6,
             VerboseLevel verbose = VerboseLevel::QUIET,
             const Scalar reg_init = 1e-10);

  const Results &getResults() const { return *results_; }
  const Workspace &getWorkspace() const { return *workspace_; }

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
  static void forwardPass(const Problem &problem, const Results &results,
                          Workspace &workspace, const Scalar alpha);

  /// @brief Try a given step size, and store the resulting cost in
  /// `Results::traj_cost_`.
  static Scalar tryStep(const Problem &problem, const Results &results,
                        Workspace &workspace, const Scalar alpha);

  void computeDirectionalDerivatives(Workspace &workspace, Results &results,
                                     Scalar &d1, Scalar &d2) const;

  /**
   * @brief    Correct the directional derivatives.
   * @details  This will re-compute the gap between the results trajectory and
   * the trial trajectory.
   */
  static void directionalDerivativeCorrection(const Problem &problem,
                                              Workspace &workspace,
                                              Results &results, Scalar &d1,
                                              Scalar &d2);

  /**
   * @brief   Computes dynamical feasibility gaps.
   * @details This computes the difference \f$x_{i+1} \ominus f(x_i, u_i)$, as
   * well as the residual of initial condition. This function will compute the
   * forward dynamics at every step to compute the forward map $f(x_i, u_i)$.
   */
  static Scalar computeInfeasibility(const Problem &problem,
                                     const std::vector<VectorXs> &xs,
                                     const std::vector<VectorXs> &us,
                                     Workspace &workspace);

  /// @brief   Perform the backward pass and compute Riccati gains.
  void backwardPass(const Problem &problem, Workspace &workspace,
                    Results &results) const;

  void increaseRegularization() {
    xreg_ *= reg_inc_factor_;
    xreg_ = std::min(xreg_, reg_max_);
    ureg_ = xreg_;
  }

  void decreaseRegularization() {
    xreg_ *= reg_dec_factor_;
    xreg_ = std::max(xreg_, reg_min_);
    ureg_ = xreg_;
  }

  /// @brief Compute the dual feasibility of the problem.
  void computeCriterion(Workspace &workspace, Results &results);

  /// @brief    Add a callback to the solver instance.
  void registerCallback(const CallbackPtr &cb) { callbacks_.push_back(cb); }

  /// @brief    Remove all callbacks from the instance.
  void clearCallbacks() { callbacks_.clear(); }

  void invokeCallbacks(Workspace &workspace, Results &results) {
    for (auto &cb : callbacks_) {
      cb->call(workspace, results);
    }
  }

  /**
   * @brief   Perform a linear rollout recovering the Newton step.
   * @details This is useful for debugging purposes.
   */
  static void linearRollout(const Problem &problem, Workspace &workspace,
                            const Results &results) {
    const auto &fs = workspace.feas_gaps_;
    auto &dxs = workspace.dxs_;
    auto &dus = workspace.dus_;
    const Manifold &space = problem.stages_[0]->xspace();
    dxs[0] = fs[0];
    const std::size_t nsteps = workspace.nsteps;
    for (std::size_t i = 0; i < nsteps; i++) {
      const StageData &sd = workspace.problem_data.getData(i);
      const DynamicsDataTpl<Scalar> &dd = stage_get_dynamics_data(sd);
      auto ff = results.getFeedforward(i);
      auto fb = results.getFeedback(i);
      dus[i] = ff + fb * dxs[i];
      dxs[i + 1] = fs[i + 1] + dd.Jx_ * dxs[i] + dd.Ju_ * dus[i];
    }
  }

  bool run(const Problem &problem,
           const std::vector<VectorXs> &xs_init = DEFAULT_VECTOR<Scalar>,
           const std::vector<VectorXs> &us_init = DEFAULT_VECTOR<Scalar>);

  static DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(StageDataTpl<Scalar> &sd) {
    return static_cast<DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }

  static const DynamicsDataTpl<Scalar> &
  stage_get_dynamics_data(const StageDataTpl<Scalar> &sd) {
    return static_cast<const DynamicsDataTpl<Scalar> &>(*sd.constraint_data[0]);
  }
};

} // namespace proxddp

#include "proxddp/fddp/solver-fddp.hxx"
