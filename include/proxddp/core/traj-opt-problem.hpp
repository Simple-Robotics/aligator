#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"

#include "proxddp/modelling/state-error.hpp"

#include <boost/optional.hpp>

namespace proxddp {

namespace {
template <typename T> void rot_left(T &v) {
  std::rotate(v.begin(), v.begin() + 1, v.end());
};

} // namespace

/**
 * @brief    Trajectory optimization problem.
 * @tparam   Scalar the scalar type.
 *
 * @details  The problem can be written as a nonlinear program:
 * \f[
 *   \begin{aligned}
 *     \min_{\bmx,\bmu}~& \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N)  \\
 *     \subjectto & \varphi(x_i, u_i, x_{i+1}) = 0, \ i \in [ 0, N-1 ] \\
 *                & g(x_i, u_i) \in \calC_i
 *   \end{aligned}
 * \f]
 */
template <typename _Scalar> struct TrajOptProblemTpl {
  using Scalar = _Scalar;
  using StageModel = StageModelTpl<Scalar>;
  using Function = StageFunctionTpl<Scalar>;
  using Data = TrajOptDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostAbstract = CostAbstractTpl<Scalar>;
  using Constraint = StageConstraintTpl<Scalar>;
  using InitCstrType = StateErrorResidualTpl<Scalar>;

  /**
   * @page trajoptproblem Trajectory optimization problems
   * @tableofcontents
   *
   * # Trajectory optimization
   *
   * The objective of this library is to model and solve optimal control
   * problems (OCPs) of the form
   *
   * \begin{align}
   *     \min_{x,u}~& \int_0^T \ell(x, u)\, dt + \ell_\mathrm{f}(x(T)) \\\\
   *     \subjectto  & \\dot x (t) = f(x(t), u(t)) \\\\
   *                 & g(x(t), u(t)) = 0 \\\\
   *                 & h(x(t), u(t)) \leq 0
   * \end{align}
   *
   * ## Transcription
   * A _transcription_ translates the continuous-time OCP to a discrete-time,
   * finite-dimensional nonlinear program. PROXDDP allows us to consider
   * transcriptions with implicit discrete dynamics: \begin{aligned}
   *     \min_{\bmx,\bmu}~& J(\bmx, \bmu) = \sum_{i=0}^{N-1} \ell_i(x_i, u_i) +
   * \ell_N(x_N) \\\\
   *     \subjectto  & f(x_i, u_i, x_{i+1}) = 0 \\\\
   *                 & g(x_i, u_i) = 0 \\\\
   *                 & h(x_i, u_i) \leq 0
   * \end{aligned}
   *
   * In PROXDDP, trajectory optimization problems are described using the class
   * TrajOptProblemTpl. Each TrajOptProblemTpl is described by a succession of
   * stages (StageModelTpl) which encompass the set of constraints and the cost
   * function (class CostAbstractTpl) for this stage.
   *
   * Additionally, a TrajOptProblemTpl must provide an initial condition @f$ x_0
   * = \bar{x} @f$, a terminal cost
   * $$
   *    \ell_{\mathrm{f}}(x_N)
   * $$
   * on the terminal state @f$x_N @f$; optionally, a terminal constraint
   * @f$g(x_N) = 0, h(x_N) \leq 0 @f$ on this state may be added.
   *
   * # Stage models
   * A stage model (StageModelTpl) describes a node in the discrete-time optimal
   * control problem: it consists in a running cost function, and a vector of
   * constraints (StageConstraintTpl), the first of which @b must describe
   * system dynamics (through a DynamicsModelTpl).
   *
   * # Example
   *
   * Define and solve an LQR (Python API):
   *
   * @include{lineno} lqr.py
   *
   */

  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  /// Initial condition
  InitCstrType init_state_error;
  /// Stages of the control problem.
  std::vector<shared_ptr<StageModel>> stages_;
  /// Terminal cost.
  shared_ptr<CostAbstract> term_cost_;
  /// (Optional) terminal constraint.
  boost::optional<Constraint> term_constraint_ = boost::none;

  VectorXs dummy_term_u0;

  TrajOptProblemTpl(const VectorXs &x0,
                    const std::vector<shared_ptr<StageModel>> &stages,
                    const shared_ptr<CostAbstract> &term_cost);

  TrajOptProblemTpl(const VectorXs &x0, const int nu,
                    const shared_ptr<Manifold> &space,
                    const shared_ptr<CostAbstract> &term_cost);

  TrajOptProblemTpl(const InitCstrType &resdl, const int nu,
                    const shared_ptr<CostAbstract> &term_cost)
      : init_state_error(resdl), term_cost_(term_cost), dummy_term_u0(nu) {
    dummy_term_u0.setZero();
  }

  /// @brief Add a stage to the control problem.
  void addStage(const shared_ptr<StageModel> &stage);

  /// @brief Get initial state constraint.
  const VectorXs &getInitState() const { return init_state_error.target_; }
  /// @brief Set initial state constraint.
  void setInitState(const ConstVectorRef x0) { init_state_error.target_ = x0; }

  /// @brief Set a terminal constraint for the model.
  void setTerminalConstraint(const Constraint &cstr);

  std::size_t numSteps() const;

  /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
  void evaluate(const std::vector<VectorXs> &xs,
                const std::vector<VectorXs> &us, Data &prob_data) const;

  /**
   * @brief Rollout the problem derivatives, stage per stage.
   *
   * @param xs State sequence
   * @param us Control sequence
   */
  void computeDerivatives(const std::vector<VectorXs> &xs,
                          const std::vector<VectorXs> &us,
                          Data &prob_data) const;

  /// @brief Pop out the first StageModel and replace by the supplied one;
  /// updates the supplied Data object.
  void replaceStageCircular(const shared_ptr<StageModel> &model,
                            const shared_ptr<typename StageModel::Data> &sd,
                            Data &data) {
    addStage(model);
    data.stage_data.push_back(sd);

    assert(!stages_.empty());
    assert(!data.stage_data.empty());

    rot_left(stages_);
    rot_left(data.stage_data);
    stages_.pop_back();
    data.stage_data.pop_back();
  }

  /// @copybrief replaceStageCircular(). This variant adds the first StageModel
  /// instance and associated data to the end.
  void replaceStageCircular(Data &data) {
    // use std::rotate to the left
    rot_left(stages_);
    rot_left(data.stage_data);
  }
};

/// @brief Problem data struct.
template <typename _Scalar> struct TrajOptDataTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using FunctionData = FunctionDataTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;

  shared_ptr<FunctionData> init_data;
  /// Data structs for each stage of the problem.
  std::vector<shared_ptr<StageData>> stage_data;
  /// Terminal cost data.
  shared_ptr<CostDataAbstractTpl<Scalar>> term_cost_data;
  /// Terminal constraint data.
  shared_ptr<FunctionDataTpl<Scalar>> term_cstr_data;

  TrajOptDataTpl() = delete;
  TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem);

  /// Get stage data for a stage by time index.
  StageData &getStageData(std::size_t i) { return *stage_data[i]; }
  /// @copydoc getStageData()
  const StageData &getStageData(std::size_t i) const { return *stage_data[i]; }

  /// Get initial constraint function data.
  FunctionData &getInitData() { return *init_data; }
  /// @copydoc getInitData()
  const FunctionData &getInitData() const { return *init_data; }

  /// Get terminal constraint data.
  FunctionData &getTermData() { return *term_cstr_data; }
  /// @copydoc getTermData()
  const FunctionData &getTermData() const { return *term_cstr_data; }
};

/**
 * @brief Compute the trajectory cost.
 *
 * @warning Call TrajOptProblemTpl::evaluate() first!
 */
template <typename Scalar>
Scalar computeTrajectoryCost(const TrajOptProblemTpl<Scalar> &problem,
                             const TrajOptDataTpl<Scalar> &problem_data);

} // namespace proxddp

#include "proxddp/core/traj-opt-problem.hxx"
