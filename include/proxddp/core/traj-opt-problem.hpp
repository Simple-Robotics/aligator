#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/stage-model.hpp"

#include "proxddp/modelling/state-error.hpp"

#include <boost/optional.hpp>

namespace proxddp {
/**
 * @brief    Shooting problem, consisting in a succession of nodes.
 *
 * @details  The problem can be written as a nonlinear program:
 * \f[
 *   \begin{aligned}
 *     \min_{\bfx,\bfu}~& \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N)  \\
 *     \subjectto & \varphi(x_i, u_i, x_{i+1}) = 0, \ i \in [ 0, N-1 ] \\
 *                & g(x_i, u_i) \in \calC_i
 *   \end{aligned}
 * \f]
 */
template <typename _Scalar> struct TrajOptProblemTpl {
  using Scalar = _Scalar;
  using StageModel = StageModelTpl<Scalar>;
  using Function = StageFunctionTpl<Scalar>;
  using TrajOptData = TrajOptDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostAbstract = CostAbstractTpl<Scalar>;
  using Constraint = StageConstraintTpl<Scalar>;
  using InitCstrType = StateErrorResidualTpl<Scalar>;

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

  TrajOptProblemTpl(const InitCstrType &resdl,
                    const int nu, const shared_ptr<CostAbstract> &term_cost)
      : init_state_error(resdl), term_cost_(term_cost),
        dummy_term_u0(nu) {
    dummy_term_u0.setZero();
  }

  /// @brief Add a stage to the control problem.
  void addStage(const shared_ptr<StageModel> &stage);

  /// @brief Get initial state constraint.
  const VectorXs &getInitState() const { return init_state_error.target_; }

  /// @brief Set a terminal constraint for the model.
  void setTerminalConstraint(const Constraint &cstr);

  std::size_t numSteps() const;

  /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
  void evaluate(const std::vector<VectorXs> &xs,
                const std::vector<VectorXs> &us, TrajOptData &prob_data) const;

  /**
   * @brief Rollout the problem derivatives, stage per stage.
   *
   * @param xs State sequence
   * @param us Control sequence
   */
  void computeDerivatives(const std::vector<VectorXs> &xs,
                          const std::vector<VectorXs> &us,
                          TrajOptData &prob_data) const;
};

/// @brief Problem data struct.
template <typename _Scalar> struct TrajOptDataTpl {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using StageData = StageDataTpl<Scalar>;

  shared_ptr<FunctionDataTpl<Scalar>> init_data;
  /// Data structs for each stage of the problem.
  std::vector<shared_ptr<StageData>> stage_data;
  /// Terminal cost data.
  shared_ptr<CostDataAbstractTpl<Scalar>> term_cost_data;
  /// Terminal constraint data.
  shared_ptr<FunctionDataTpl<Scalar>> term_cstr_data;

  TrajOptDataTpl() = delete;
  TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem);

  StageData &getData(std::size_t i) { return *stage_data[i]; }
  const StageData &getData(std::size_t i) const { return *stage_data[i]; }
};

/**
 * @brief Compute the trajectory cost.
 *
 * @warning Call TrajOptProblemTpl::evaluate() first!
 */
template <typename Scalar>
Scalar computeTrajectoryCost(const TrajOptProblemTpl<Scalar> &problem,
                             const TrajOptDataTpl<Scalar> &problem_data) {
  Scalar traj_cost = 0.;

  const std::size_t nsteps = problem.numSteps();
  for (std::size_t step = 0; step < nsteps; step++) {
    const StageDataTpl<Scalar> &sd = problem_data.getData(step);
    traj_cost += sd.cost_data->value_;
  }
  traj_cost += problem_data.term_cost_data->value_;

  return traj_cost;
}

} // namespace proxddp

#include "proxddp/core/traj-opt-problem.hxx"
