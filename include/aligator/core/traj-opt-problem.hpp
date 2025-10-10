/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/stage-model.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

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
 * finite-dimensional nonlinear program. Aligator allows us to consider
 * transcriptions with implicit discrete dynamics: \begin{aligned}
 *     \min_{\bmx,\bmu}~& J(\bmx, \bmu) = \sum_{i=0}^{N-1} \ell_i(x_i, u_i) +
 * \ell_N(x_N) \\\\
 *     \subjectto  & f(x_i, u_i, x_{i+1}) = 0 \\\\
 *                 & g(x_i, u_i) = 0 \\\\
 *                 & h(x_i, u_i) \leq 0
 * \end{aligned}
 *
 * In aligator, trajectory optimization problems are described using the class
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

/// @brief Assign a vector of Eigen types into another, ensure there is no
/// resize
template <typename T1, typename T2>
[[nodiscard]] bool assign_no_resize(const std::vector<T1> &lhs,
                                    std::vector<T2> &rhs) {
  static_assert(std::is_base_of_v<Eigen::EigenBase<T1>, T1>,
                "T1 should be an Eigen object!");
  static_assert(std::is_base_of_v<Eigen::EigenBase<T2>, T2>,
                "T2 should be an Eigen object!");
  if (lhs.size() != rhs.size())
    return false;

  const auto same_dims = [](auto &x, auto &y) {
    return (x.cols() == y.cols()) && (x.rows() == y.rows());
  };

  for (std::size_t i = 0; i < lhs.size(); i++) {
    if (!same_dims(lhs[i], rhs[i]))
      return false;
    rhs[i] = lhs[i];
  }
  return true;
}

/**
 * @brief    Trajectory optimization problem.
 * @tparam   Scalar the scalar type.
 *
 * @details  The problem can be written as a nonlinear program:
 * \f[
 *   \begin{aligned}
 *     \min_{\bmx,\bmu}~& \sum_{i=0}^{N-1} \ell_i(x_i, u_i) + \ell_N(x_N)  \\
 *     \subjectto & \varphi(x_i, u_i, x_{i+1}) = 0, \ 0 \leq i < N \\
 *                & g(x_i, u_i) \in \calC_i
 *   \end{aligned}
 * \f]
 */
template <typename _Scalar> struct TrajOptProblemTpl {
  using Self = TrajOptProblemTpl;
  using Scalar = _Scalar;

  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using StageModel = StageModelTpl<Scalar>;
  using StageFunction = StageFunctionTpl<Scalar>;
  using UnaryFunction = UnaryFunctionTpl<Scalar>;
  using Data = TrajOptDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using CostAbstract = CostAbstractTpl<Scalar>;
  using ConstraintSet = ConstraintSetTpl<Scalar>;
  using StateErrorResidual = StateErrorResidualTpl<Scalar>;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  using StageConstraint = StageConstraintTpl<Scalar>;
#pragma GCC diagnostic pop
  using InitializationStrategy =
      std::function<void(const Self &, std::vector<VectorXs> &)>;

  /// Initial condition
  xyz::polymorphic<UnaryFunction> init_constraint_;
  /// Stages of the control problem.
  std::vector<xyz::polymorphic<StageModel>> stages_;
  /// Terminal cost.
  xyz::polymorphic<CostAbstract> term_cost_;
  /// Terminal constraints.
  ConstraintStackTpl<Scalar> term_cstrs_;
  /// Dummy, "neutral" control value.
  VectorXs unone_;

  /// @name Constructors with pre-allocated stages
  /// @{

  /// @brief Constructor with a given constraint function of any given type.
  TrajOptProblemTpl(xyz::polymorphic<UnaryFunction> init_constraint,
                    const std::vector<xyz::polymorphic<StageModel>> &stages,
                    xyz::polymorphic<CostAbstract> term_cost);

  /// @brief Constructor for an initial value problem.
  TrajOptProblemTpl(const ConstVectorRef &x0,
                    const std::vector<xyz::polymorphic<StageModel>> &stages,
                    xyz::polymorphic<CostAbstract> term_cost);
  /// @}

  /// @name Constructors without pre-allocated stages
  /// @{

  /// @brief Constructor with a given constraint function of any given type.
  TrajOptProblemTpl(xyz::polymorphic<UnaryFunction> init_constraint,
                    xyz::polymorphic<CostAbstract> term_cost);

  /// @brief Constructor for an initial value problem.
  TrajOptProblemTpl(const ConstVectorRef &x0, const int nu,
                    xyz::polymorphic<Manifold> space,
                    xyz::polymorphic<CostAbstract> term_cost);
  /// @}

  bool initCondIsStateError() const {
    assert(init_cond_is_state_error_ == checkInitCondIsStateError());
    return init_cond_is_state_error_;
  }

  /// @brief Add a stage to the control problem.
  void addStage(const xyz::polymorphic<StageModel> &stage);

  /// @brief Get initial state constraint.
  ConstVectorRef getInitState() const {
    if (!initCondIsStateError()) {
      ALIGATOR_RUNTIME_ERROR(
          "Initial condition is not a StateErrorResidual.\n");
    }
    return static_cast<StateErrorResidual const *>(&*init_constraint_)->target_;
  }

  /// @brief Set initial state constraint.
  void setInitState(const ConstVectorRef &x0) {
    if (!initCondIsStateError()) {
      ALIGATOR_RUNTIME_ERROR(
          "Initial condition is not a StateErrorResidual.\n");
    }
    static_cast<StateErrorResidual *>(&*init_constraint_)->target_ = x0;
  }

  /// @brief Add a terminal constraint for the model.
  ALIGATOR_DEPRECATED void addTerminalConstraint(const StageConstraint &cstr);
  /// @copybrief addTerminalConstraint()
  void addTerminalConstraint(const xyz::polymorphic<StageFunction> &func,
                             const xyz::polymorphic<ConstraintSet> &set) {
    this->term_cstrs_.pushBack(func, set);
  }
  /// @brief Remove all terminal constraints.
  void removeTerminalConstraints() { term_cstrs_.clear(); }

  [[nodiscard]] std::size_t numSteps() const;

  /// @brief Rollout the problem costs, constraints, dynamics, stage per stage.
  Scalar evaluate(const std::vector<VectorXs> &xs,
                  const std::vector<VectorXs> &us, Data &prob_data,
                  std::size_t num_threads = 1) const;

  /**
   * @brief Rollout the problem derivatives, stage per stage.
   *
   * @param xs State sequence
   * @param us Control sequence
   * @param prob_data Problem data
   * @param num_threads Number of threads to use
   * @param compute_second_order Whether to compute second-order derivatives
   */
  void computeDerivatives(const std::vector<VectorXs> &xs,
                          const std::vector<VectorXs> &us, Data &prob_data,
                          std::size_t num_threads = 1,
                          bool compute_second_order = true) const;

  /// @brief Pop out the first StageModel and replace by the supplied one;
  /// updates the supplied problem data (TrajOptDataTpl) object.
  void replaceStageCircular(const xyz::polymorphic<StageModel> &model);

  bool checkIntegrity() const;

  /// @brief Set a function to initialize the state trajectory.
  //
  /// The class constructor defaults the strategy function to xs_default_init.
  /// @warning Call this set before the solver's setup().
  /// @tparam Callable Functional type convertible to InitializationStrategy.
  ///
  /// @sa initializeSolution()
  template <typename Callable> void setInitializationStrategy(Callable &&func) {
    static_assert(std::is_convertible_v<Callable, InitializationStrategy>);
    this->xs_init_strategy_ = std::forward<Callable>(func);
  }

  /// @brief Execute the initialization strategy to generate an initial
  /// candidate solution to the problem.
  ///
  /// @sa setInitializationStrategy()
  void initializeSolution(std::vector<VectorXs> &xs,
                          std::vector<VectorXs> &us) const {
    xs_init_strategy_(*this, xs);
    us_default_init(*this, us);
  }

  /// @copydoc initializeSolution()
  void initializeSolution(std::vector<VectorXs> &xs, std::vector<VectorXs> &us,
                          std::vector<VectorXs> &vs,
                          std::vector<VectorXs> &lbdas) const {
    const size_t nsteps = numSteps();
    initializeSolution(xs, us);
    // initialize multipliers...
    vs.resize(nsteps + 1);
    lbdas.resize(nsteps + 1);
    lbdas[0].setZero(init_constraint_->nr);
    for (size_t i = 0; i < nsteps; i++) {
      const StageModelTpl<Scalar> &sm = *stages_[i];
      lbdas[i + 1].setZero(sm.ndx2());
      vs[i].setZero(sm.nc());
    }

    if (!term_cstrs_.empty()) {
      vs[nsteps].setZero(term_cstrs_.totalDim());
    }
  }

  /// @copydoc initializeSolution()
  [[nodiscard]] auto initializeSolution() const {
    std::vector<VectorXs> xs, us, vs, lbdas;
    initializeSolution(xs, us, vs, lbdas);
    return std::make_tuple(std::move(xs), std::move(us), std::move(vs),
                           std::move(lbdas));
  }

private:
  InitializationStrategy xs_init_strategy_;

  // Check if the initial state is a StateErrorResidual.
  // Since this is a costly operation (dynamic_cast), we cache the result.
  bool checkInitCondIsStateError() const;
  bool init_cond_is_state_error_ = false;
};

/// @brief Default-initialize a trajectory to the neutral states for each state
/// space at each stage.
template <typename Scalar>
void xs_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &xs) {
  const std::size_t nsteps = problem.numSteps();
  xs.resize(nsteps + 1);
  if (problem.initCondIsStateError()) {
    xs[0] = problem.getInitState();
  } else {
    if (problem.stages_.size() > 0) {
      xs[0] = problem.stages_[0]->xspace().neutral();
    } else {
      ALIGATOR_RUNTIME_ERROR(
          "The problem should have either a StateErrorResidual as an initial "
          "condition or at least one stage.");
    }
  }
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    xs[i + 1] = sm.xspace_next().neutral();
  }
}

/// @brief Default-initialize a controls trajectory from the neutral element of
/// each control space.
template <typename Scalar>
void us_default_init(const TrajOptProblemTpl<Scalar> &problem,
                     std::vector<typename math_types<Scalar>::VectorXs> &us) {
  const std::size_t nsteps = problem.numSteps();
  us.resize(nsteps);
  for (std::size_t i = 0; i < nsteps; i++) {
    const StageModelTpl<Scalar> &sm = *problem.stages_[i];
    us[i] = sm.uspace().neutral();
  }
}

/// @brief Check the input state-control trajectory is a consistent warm-start
/// for the output.
///
/// @details If the state trajectory @p xs_in is empty, then both states and
/// controls will be reinitialized using the @p problem object's set
/// initialization strategy. Otherwise, if the controls container is empty, they
/// (**only** the controls) will be default-initialized. Finally, if neither
/// are empty, we attempt to assign the given @p xs_in and
/// @p us_in values.
template <typename Scalar>
void check_initial_guess_and_assign(
    const TrajOptProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::VectorOfVectors &xs_in,
    const typename math_types<Scalar>::VectorOfVectors &us_in,
    typename math_types<Scalar>::VectorOfVectors &xs_out,
    typename math_types<Scalar>::VectorOfVectors &us_out) {
  if (xs_in.empty()) {
    problem.initializeSolution(xs_out, us_out);
  } else if (us_in.empty()) {
    us_default_init(problem, us_out);
  } else {
    if (!assign_no_resize(xs_in, xs_out))
      ALIGATOR_RUNTIME_ERROR("warm-start for xs has wrong size!");
    if (!assign_no_resize(us_in, us_out))
      ALIGATOR_RUNTIME_ERROR("warm-start for us has wrong size!");
  }
}

namespace internal {
/// Get dimension of problem's last stage/cost function.
template <typename Scalar>
int problem_last_ndx_helper(const TrajOptProblemTpl<Scalar> &problem) {
  return problem.term_cost_->ndx();
}
} // namespace internal

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct TrajOptProblemTpl<context::Scalar>;
#endif
} // namespace aligator
