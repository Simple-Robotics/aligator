/// @file solver-util.hpp
/// @brief Common utilities for all solvers.
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/traj-opt-problem.hpp"

namespace aligator {

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
void check_trajectory_and_assign(
    const TrajOptProblemTpl<Scalar> &problem,
    const typename math_types<Scalar>::VectorOfVectors &xs_in,
    const typename math_types<Scalar>::VectorOfVectors &us_in,
    typename math_types<Scalar>::VectorOfVectors &xs_out,
    typename math_types<Scalar>::VectorOfVectors &us_out) {
  if (xs_in.empty()) {
    problem.executeInitialization(xs_out, us_out);
  } else if (us_in.empty()) {
    us_default_init(problem, us_out);
  } else {
    if (!assign_no_resize(xs_in, xs_out))
      ALIGATOR_RUNTIME_ERROR("warm-start for xs has wrong size!");
    if (!assign_no_resize(us_in, us_out))
      ALIGATOR_RUNTIME_ERROR("warm-start for us has wrong size!");
  }
}

} // namespace aligator
