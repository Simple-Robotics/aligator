/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"

namespace aligator {

/// @brief Problem data struct.
template <typename _Scalar> struct TrajOptDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  /// Current cost in the TO problem.
  Scalar cost_ = 0.;

  /// Data for the initial condition.
  shared_ptr<StageFunctionData> init_data;
  /// Data structs for each stage of the problem.
  std::vector<shared_ptr<StageData>> stage_data;
  /// Terminal cost data.
  shared_ptr<CostData> term_cost_data;
  /// Terminal constraint data.
  std::vector<shared_ptr<StageFunctionData>> term_cstr_data;

  inline std::size_t numSteps() const { return stage_data.size(); }

  TrajOptDataTpl() = default;
  TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem);
};

/// @brief Helper for computing the trajectory cost (from pre-computed problem
/// data).
/// @warning Call TrajOptProblemTpl::evaluate() first!
template <typename Scalar>
Scalar computeTrajectoryCost(const TrajOptDataTpl<Scalar> &problem_data);

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/traj-opt-data.txx"
#endif
