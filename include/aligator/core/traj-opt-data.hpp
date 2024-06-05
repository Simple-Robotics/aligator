/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/stage-data.hpp"

namespace aligator {

/// @brief Problem data struct.
template <typename _Scalar> struct TrajOptDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using ConstraintType = StageConstraintTpl<Scalar>;
  using StageData = StageDataTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  /// Current cost in the TO problem.
  Scalar cost_ = 0.;

  /// init_condition_common_model_container_ Data
  CommonModelDataContainer init_condition_common_model_data_container;
  /// Data for the initial condition.
  shared_ptr<StageFunctionData> init_data;
  /// Data structs for each stage of the problem.
  std::vector<shared_ptr<StageData>> stage_data;
  /// term_common_model_container_ Data
  CommonModelDataContainer term_common_model_data_container;
  /// Terminal cost data.
  shared_ptr<CostData> term_cost_data;
  /// Terminal constraint data.
  std::vector<shared_ptr<StageFunctionData>> term_cstr_data;

  inline std::size_t numSteps() const { return stage_data.size(); }

  TrajOptDataTpl() = default;
  TrajOptDataTpl(const TrajOptProblemTpl<Scalar> &problem);
};

} // namespace aligator

#include "aligator/core/traj-opt-data.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/traj-opt-data.txx"
#endif
