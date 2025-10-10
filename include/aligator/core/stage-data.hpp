/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, 2024-2025 INRIA
#pragma once

#include "aligator/context.hpp"

namespace aligator {

/// @brief    Data struct for stage models StageModelTpl.
template <typename _Scalar> struct StageDataTpl {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using StageModel = StageModelTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using DynamicsData = ExplicitDynamicsDataTpl<Scalar>;

  /// Data structs for the functions involved in the constraints.
  std::vector<shared_ptr<StageFunctionData>> constraint_data;
  /// Data for the running costs.
  shared_ptr<CostData> cost_data;
  // Data for the system dynamics.
  shared_ptr<DynamicsData> dynamics_data;

  /// @brief    Constructor.
  ///
  /// @details  The constructor initializes or fills in the data members using
  /// move semantics.
  explicit StageDataTpl(const StageModel &stage_model);

  virtual ~StageDataTpl() = default;

  /// @brief Check data integrity.
  virtual void checkData() {
    constexpr std::string_view msg = "StageData integrity check failed.";
    if (cost_data == nullptr)
      ALIGATOR_RUNTIME_ERROR("{} (cost_data cannot be nullptr)", msg);
    if (dynamics_data == nullptr)
      ALIGATOR_RUNTIME_ERROR("{} (dynamics_data cannot be nullptr)", msg);
  }
};
#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
extern template struct StageDataTpl<context::Scalar>;
#endif
} // namespace aligator
