/// @file
/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/fwd.hpp"
#include "aligator/core/dynamics.hpp"
#include "aligator/core/constraint.hpp"
#include "aligator/core/clone.hpp"
#include "aligator/core/common-model-data-container.hpp"

namespace aligator {

/// @brief    Data struct for stage models StageModelTpl.
template <typename _Scalar>
struct StageDataTpl : Cloneable<StageDataTpl<_Scalar>> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  using StageModel = StageModelTpl<Scalar>;
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;
  using CommonModelDataContainer = CommonModelDataContainerTpl<Scalar>;

  /// Store all CommonModelData
  CommonModelDataContainer common_model_data_container;
  /// Data structs for the functions involved in the constraints.
  std::vector<shared_ptr<StageFunctionData>> constraint_data;
  /// Data for the running costs.
  shared_ptr<CostDataAbstract> cost_data;
  // Data for the system dynamics.
  shared_ptr<DynamicsData> dynamics_data;

  /// @brief    Constructor.
  ///
  /// @details  The constructor initializes or fills in the data members using
  /// move semantics.
  explicit StageDataTpl(const StageModel &stage_model)
      : common_model_data_container(
            stage_model.common_model_container_.createData()),
        constraint_data(stage_model.numConstraints()),
        cost_data(stage_model.cost_->createData()),
        dynamics_data(stage_model.dynamics_->createData()) {
    const std::size_t nc = stage_model.numConstraints();

    for (std::size_t j = 0; j < nc; j++) {
      const auto &func = stage_model.constraints_[j].func;
      constraint_data[j] = func->createData(common_model_data_container);
    }
  }

  virtual ~StageDataTpl() = default;

  /// @brief Check data integrity.
  virtual void checkData() {
    const char msg[] = "StageData integrity check failed.";
    if (cost_data == nullptr)
      ALIGATOR_RUNTIME_ERROR(
          fmt::format("{} (cost_data cannot be nullptr)", msg));
    if (dynamics_data == nullptr)
      ALIGATOR_RUNTIME_ERROR(
          fmt::format("{} (dynamics_data cannot be nullptr)", msg));
  }

protected:
  StageDataTpl() = default;
  virtual StageDataTpl *clone_impl() const override {
    return new StageDataTpl(*this);
  }
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/core/stage-data.txx"
#endif
