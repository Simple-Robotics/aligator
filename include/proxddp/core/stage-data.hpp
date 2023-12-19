/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/clone.hpp"

namespace proxddp {

/// @brief    Data struct for stage models StageModelTpl.
template <typename _Scalar>
struct StageDataTpl : Cloneable<StageDataTpl<_Scalar>> {
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);

  using StageModel = StageModelTpl<Scalar>;
  using CostDataAbstract = CostDataAbstractTpl<Scalar>;
  using StageFunctionData = StageFunctionDataTpl<Scalar>;
  using DynamicsData = DynamicsDataTpl<Scalar>;

  /// Data structs for the functions involved in the constraints.
  std::vector<shared_ptr<StageFunctionData>> constraint_data;
  /// Data for the running costs.
  shared_ptr<CostDataAbstract> cost_data;

  /// @brief    Constructor.
  ///
  /// @details  The constructor initializes or fills in the data members using
  /// move semantics.
  explicit StageDataTpl(const StageModel &stage_model);

  virtual ~StageDataTpl() = default;

  DynamicsData &dyn_data() { return *dynamics_data; }

  const DynamicsData &dyn_data() const { return *dynamics_data; }

  /// @brief Check data integrity.
  virtual void checkData() {
    const char msg[] = "StageData integrity check failed.";
    if (constraint_data.size() == 0) {
      PROXDDP_RUNTIME_ERROR(fmt::format("{} (constraint_data empty)", msg));
    }
    if (cost_data == 0) {
      PROXDDP_RUNTIME_ERROR(fmt::format("{} (cost_data is nullptr)", msg));
    }

    if (dynamics_data == nullptr) {
      PROXDDP_RUNTIME_ERROR(
          fmt::format("{} (constraint_data[0] should be dynamics data)", msg));
    }
  }

protected:
  // Shortcut pointer to dynamics' data.
  shared_ptr<DynamicsData> dynamics_data;
  StageDataTpl() = default;
  virtual StageDataTpl *clone_impl() const override {
    return new StageDataTpl(*this);
  }
};

} // namespace proxddp

#include "proxddp/core/stage-data.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/core/stage-data.txx"
#endif
