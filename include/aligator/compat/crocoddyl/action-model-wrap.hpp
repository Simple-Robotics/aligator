/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/compat/crocoddyl/fwd.hpp"
#include "aligator/compat/crocoddyl/state-wrap.hpp"

#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/stage-model.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/stage-data.hpp"
#include <crocoddyl/core/action-base.hpp>

namespace aligator {
namespace compat {
namespace croc {
template <typename Scalar>
struct NoOpDynamics final : ExplicitDynamicsModelTpl<Scalar> {
  using Base = ExplicitDynamicsModelTpl<Scalar>;
  using DynData = ExplicitDynamicsDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  NoOpDynamics(const xyz::polymorphic<Manifold> &state, const int nu)
      : Base(state, nu) {}

  void forward(const ConstVectorRef &, const ConstVectorRef &,
               DynData &) const override {}

  void dForward(const ConstVectorRef &, const ConstVectorRef &,
                DynData &) const override {}
};

template <typename Scalar>
struct DynamicsDataWrapperTpl final : ExplicitDynamicsDataTpl<Scalar> {
  using Base = ExplicitDynamicsDataTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  explicit DynamicsDataWrapperTpl(const CrocActionModel &action_model)
      : Base((int)action_model.get_state()->get_ndx(),
             (int)action_model.get_nu(),
             (int)action_model.get_state()->get_nx(),
             (int)action_model.get_state()->get_ndx()) {}
};

/// @brief Wraps a crocoddyl::ActionModelAbstract
///
/// This data structure rewires an ActionModel into a StageModel object.
template <typename Scalar>
struct ActionModelWrapperTpl final : StageModelTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageModelTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;
  using CostAbstract = CostAbstractTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateWrapper = StateWrapperTpl<Scalar>;
  using ActionDataWrap = ActionDataWrapperTpl<Scalar>;
  using DynDataWrap = DynamicsDataWrapperTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;

  shared_ptr<CrocActionModel> action_model_;

  explicit ActionModelWrapperTpl(shared_ptr<CrocActionModel> action_model);

  bool hasDynModel() const override { return false; }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                Data &data) const override;

  void computeFirstOrderDerivatives(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    Data &data) const override;

  /// Does nothing for this class.
  void computeSecondOrderDerivatives(const ConstVectorRef & /*x*/,
                                     const ConstVectorRef & /*u*/,
                                     Data & /*data*/) const override {}

  shared_ptr<Data> createData() const override;
};

/**
 * @brief A complicated child class to StageDataTpl which pipes Crocoddyl's data
 * to the right places.
 */
template <typename Scalar> struct ActionDataWrapperTpl : StageDataTpl<Scalar> {
  using Base = StageDataTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using CrocActionData = crocoddyl::ActionDataAbstractTpl<Scalar>;
  using DynamicsDataWrapper = DynamicsDataWrapperTpl<Scalar>;

  shared_ptr<CrocActionData> croc_action_data;

  ActionDataWrapperTpl(const ActionModelWrapperTpl<Scalar> &croc_action_model);

  void checkData();

protected:
  // utility pointer
  shared_ptr<DynamicsDataWrapper> dynamics_data;
  friend ActionModelWrapperTpl<Scalar>;
};

extern template struct ActionModelWrapperTpl<context::Scalar>;
extern template struct ActionDataWrapperTpl<context::Scalar>;

} // namespace croc
} // namespace compat
} // namespace aligator
