/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/compat/crocoddyl/cost-wrap.hpp"
#include "aligator/compat/crocoddyl/state-wrap.hpp"
#include "aligator/compat/crocoddyl/dynamics-wrap.hpp"

#include "aligator/core/stage-model.hpp"
#include <crocoddyl/core/action-base.hpp>

namespace aligator {
namespace compat {
namespace croc {

/**
 * @brief Wraps a crocoddyl::ActionModelAbstract
 *
 * This data structure rewires an ActionModel into a StageModel object.
 */
template <typename Scalar>
struct ActionModelWrapperTpl : StageModelTpl<Scalar> {
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageModelTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;
  using Dynamics = typename Base::Dynamics;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateWrapper = StateWrapperTpl<Scalar>;
  using ActionDataWrap = ActionDataWrapperTpl<Scalar>;
  using DynDataWrap = DynamicsDataWrapperTpl<Scalar>;

  boost::shared_ptr<CrocActionModel> action_model_;

  explicit ActionModelWrapperTpl(
      boost::shared_ptr<CrocActionModel> action_model);

  bool has_dyn_model() const override { return false; }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const override;

  void computeFirstOrderDerivatives(const ConstVectorRef &x,
                                    const ConstVectorRef &u,
                                    const ConstVectorRef &y,
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
template <typename Scalar>
struct ActionDataWrapperTpl : public StageDataTpl<Scalar> {
  using Base = StageDataTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using CrocActionData = crocoddyl::ActionDataAbstractTpl<Scalar>;
  using DynamicsDataWrapper = DynamicsDataWrapperTpl<Scalar>;

  boost::shared_ptr<CrocActionData> croc_action_data;

  ActionDataWrapperTpl(
      const boost::shared_ptr<CrocActionModel> &croc_action_model);

  void checkData();

protected:
  // utility pointer
  shared_ptr<DynamicsDataWrapper> dynamics_data;
  friend ActionModelWrapperTpl<Scalar>;
};

} // namespace croc
} // namespace compat
} // namespace aligator

#include "aligator/compat/crocoddyl/action-model-wrap.hxx"
