/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/cost-wrap.hpp"
#include "proxddp/compat/crocoddyl/state-wrap.hpp"
#include "proxddp/compat/crocoddyl/dynamics-wrap.hpp"

#include "proxddp/core/stage-model.hpp"
#include <crocoddyl/core/action-base.hpp>

#include "proxddp/utils/exceptions.hpp"

namespace proxddp {
namespace compat {
namespace croc {

/**
 * @brief Wraps a crocoddyl::ActionModelAbstract
 *
 * This data structure rewires an ActionModel into a StageModel object.
 */
template <typename Scalar>
struct ActionModelWrapperTpl : StageModelTpl<Scalar> {
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageModelTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;
  using Constraint = StageConstraintTpl<Scalar>;
  using Dynamics = typename Base::Dynamics;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  using StateWrapper = StateWrapperTpl<Scalar>;
  using ActionDataWrap = ActionDataWrapperTpl<Scalar>;
  using DynDataWrap = DynamicsDataWrapperTpl<Scalar>;

  boost::shared_ptr<CrocActionModel> action_model_;

  explicit ActionModelWrapperTpl(
      boost::shared_ptr<CrocActionModel> action_model);

  bool has_dyn_model() const { return false; }
  const Dynamics &dyn_model() const {
    PROXDDP_RUNTIME_ERROR("There is no dyn_model() for this class.");
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const;

  void computeDerivatives(const ConstVectorRef &x, const ConstVectorRef &u,
                          const ConstVectorRef &y, Data &data) const;

  shared_ptr<Data> createData() const;
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
} // namespace proxddp

#include "proxddp/compat/crocoddyl/action-model-wrap.hxx"
