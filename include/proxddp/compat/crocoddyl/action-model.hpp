#pragma once

#include "proxddp/core/stage-model.hpp"
#include <crocoddyl/core/action-base.hpp>

#include "proxddp/compat/crocoddyl/cost.hpp"

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar> struct ActionDataWrapper;

/**
 * @brief Wraps a crocoddyl::ActionModelAbstract
 *
 * This data structure rewires an ActionModel into a StageModel object.
 */
template <typename Scalar>
struct ActionModelWrapperTpl : public StageModelTpl<Scalar> {
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageModelTpl<Scalar>;
  using Data = StageDataTpl<Scalar>;
  using ActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  shared_ptr<ActionModel> action_model;

  ActionModelWrapperTpl(const shared_ptr<ActionModel> &action_model)
      : Base(), action_model(action_model) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const {
    ActionDataWrapper<Scalar> &d =
        static_cast<ActionDataWrapper<Scalar> &>(data);
    ActionModel &m = *action_model;
    m.calc(d.croc_data, x, u);
  }

  void computeDerivatives(const ConstVectorRef &x, const ConstVectorRef &u,
                          const ConstVectorRef &y, Data &data) const {
    ActionDataWrapper<Scalar> &d =
        static_cast<ActionDataWrapper<Scalar> &>(data);
    ActionModel &m = *action_model;
    m.calcDiff(d.croc_data, x, u);
  }

  shared_ptr<Data> createData() const {
    auto cd = action_model->createData();
    return std::make_shared<ActionDataWrapper<Scalar>>(*action_model, cd);
  }
};

/**
 * @brief A complicated child class to StageDataTpl which pipes Crocoddyl's data
 * to the right places.
 */
template <typename Scalar>
struct ActionDataWrapper : public StageDataTpl<Scalar> {
  using Base = StageDataTpl<Scalar>;
  using ActionDataAbstract = crocoddyl::ActionDataAbstractTpl<Scalar>;
  boost::shared_ptr<ActionDataAbstract> croc_data;
  explicit ActionDataWrapper(
      const ActionModelWrapperTpl<Scalar> &wrapped_action_model,
      boost::shared_ptr<ActionDataAbstract> action_data)
      : Base(wrapped_action_model), croc_data(action_data) {}
};

} // namespace croc
} // namespace compat
} // namespace proxddp
