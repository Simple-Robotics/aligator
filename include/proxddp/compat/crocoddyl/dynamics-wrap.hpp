/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/compat/crocoddyl/fwd.hpp"
#include "proxddp/core/explicit-dynamics.hpp"
#include <crocoddyl/core/action-base.hpp>

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar>
struct DynamicsDataWrapperTpl : ExplicitDynamicsDataTpl<Scalar> {
  using Base = ExplicitDynamicsDataTpl<Scalar>;
  using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
  explicit DynamicsDataWrapperTpl(const CrocActionModel *action_model)
      : Base((int)action_model->get_state()->get_ndx(),
             (int)action_model->get_nu(),
             (int)action_model->get_state()->get_nx(),
             (int)action_model->get_state()->get_ndx()) {}
};

} // namespace croc
} // namespace compat
} // namespace proxddp
