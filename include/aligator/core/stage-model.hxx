/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/stage-model.hpp"
#include "aligator/core/stage-data.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/tracy.hpp"

namespace aligator {

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(const PolyCost &cost,
                                     const PolyDynamics &dynamics)
    : xspace_(dynamics->space_)
    , xspace_next_(dynamics->space_next_)
    , uspace_(VectorSpaceTpl<Scalar>(dynamics->nu))
    , cost_(cost)
    , dynamics_(dynamics) {

  if (cost->nu != dynamics->nu) {
    ALIGATOR_RUNTIME_ERROR(
        "Inconsistent control dimension cost.nu ({:d}) and dynamics.nu ({:d}).",
        cost->nu, dynamics->nu);
  }
}

template <typename Scalar>
void StageModelTpl<Scalar>::addConstraint(const PolyFunction &func,
                                          const PolyConstraintSet &cstr_set) {
  if (func->nu != this->nu()) {
    ALIGATOR_RUNTIME_ERROR(
        "Function has the wrong dimension for u: got {:d}, expected {:d}",
        func->nu, this->nu());
  }
  constraints_.pushBack(func, cstr_set);
}

template <typename Scalar>
void StageModelTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                     const ConstVectorRef &u,
                                     Data &data) const {
  ALIGATOR_TRACY_ZONE_SCOPED_N("StageModel::evaluate");
  dynamics_->forward(x, u, *data.dynamics_data);
  for (std::size_t j = 0; j < numConstraints(); j++) {
    constraints_.funcs[j]->evaluate(x, u, *data.constraint_data[j]);
  }
  cost_->evaluate(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeFirstOrderDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const {
  ALIGATOR_TRACY_ZONE_SCOPED_N("StageModel::computeFirstOrderDerivatives");
  dynamics_->dForward(x, u, *data.dynamics_data);
  for (std::size_t j = 0; j < numConstraints(); j++) {
    constraints_.funcs[j]->computeJacobians(x, u, *data.constraint_data[j]);
  }
  cost_->computeGradients(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeSecondOrderDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const {
  ALIGATOR_TRACY_ZONE_SCOPED_N("StageModel::computeSecondOrderDerivatives");
  cost_->computeHessians(x, u, *data.cost_data);
}

template <typename Scalar>
auto StageModelTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<Data>(*this);
}

} // namespace aligator
