/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/stage-model.hpp"
#include "aligator/utils/exceptions.hpp"

#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace aligator {

namespace {

using proxsuite::nlp::VectorSpaceTpl;
template <typename T> auto make_vector_space(const int n) {
  return std::make_shared<VectorSpaceTpl<T>>(n);
}

} // namespace

/* StageModelTpl */

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(CostPtr cost, DynamicsPtr dynamics)
    : xspace_(dynamics->space_), xspace_next_(dynamics->space_next_),
      uspace_(make_vector_space<Scalar>(dynamics->nu)), cost_(cost),
      dynamics_(dynamics) {

  if (cost->nu != dynamics->nu) {
    ALIGATOR_RUNTIME_ERROR(fmt::format("Control dimensions cost.nu ({:d}) and "
                                       "dynamics.nu ({:d}) are inconsistent.",
                                       cost->nu, dynamics->nu));
  }
}

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(ManifoldPtr space, const int nu)
    : xspace_(space), xspace_next_(space),
      uspace_(make_vector_space<Scalar>(nu)) {}

template <typename Scalar>
template <typename Cstr>
void StageModelTpl<Scalar>::addConstraint(Cstr &&cstr) {
  const int c_nu = cstr.func->nu;
  if (c_nu != this->nu()) {
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}", c_nu,
        this->nu()));
  }
  constraints_.pushBack(std::forward<Cstr>(cstr));
}

template <typename Scalar>
void StageModelTpl<Scalar>::addConstraint(FunctionPtr func,
                                          ConstraintSetPtr cstr_set) {
  if (func->nu != this->nu()) {
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}",
        func->nu, this->nu()));
  }
  constraints_.pushBack(Constraint{func, cstr_set});
}

template <typename Scalar> void StageModelTpl<Scalar>::configure() const {
  // Create and configure builder
  for (std::size_t j = 0; j < numConstraints(); j++) {
    const Constraint &cstr = constraints_[j];
    cstr.func->configure(common_model_builder_container_);
  }
  cost_->configure(common_model_builder_container_);
  // TODO find a way to not store the builder container

  // Create common_container_
  common_model_container_ =
      common_model_builder_container_.createCommonModelContainer();
}

template <typename Scalar>
void StageModelTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                     const ConstVectorRef &u,
                                     const ConstVectorRef &y,
                                     Data &data) const {
  common_model_container_.evaluate(x, u, data.common_model_data_container);
  dynamics_->evaluate(x, u, y, *data.dynamics_data);
  for (std::size_t j = 0; j < numConstraints(); j++) {
    const Constraint &cstr = constraints_[j];
    cstr.func->evaluate(x, u, y, *data.constraint_data[j]);
  }
  cost_->evaluate(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeFirstOrderDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, const ConstVectorRef &y,
    Data &data) const {
  common_model_container_.computeGradients(x, u,
                                           data.common_model_data_container);
  dynamics_->computeJacobians(x, u, y, *data.dynamics_data);
  for (std::size_t j = 0; j < numConstraints(); j++) {
    const Constraint &cstr = constraints_[j];
    cstr.func->computeJacobians(x, u, y, *data.constraint_data[j]);
  }
  cost_->computeGradients(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeSecondOrderDerivatives(
    const ConstVectorRef &x, const ConstVectorRef &u, Data &data) const {
  common_model_container_.computeHessians(x, u,
                                          data.common_model_data_container);
  cost_->computeHessians(x, u, *data.cost_data);
}

template <typename Scalar>
auto StageModelTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<Data>(*this);
}

} // namespace aligator
