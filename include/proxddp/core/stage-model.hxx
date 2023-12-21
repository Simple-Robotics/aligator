/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/stage-model.hpp"
#include "proxddp/utils/exceptions.hpp"

#include <proxsuite-nlp/modelling/constraints/equality-constraint.hpp>
#include <proxsuite-nlp/modelling/spaces/vector-space.hpp>

namespace aligator {

namespace {

using proxsuite::nlp::VectorSpaceTpl;
template <typename T>
shared_ptr<VectorSpaceTpl<T>> make_vector_space(const int n) {
  return std::make_shared<VectorSpaceTpl<T>>(n);
}

} // namespace

/* StageModelTpl */

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(CostPtr cost, DynamicsPtr dyn_model)
    : xspace_(dyn_model->space_), xspace_next_(dyn_model->space_next_),
      uspace_(make_vector_space<Scalar>(dyn_model->nu)), cost_(cost) {

  if (cost->nu != dyn_model->nu) {
    PROXDDP_RUNTIME_ERROR(fmt::format("Control dimensions cost.nu ({:d}) and "
                                      "dyn_model.nu ({:d}) are inconsistent.",
                                      cost->nu, dyn_model->nu));
  }

  using EqualitySet = proxsuite::nlp::EqualityConstraint<Scalar>;
  constraints_.pushBack(Constraint{dyn_model, std::make_shared<EqualitySet>()});
}

template <typename Scalar>
StageModelTpl<Scalar>::StageModelTpl(ManifoldPtr space, const int nu)
    : xspace_(space), xspace_next_(space),
      uspace_(make_vector_space<Scalar>(nu)) {}

template <typename Scalar> inline int StageModelTpl<Scalar>::numPrimal() const {
  return this->nu() + this->ndx2();
}

template <typename Scalar> inline int StageModelTpl<Scalar>::numDual() const {
  return (int)constraints_.totalDim();
}

template <typename Scalar>
template <typename T>
void StageModelTpl<Scalar>::addConstraint(T &&cstr) {
  const int c_nu = cstr.func->nu;
  if (c_nu != this->nu()) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}", c_nu,
        this->nu()));
  }
  constraints_.pushBack(std::forward<T>(cstr));
}

template <typename Scalar>
void StageModelTpl<Scalar>::addConstraint(FunctionPtr func,
                                          ConstraintSetPtr cstr_set) {
  if (func->nu != this->nu()) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}",
        func->nu, this->nu()));
  }
  constraints_.pushBack(Constraint{func, cstr_set});
}

template <typename Scalar>
void StageModelTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                     const ConstVectorRef &u,
                                     const ConstVectorRef &y,
                                     Data &data) const {
  for (std::size_t j = 0; j < numConstraints(); j++) {
    const Constraint &cstr = constraints_[j];
    cstr.func->evaluate(x, u, y, *data.constraint_data[j]);
  }
  cost_->evaluate(x, u, *data.cost_data);
}

template <typename Scalar>
void StageModelTpl<Scalar>::computeDerivatives(const ConstVectorRef &x,
                                               const ConstVectorRef &u,
                                               const ConstVectorRef &y,
                                               Data &data) const {
  for (std::size_t j = 0; j < numConstraints(); j++) {
    const Constraint &cstr = constraints_[j];
    cstr.func->computeJacobians(x, u, y, *data.constraint_data[j]);
  }
  cost_->computeGradients(x, u, *data.cost_data);
  cost_->computeHessians(x, u, *data.cost_data);
}

template <typename Scalar>
auto StageModelTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<Data>(*this);
}

} // namespace aligator
