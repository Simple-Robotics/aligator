/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/core/stage-model.hpp"
#include "proxddp/utils/exceptions.hpp"

#include <proxnlp/modelling/constraints/equality-constraint.hpp>
#include <proxnlp/modelling/spaces/vector-space.hpp>

namespace proxddp {

namespace {

using proxnlp::VectorSpaceTpl;
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
                                      "dyn_model.nu ({;d}) are inconsistent.",
                                      cost->nu, dyn_model->nu));
  }

  using EqualitySet = proxnlp::EqualityConstraint<Scalar>;
  constraints_.push_back(
      Constraint{dyn_model, std::make_shared<EqualitySet>()});
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
  constraints_.push_back(std::forward<T>(cstr));
}

template <typename Scalar>
void StageModelTpl<Scalar>::addConstraint(FunctionPtr func,
                                          ConstraintSetPtr cstr_set) {
  if (func->nu != this->nu()) {
    PROXDDP_RUNTIME_ERROR(fmt::format(
        "Function has the wrong dimension for u: got {:d}, expected {:d}",
        func->nu, this->nu()));
  }
  constraints_.push_back(Constraint{func, cstr_set});
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
shared_ptr<StageDataTpl<Scalar>> StageModelTpl<Scalar>::createData() const {
  return std::make_shared<Data>(*this);
}

template <typename Scalar>
std::ostream &operator<<(std::ostream &oss,
                         const StageModelTpl<Scalar> &stage) {
  oss << "StageModel { ";
  if (stage.ndx1() == stage.ndx2()) {
    oss << "ndx: " << stage.ndx1() << ", "
        << "nu:  " << stage.nu();
  } else {
    oss << "ndx1:" << stage.ndx1() << ", "
        << "nu:  " << stage.nu() << ", "
        << "ndx2:" << stage.ndx2();
  }

  if (stage.numConstraints() > 0) {
    oss << ", ";
    oss << "nc: " << stage.numConstraints();
  }

  oss << " }";
  return oss;
}

/* StageDataTpl */

template <typename Scalar>
StageDataTpl<Scalar>::StageDataTpl(const StageModel &stage_model)
    : constraint_data(stage_model.numConstraints()),
      cost_data(stage_model.cost_->createData()) {
  using Function = StageFunctionTpl<Scalar>;
  const std::size_t nc = stage_model.numConstraints();
  constraint_data.reserve(nc);
  for (std::size_t j = 0; j < nc; j++) {
    const shared_ptr<Function> &func = stage_model.constraints_[j].func;
    constraint_data[j] = func->createData();
  }
}
} // namespace proxddp
