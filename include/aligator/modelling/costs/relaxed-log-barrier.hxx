/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./relaxed-log-barrier.hpp"

namespace aligator {

template <typename Scalar>
RelaxedLogBarrierCostTpl<Scalar>::RelaxedLogBarrierCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const ConstVectorRef &scale, const Scalar threshold)
    : Base(space, function->nu), barrier_weights_(scale), residual_(function),
      threshold_(threshold) {
  if (scale.size() != function->nr) {
    ALIGATOR_RUNTIME_ERROR(
        "scale argument dimension ({:d}) != function codimension ({:d})",
        scale.size(), function->nr);
  }
  bool negs = (scale.array() <= 0.0).any();
  if (negs) {
    ALIGATOR_RUNTIME_ERROR("scale coefficients must be > 0.");
  }
  if (threshold_ <= 0.) {
    ALIGATOR_RUNTIME_ERROR("threshold must be > 0.");
  }
}

template <typename Scalar>
RelaxedLogBarrierCostTpl<Scalar>::RelaxedLogBarrierCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const Scalar scale, const Scalar threshold)
    : RelaxedLogBarrierCostTpl(
          space, function, VectorXs::Constant(function->nr, scale), threshold) {
}

template <typename Scalar>
void RelaxedLogBarrierCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                                const ConstVectorRef &u,
                                                CostDataAbstract &data) const {
  Data &d = static_cast<Data &>(data);
  residual_->evaluate(x, u, *d.residual_data);
  d.value_ = 0.;
  const int nrows = residual_->nr;
  for (size_t i = 0; i < nrows; i++) {
    if (d.residual_data->value_[i] < threshold_) {
      Scalar sq = (d.residual_data->value_[i] - 2 * threshold_) / threshold_;
      d.value_ += 0.5 * barrier_weights_(i) * (sq * sq - 1) - log(threshold_);
    } else {
      d.value_ -= barrier_weights_(i) * log(d.residual_data->value_[i]);
    }
  }
}

template <typename Scalar>
void RelaxedLogBarrierCostTpl<Scalar>::computeGradients(
    const ConstVectorRef &x, const ConstVectorRef &u,
    CostDataAbstract &data) const {
  Data &d = static_cast<Data &>(data);
  StageFunctionDataTpl<Scalar> &res_data = *d.residual_data;
  MatrixRef J = res_data.jac_buffer_.leftCols(data.grad_.size());
  residual_->computeJacobians(x, u, res_data);
  d.grad_.setZero();
  VectorXs &v = res_data.value_;
  const int nrows = residual_->nr;
  for (int i = 0; i < nrows; i++) {
    auto g_i = J.row(i);
    if (d.residual_data->value_[i] < threshold_) {
      d.grad_.noalias() += barrier_weights_(i) * g_i * (v(i) - 2 * threshold_) /
                           (threshold_ * threshold_);
    } else {
      d.grad_.noalias() -= barrier_weights_(i) * g_i / v(i);
    }
  }
}

template <typename Scalar>
void RelaxedLogBarrierCostTpl<Scalar>::computeHessians(
    const ConstVectorRef &, const ConstVectorRef &,
    CostDataAbstract &data) const {
  Data &d = static_cast<Data &>(data);
  StageFunctionDataTpl<Scalar> &res_data = *d.residual_data;
  const Eigen::Index size = data.grad_.size();
  d.hess_.setZero();
  MatrixRef J = res_data.jac_buffer_.leftCols(size);
  VectorXs &v = res_data.value_;
  const int nrows = residual_->nr;
  for (int i = 0; i < nrows; i++) {
    auto g_i = J.row(i); // row vector
    if (d.residual_data->value_[i] < threshold_) {
      Scalar sq = (v(i) - 2 * threshold_) / (threshold_ * threshold_);
      d.hess_.noalias() += barrier_weights_(i) * barrier_weights_(i) *
                           (g_i.transpose() * g_i) * sq * sq;
    } else {
      d.hess_.noalias() += barrier_weights_(i) * barrier_weights_(i) *
                           (g_i.transpose() * g_i) / (v(i) * v(i));
    }
  }
}
} // namespace aligator
