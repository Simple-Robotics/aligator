/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "./log-residual-cost.hpp"

namespace aligator {

template <typename Scalar>
LogResidualCostTpl<Scalar>::LogResidualCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const ConstVectorRef &scale)
    : Base(space, function->nu), barrier_weights_(scale), residual_(function) {
  if (scale.size() != function->nr) {
    ALIGATOR_RUNTIME_ERROR(fmt::format(
        "scale argument dimension ({:d}) != function codimension ({:d})",
        scale.size(), function->nr));
  }
  bool negs = (scale.array() <= 0.0).any();
  if (negs) {
    ALIGATOR_RUNTIME_ERROR("scale coefficients must be > 0.");
  }
}

template <typename Scalar>
LogResidualCostTpl<Scalar>::LogResidualCostTpl(
    xyz::polymorphic<Manifold> space, xyz::polymorphic<StageFunction> function,
    const Scalar scale)
    : LogResidualCostTpl(space, function,
                         VectorXs::Constant(function->nr, scale)) {}

template <typename Scalar>
void LogResidualCostTpl<Scalar>::evaluate(const ConstVectorRef &x,
                                          const ConstVectorRef &u,
                                          CostDataAbstract &data) const {
  Data &d = static_cast<Data &>(data);
  residual_->evaluate(x, u, *d.residual_data);
  d.value_ =
      barrier_weights_.dot(d.residual_data->value_.array().log().matrix());
}

template <typename Scalar>
void LogResidualCostTpl<Scalar>::computeGradients(
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
    d.grad_.noalias() += barrier_weights_(i) * g_i / v(i);
  }
}

template <typename Scalar>
void LogResidualCostTpl<Scalar>::computeHessians(const ConstVectorRef &,
                                                 const ConstVectorRef &,
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
    d.hess_.noalias() +=
        barrier_weights_(i) * (g_i.transpose() * g_i) / (v(i) * v(i));
  }
}
} // namespace aligator
