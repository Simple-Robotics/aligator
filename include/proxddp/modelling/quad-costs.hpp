#pragma once

#include "proxddp/core/cost-abstract.hpp"

namespace proxddp {

/// @brief Constant cost.
template <typename _Scalar> struct ConstantCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  Scalar value_;
  ConstantCostTpl(const int ndx, const int nu, const Scalar value)
      : Base(ndx, nu), value_(value) {}
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                CostData &data) const {
    data.value_ = value_;
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &) const {}

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &) const {}
};

template <typename Scalar> struct QuadraticCostDataTpl;

/// @brief Euclidean quadratic cost.
template <typename _Scalar> struct QuadraticCostTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;

  using Data = QuadraticCostDataTpl<Scalar>;

  MatrixXs weights_x;
  MatrixXs weights_u;
  VectorXs interp_x;
  VectorXs interp_u;

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u,
                   const ConstVectorRef &interp_x,
                   const ConstVectorRef &interp_u)
      : Base((int)w_x.cols(), (int)w_u.cols()), weights_x(w_x), weights_u(w_u),
        interp_x(interp_x), interp_u(interp_u) {}

  QuadraticCostTpl(const ConstMatrixRef &w_x, const ConstMatrixRef &w_u)
      : QuadraticCostTpl(w_x, w_u, VectorXs::Zero(w_x.cols()),
                         VectorXs::Zero(w_u.cols())) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const {
    Data &d = static_cast<Data &>(data);
    d.w_times_x_ = weights_x * x;
    d.w_times_u_ = weights_u * u;
    data.value_ = Scalar(0.5) * x.dot(d.w_times_x_ + 2 * interp_x) +
                  Scalar(0.5) * u.dot(d.w_times_u_ + 2 * interp_u);
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const {
    Data &d = static_cast<Data &>(data);
    d.Lx_ = d.w_times_x_ + interp_x;
    d.Lu_ = d.w_times_u_ + interp_u;
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &data) const {
    data.Lxx_ = weights_x;
    data.Luu_ = weights_u;
  }

  shared_ptr<CostData> createData() const {
    auto data = std::make_shared<Data>(this->ndx, this->nu);
    return data;
  }
};

template <typename Scalar>
struct QuadraticCostDataTpl : CostDataAbstractTpl<Scalar> {
  using Base = CostDataAbstractTpl<Scalar>;
  using VectorXs = typename Base::VectorXs;
  VectorXs w_times_x_, w_times_u_;

  QuadraticCostDataTpl(const int nx, const int nu)
      : Base(nx, nu), w_times_x_(nx), w_times_u_(nu) {
    w_times_x_.setZero();
    w_times_u_.setZero();
  }
};

} // namespace proxddp
