#pragma once

#include "proxddp/fwd.hpp"

namespace proxddp {

/// @brief  Data for proximal penalty.
template <typename Scalar> struct ProximalDataTpl;

/**
 * @brief   Proximal penalty cost.
 *
 * @details This cost holds const Eigen::Ref references to the proximal targets.
 *          This will be the proximal solver's previous iterates.
 */
template <typename _Scalar>
struct ProximalPenaltyTpl : CostAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);

  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using Base = CostAbstractTpl<Scalar>;
  using CostData = CostDataAbstractTpl<Scalar>;
  using Data = ProximalDataTpl<Scalar>;

  const ManifoldPtr xspace_, uspace_;
  const ConstVectorRef x_ref, u_ref;
  /// Whether to exclude the control term of the penalty. Switch to true e.g.
  /// for terminal node.
  const bool no_ctrl_term;

  ProximalPenaltyTpl(const ManifoldPtr &xspace, const ManifoldPtr &uspace,
                     const ConstVectorRef &xt, const ConstVectorRef &ut,
                     const bool no_ctrl_term)
      : Base(xspace->ndx(), uspace->ndx()), xspace_(xspace), uspace_(uspace),
        x_ref(xt), u_ref(ut), no_ctrl_term(no_ctrl_term) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                CostData &data) const {
    Data &d = static_cast<Data &>(data);
    // x (-) x_ref
    xspace_->difference(x_ref, x, d.dx_);
    d.value_ = 0.5 * d.dx_.squaredNorm();
    if (this->no_ctrl_term)
      return;
    uspace_->difference(u_ref, u, d.du_);
    d.value_ += 0.5 * d.du_.squaredNorm();
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &u,
                        CostData &data) const {
    Data &d = static_cast<Data &>(data);
    xspace_->Jdifference(x_ref, x, d.Jx_, 1);
    d.Lx_ = d.Jx_.transpose() * d.dx_;
    if (this->no_ctrl_term)
      return;
    uspace_->Jdifference(u_ref, u, d.Ju_, 1);
    d.Lu_ = d.Ju_.transpose() * d.du_;
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &data) const {
    Data &d = static_cast<Data &>(data);
    d.Lxx_ = d.Jx_.transpose() * d.Jx_;
    if (this->no_ctrl_term)
      return;
    d.Luu_ = d.Ju_.transpose() * d.Ju_;
  }

  shared_ptr<CostData> createData() const {
    return std::make_shared<Data>(this);
  }
};

template <typename Scalar>
struct ProximalDataTpl : CostDataAbstractTpl<Scalar> {
  using Base = CostDataAbstractTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using Base::ndx_;
  using Base::nu_;
  VectorXs dx_, du_;
  MatrixXs Jx_, Ju_;
  explicit ProximalDataTpl(const ProximalPenaltyTpl<Scalar> *model)
      : Base(model->xspace_->ndx(), model->uspace_->ndx()), dx_(ndx_), du_(nu_),
        Jx_(ndx_, ndx_), Ju_(nu_, nu_) {
    dx_.setZero();
    du_.setZero();
    Jx_.setZero();
    Ju_.setZero();
  }
};

} // namespace proxddp
