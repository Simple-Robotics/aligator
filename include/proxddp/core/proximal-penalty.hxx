#pragma once


namespace proxddp {


template <typename Scalar>
ProximalDataTpl<Scalar>::ProximalDataTpl(const ProximalPenaltyTpl<Scalar> *model)
    : Base(model->ndx, model->nu), dx_(ndx_), du_(nu_),
      Jx_(ndx_, ndx_), Ju_(nu_, nu_) {
  dx_.setZero();
  du_.setZero();
  Jx_.setZero();
  Ju_.setZero();
}

} // namespace proxddp

