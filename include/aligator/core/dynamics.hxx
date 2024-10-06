/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/dynamics.hpp"

namespace aligator {

template <typename Scalar>
DynamicsModelTpl<Scalar>::DynamicsModelTpl(xyz::polymorphic<Manifold> space,
                                           const int nu)
    : DynamicsModelTpl(space, nu, space) {}

template <typename Scalar>
DynamicsModelTpl<Scalar>::DynamicsModelTpl(xyz::polymorphic<Manifold> space,
                                           const int nu,
                                           xyz::polymorphic<Manifold> space2)
    : space_(std::move(space)), space_next_(std::move(space2)),
      ndx1(space_->ndx()), nu(nu), ndx2(space_next_->ndx()) {}

template <typename Scalar>
void DynamicsModelTpl<Scalar>::computeVectorHessianProducts(
    const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
    const ConstVectorRef &, Data &) const {
  // no-op
}

template <typename Scalar>
auto DynamicsModelTpl<Scalar>::createData() const -> shared_ptr<Data> {
  return std::make_shared<Data>(*this);
}

template <typename Scalar>
DynamicsDataTpl<Scalar>::DynamicsDataTpl(const DynamicsModelTpl<Scalar> &model)
    : DynamicsDataTpl(model.ndx1, model.nu, model.ndx2) {}

template <typename Scalar>
DynamicsDataTpl<Scalar>::DynamicsDataTpl(const int ndx1, const int nu,
                                         const int ndx2)
    : ndx1(ndx1), nu(nu), ndx2(ndx2), value_(ndx2), valref_(value_),
      jac_buffer_(ndx2, nvar), Jx_(jac_buffer_.leftCols(ndx1)),
      Ju_(jac_buffer_.middleCols(ndx1, nu)), Jy_(jac_buffer_.rightCols(ndx2)),
      Hxx_(ndx1, ndx1), Hxu_(ndx1, nu), Hxy_(ndx1, ndx2), Huu_(nu, nu),
      Huy_(nu, ndx2), Hyy_(ndx2, ndx2) {
  value_.setZero();
  jac_buffer_.setZero();
  Hxx_.setZero();
  Hxu_.setZero();
  Hxy_.setZero();
  Huu_.setZero();
  Huy_.setZero();
  Hyy_.setZero();
}

} // namespace aligator
