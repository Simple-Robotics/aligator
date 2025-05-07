/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"
#include <fmt/format.h>

namespace aligator {

template <typename Scalar>
StageFunctionTpl<Scalar>::StageFunctionTpl(const int ndx, const int nu,
                                           const int nr)
    : ndx1(ndx)
    , nu(nu)
    , nr(nr) {}

template <typename Scalar>
void StageFunctionTpl<Scalar>::computeVectorHessianProducts(
    const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
    Data &) const {}

template <typename Scalar>
shared_ptr<StageFunctionDataTpl<Scalar>>
StageFunctionTpl<Scalar>::createData() const {
  return std::make_shared<Data>(*this);
}

/* StageFunctionDataTpl */

template <typename Scalar>
StageFunctionDataTpl<Scalar>::StageFunctionDataTpl(const int ndx1, const int nu,
                                                   const int nr)
    : ndx1(ndx1)
    , nu(nu)
    , nr(nr)
    , value_(nr)
    , valref_(value_)
    , jac_buffer_(nr, nvar)
    , vhp_buffer_(nvar, nvar)
    , Jx_(jac_buffer_.leftCols(ndx1))
    , Ju_(jac_buffer_.middleCols(ndx1, nu))
    , Hxx_(vhp_buffer_.topLeftCorner(ndx1, ndx1))
    , Hxu_(vhp_buffer_.topRows(ndx1).middleCols(ndx1, nu))
    , Huu_(vhp_buffer_.middleRows(ndx1, nu).middleCols(ndx1, nu)) {
  value_.setZero();
  jac_buffer_.setZero();
  vhp_buffer_.setZero();
}

template <typename Scalar>
StageFunctionDataTpl<Scalar>::StageFunctionDataTpl(
    const StageFunctionTpl<Scalar> &model)
    : StageFunctionDataTpl(model.ndx1, model.nu, model.nr) {}

template <typename T>
std::ostream &operator<<(std::ostream &oss,
                         const StageFunctionDataTpl<T> &self) {
  oss << "StageFunctionData { ";
  oss << fmt::format("ndx : {:d}", self.ndx1);
  oss << ",  ";
  oss << fmt::format("nu:   {:d}", self.nu);
  oss << ",  ";
  oss << fmt::format("nr:   {:d}", self.nr);
  oss << " }";
  return oss;
}

} // namespace aligator
