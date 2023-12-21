/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

template <typename Scalar>
StageFunctionTpl<Scalar>::StageFunctionTpl(const int ndx1, const int nu,
                                           const int ndx2, const int nr)
    : ndx1(ndx1), nu(nu), ndx2(ndx2), nr(nr) {}

template <typename Scalar>
StageFunctionTpl<Scalar>::StageFunctionTpl(const int ndx, const int nu,
                                           const int nr)
    : StageFunctionTpl(ndx, nu, ndx, nr) {}

template <typename Scalar>
void StageFunctionTpl<Scalar>::computeVectorHessianProducts(
    const ConstVectorRef &, const ConstVectorRef &, const ConstVectorRef &,
    const ConstVectorRef &, Data &) const {}

template <typename Scalar>
shared_ptr<StageFunctionDataTpl<Scalar>>
StageFunctionTpl<Scalar>::createData() const {
  return std::make_shared<Data>(ndx1, nu, ndx2, nr);
}

/* StageFunctionDataTpl */

template <typename Scalar>
StageFunctionDataTpl<Scalar>::StageFunctionDataTpl(const int ndx1, const int nu,
                                                   const int ndx2, const int nr)
    : ndx1(ndx1), nu(nu), ndx2(ndx2), nr(nr), value_(nr), valref_(value_),
      jac_buffer_(nr, nvar), vhp_buffer_(nvar, nvar),
      Jx_(jac_buffer_.leftCols(ndx1)), Ju_(jac_buffer_.middleCols(ndx1, nu)),
      Jy_(jac_buffer_.rightCols(ndx2)),
      Hxx_(vhp_buffer_.topLeftCorner(ndx1, ndx1)),
      Hxu_(vhp_buffer_.topRows(ndx1).middleCols(ndx1, nu)),
      Hxy_(vhp_buffer_.topRightCorner(ndx1, ndx2)),
      Huu_(vhp_buffer_.middleRows(ndx1, nu).middleCols(ndx1, nu)),
      Huy_(vhp_buffer_.middleRows(ndx1, nu).rightCols(ndx2)),
      Hyy_(vhp_buffer_.bottomRightCorner(ndx2, ndx2)) {
  value_.setZero();
  jac_buffer_.setZero();
  vhp_buffer_.setZero();
}

template <typename T>
std::ostream &operator<<(std::ostream &oss,
                         const StageFunctionDataTpl<T> &self) {
  oss << "StageFunctionData { ";
  if (self.ndx1 == self.ndx2) {
    oss << fmt::format("ndx : {:d}", self.ndx1);
    oss << ",  ";
  } else {
    oss << fmt::format("ndx1: {:d}", self.ndx1);
    oss << ",  ";
    oss << fmt::format("ndx2: {:d}", self.ndx2);
    oss << ",  ";
  }
  oss << fmt::format("nu:   {:d}", self.nu);
  oss << ",  ";
  oss << fmt::format("nr:   {:d}", self.nr);
  oss << " }";
  return oss;
}

} // namespace aligator
