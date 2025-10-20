/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/manifold-base.hpp"

#include <type_traits>

namespace aligator {

/// @brief    Standard Euclidean vector space.
template <typename _Scalar, int _Dim>
struct VectorSpaceTpl : ManifoldAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  static constexpr int Dim = _Dim;
  using Base = ManifoldAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base::ndx;
  using Base::nx;

  /// @brief    Default constructor where the dimension is supplied.
  explicit VectorSpaceTpl(const int dim)
      : Base(dim, dim) {
    static_assert(
        Dim == Eigen::Dynamic,
        "This constructor is only valid if the dimension is dynamic.");
  }

  int dim() const { return this->nx_; }

  /// @brief    Default constructor without arguments.
  ///
  /// @details  This constructor is disabled if the dimension is not known at
  /// compile time.
  template <int N = Dim,
            typename = typename std::enable_if_t<N != Eigen::Dynamic>>
  explicit VectorSpaceTpl()
      : Base(Dim, Dim) {}

  /// Build from VectorSpaceTpl of different dimension
  template <int OtherDim>
  VectorSpaceTpl(const VectorSpaceTpl<Scalar, OtherDim> &other)
      : Base(other.nx_, other.nx_) {
    static_assert((Dim == OtherDim) || (Dim == Eigen::Dynamic));
  }

protected:
  /// @name implementations

  /* Integrate */

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {
    out = x + v;
  }

  void Jintegrate_impl(const ConstVectorRef &, const ConstVectorRef &,
                       MatrixRef Jout, int) const {
    Jout.setIdentity();
  }

  void JintegrateTransport_impl(const ConstVectorRef &, const ConstVectorRef &,
                                MatrixRef, int) const {}

  /* Difference */

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef out) const {
    out = x1 - x0;
  }

  void Jdifference_impl(const ConstVectorRef &, const ConstVectorRef &,
                        MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      Jout.setIdentity() *= -1;
      break;
    case 1:
      Jout.setIdentity();
      break;
    default:
      ALIGATOR_DOMAIN_ERROR("Wrong arg value.");
    }
  }

  void interpolate_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        const Scalar &u, VectorRef out) const {
    out = u * x1 + (static_cast<Scalar>(1.) - u) * x0;
  }
};

} // namespace aligator
