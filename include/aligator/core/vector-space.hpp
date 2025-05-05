/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "manifold-base.hpp"

#include <type_traits>

namespace aligator {

/// @brief    Standard Euclidean vector space.
template <typename _Scalar, int _Dim>
struct VectorSpaceTpl : public ManifoldAbstractTpl<_Scalar> {
  using Scalar = _Scalar;
  static constexpr int Dim = _Dim;
  using Base = ManifoldAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  int dim_;

  /// @brief    Default constructor where the dimension is supplied.
  VectorSpaceTpl(const int dim) : Base(), dim_(dim) {
    static_assert(
        Dim == Eigen::Dynamic,
        "This constructor is only valid if the dimension is dynamic.");
  }

  /// @brief    Default constructor without arguments.
  ///
  /// @details  This constructor is disabled if the dimension is not known at
  /// compile time.
  template <int N = Dim,
            typename = typename std::enable_if_t<N != Eigen::Dynamic>>
  VectorSpaceTpl() : Base(), dim_(Dim) {}

  inline int nx() const { return dim_; }
  inline int ndx() const { return dim_; }

  /// Build from VectorSpaceTpl of different dimension
  template <int OtherDim>
  VectorSpaceTpl(const VectorSpaceTpl<Scalar, OtherDim> &other)
      : Base(), dim_(other.dim_) {
    static_assert((Dim == OtherDim) || (Dim == Eigen::Dynamic));
  }

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

  void JintegrateTransport(const ConstVectorRef &, const ConstVectorRef &,
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
      Jout = -MatrixXs::Identity(ndx(), ndx());
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
