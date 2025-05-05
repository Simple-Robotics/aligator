/// @file
/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, 2022-2025 INRIA
#pragma once

#include "aligator/core/manifold-base.hpp"

#include <pinocchio/multibody/liegroup/liegroup-base.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>

namespace aligator {

namespace pin = pinocchio;

/// Type trait. Indicates whether @tparam G is derived from
/// pinocchio::LieGroupBase.
template <typename G>
using is_pinocchio_lie_group = std::is_base_of<pin::LieGroupBase<G>, G>;

/// @brief  Wrap a Pinocchio Lie group into a ManifoldAbstractTpl object.
///
template <typename _LieGroup>
struct PinocchioLieGroup
    : public ManifoldAbstractTpl<typename _LieGroup::Scalar> {
public:
  using LieGroup = _LieGroup;
  using Scalar = typename LieGroup::Scalar;
  using Base = ManifoldAbstractTpl<Scalar>;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  static_assert(is_pinocchio_lie_group<LieGroup>::value,
                "LieGroup template argument should be a subclass of "
                "pinocchio::LieGroupBase.");

  LieGroup lg_;
  PinocchioLieGroup() {}
  PinocchioLieGroup(const LieGroup &lg) : lg_(lg) {}
  PinocchioLieGroup(LieGroup &&lg) : lg_(std::move(lg)) {}
  PinocchioLieGroup(const PinocchioLieGroup &lg) = default;
  PinocchioLieGroup(PinocchioLieGroup &&lg) = default;

  template <typename... Args>
  PinocchioLieGroup(Args &&...args) : lg_(std::forward<Args>(args)...) {}

  operator LieGroup() { return lg_; }

  inline int nx() const { return lg_.nq(); }
  inline int ndx() const { return lg_.nv(); }

  bool isNormalized(const ConstVectorRef &x) const {
    if (x.size() < nx())
      return false;
    return lg_.isNormalized(x);
  }

protected:
  /// \name Implementations

  void integrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                      VectorRef out) const {
    lg_.integrate(x, v, out);
  }

  void difference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                       VectorRef vout) const {
    lg_.difference(x0, x1, vout);
  }

  void Jintegrate_impl(const ConstVectorRef &x, const ConstVectorRef &v,
                       MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      lg_.dIntegrate_dq(x, v, Jout);
      break;
    case 1:
      lg_.dIntegrate_dv(x, v, Jout);
      break;
    }
  }

  void JintegrateTransport_impl(const ConstVectorRef &x,
                                const ConstVectorRef &v, MatrixRef Jout,
                                int arg) const {
    lg_.dIntegrateTransport(x, v, Jout, pin::ArgumentPosition(arg));
  }

  void Jdifference_impl(const ConstVectorRef &x0, const ConstVectorRef &x1,
                        MatrixRef Jout, int arg) const {
    switch (arg) {
    case 0:
      lg_.dDifference(x0, x1, Jout, pin::ARG0);
      break;
    case 1:
      lg_.dDifference(x0, x1, Jout, pin::ARG1);
      break;
    }
  }

  virtual void interpolate_impl(const ConstVectorRef &x0,
                                const ConstVectorRef &x1, const Scalar &u,
                                VectorRef out) const {
    lg_.interpolate(x0, x1, u, out);
  }

  virtual void neutral_impl(VectorRef out) const { out = lg_.neutral(); }

  virtual void rand_impl(VectorRef out) const { out = lg_.random(); }
};

template <int D, typename Scalar>
using SETpl = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<D, Scalar>>;

template <int D, typename Scalar>
using SOTpl = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<D, Scalar>>;

} // namespace aligator
