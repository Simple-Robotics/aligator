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
inline constexpr bool is_pinocchio_lie_group =
    std::is_base_of_v<pin::LieGroupBase<G>, G>;

/// @brief  Wrap a Pinocchio Lie group into a ManifoldAbstractTpl object.
///
template <typename G, std::enable_if_t<is_pinocchio_lie_group<G>> * = nullptr>
struct PinocchioLieGroup : ManifoldAbstractTpl<typename G::Scalar> {
public:
  using LieGroup = G;
  using Scalar = typename LieGroup::Scalar;
  using Base = ManifoldAbstractTpl<Scalar>;
  using Base::ndx;
  using Base::nx;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);

  LieGroup lg_;

  PinocchioLieGroup()
      : Base(0, 0) {
    this->nx_ = lg_.nq();
    this->ndx_ = lg_.nv();
  }
  PinocchioLieGroup(const LieGroup &lg)
      : Base(lg.nq(), lg.nv())
      , lg_(lg) {}
  PinocchioLieGroup(const PinocchioLieGroup &lg) = default;
  PinocchioLieGroup(PinocchioLieGroup &&lg) = default;

  template <typename... Args>
  PinocchioLieGroup(Args &&...args)
      : Base(0, 0)
      , lg_(std::forward<Args>(args)...) {
    this->nx_ = lg_.nq();
    this->ndx_ = lg_.nv();
  }

  operator LieGroup() const { return lg_; }

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

template <typename G> PinocchioLieGroup(const G &) -> PinocchioLieGroup<G>;

template <int D, typename Scalar>
using SETpl = PinocchioLieGroup<pin::SpecialEuclideanOperationTpl<D, Scalar>>;

template <int D, typename Scalar>
using SOTpl = PinocchioLieGroup<pin::SpecialOrthogonalOperationTpl<D, Scalar>>;

} // namespace aligator
