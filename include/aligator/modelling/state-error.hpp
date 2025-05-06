#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/core/unary-function.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"

namespace aligator {

namespace detail {

/// @brief Residual \f$r(z) = z \ominus z_{tar} \f$
/// @details The arg parameter decides with respect to which the error
/// computation operates -- state `x` or control `u`.. We use SFINAE to enable
/// or disable the relevant constructors.
template <typename _Scalar, unsigned int arg>
struct StateOrControlErrorResidual;

/// @brief Pure state residual.
template <typename _Scalar>
struct StateOrControlErrorResidual<_Scalar, 0> : UnaryFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using Data = StageFunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VectorSpace = VectorSpaceTpl<Scalar, Eigen::Dynamic>;
  using PolyManifold = xyz::polymorphic<Manifold>;

  PolyManifold space_;
  VectorXs target_;

  StateOrControlErrorResidual(const PolyManifold &xspace, const int nu,
                              const ConstVectorRef &target)
      : Base(xspace->ndx(), nu, xspace->ndx()), space_(xspace),
        target_(target) {
    validate();
  }

  template <typename U, typename std::enable_if_t<
                            is_polymorphic_of_v<Manifold, U>, int> = 0>
  StateOrControlErrorResidual(U &&xspace, const int nu,
                              const ConstVectorRef &target)
      : Base(xspace.ndx(), nu, xspace.ndx()), space_{std::forward<U>(xspace)},
        target_(target) {
    validate();
  }

  void evaluate(const ConstVectorRef &x, Data &data) const override {
    space_->difference(target_, x, data.value_);
  }

  void computeJacobians(const ConstVectorRef &x, Data &data) const override {
    space_->Jdifference(target_, x, data.Jx_, 1);
  }

protected:
  void validate() const {
    if (!space_->isNormalized(target_)) {
      ALIGATOR_RUNTIME_ERROR(
          "Target parameter invalid (not a viable element of state manifold.)");
    }
  }
};

template <typename _Scalar, unsigned int arg>
struct StateOrControlErrorResidual : StageFunctionTpl<_Scalar> {
  static_assert(arg == 1, "arg value must be 1");
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = StageFunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VectorSpace = VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  xyz::polymorphic<Manifold> space_;
  VectorXs target_;

  /// @brief Constructor using the state space dimension, control manifold and
  ///        control target.
  template <typename U,
            typename = std::enable_if_t<!is_polymorphic_of_v<Manifold, U>>>
  StateOrControlErrorResidual(const int ndx, U &&uspace,
                              const ConstVectorRef &target)
      : Base(ndx, uspace.nx(), uspace.ndx()), space_(std::forward<U>(uspace)),
        target_(target) {
    validate();
  }

  StateOrControlErrorResidual(const int ndx,
                              const xyz::polymorphic<Manifold> &uspace,
                              const ConstVectorRef &target)
      : Base(ndx, uspace->nx(), uspace->ndx()), space_(uspace),
        target_(target) {
    validate();
  }

  StateOrControlErrorResidual(const int ndx, const int nu)
      : Base(ndx, nu, nu), space_(VectorSpace(nu)), target_(space_->neutral()) {
  }

  /// @brief Constructor using state space and control space dimensions,
  ///        the control space is assumed to be Euclidean.
  StateOrControlErrorResidual(const int ndx, const ConstVectorRef &target)
      : StateOrControlErrorResidual(ndx, (int)target.size()) {
    target_ = target;
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                Data &data) const override {
    switch (arg) {
    case 1:
      space_->difference(target_, u, data.value_);
      break;
    default:
      break;
    }
  }

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &u,
                        Data &data) const override {
    switch (arg) {
    case 1:
      space_->Jdifference(target_, u, data.Ju_, 1);
      break;
    default:
      break;
    }
  }

protected:
  void validate() const {
    if (!space_->isNormalized(target_)) {
      ALIGATOR_RUNTIME_ERROR(
          "Target parameter invalid (not a viable element of state manifold.)");
    }
  }
};

} // namespace detail

/// \brief State error \f$x \ominus x_\text{ref}\f$.
///
/// This can be used, by using the manifold neutral element as the reference,
/// to define constraints on the state (e.g. joint position or velocity limits).
template <typename Scalar>
struct StateErrorResidualTpl : detail::StateOrControlErrorResidual<Scalar, 0> {
  using Base = detail::StateOrControlErrorResidual<Scalar, 0>;
  using Base::Base;
};

template <typename Scalar>
struct ControlErrorResidualTpl
    : detail::StateOrControlErrorResidual<Scalar, 1> {
  using Base = detail::StateOrControlErrorResidual<Scalar, 1>;
  using Base::Base;
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/state-error.txx"
#endif
