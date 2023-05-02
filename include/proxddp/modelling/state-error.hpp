#pragma once

#include "proxddp/core/function-abstract.hpp"
#include "proxddp/core/unary-function.hpp"
#include <proxnlp/modelling/spaces/vector-space.hpp>

namespace proxddp {

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
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar);
  using Data = FunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VectorSpace = proxnlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  shared_ptr<Manifold> space_;
  VectorXs target_;

  StateOrControlErrorResidual(const shared_ptr<Manifold> &xspace, const int nu,
                              const ConstVectorRef &target)
      : Base(xspace->ndx(), nu, xspace->ndx()), space_(xspace),
        target_(target) {}

  void evaluate(const ConstVectorRef &x, Data &data) const override {
    space_->difference(target_, x, data.value_);
  }

  void computeJacobians(const ConstVectorRef &x, Data &data) const override {
    space_->Jdifference(target_, x, data.Jx_, 1);
  }
};

template <typename _Scalar, unsigned int arg>
struct StateOrControlErrorResidual : StageFunctionTpl<_Scalar> {
  static_assert(arg > 0 && arg <= 2, "arg value must be 1 or 2!");
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using Data = FunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VectorSpace = proxnlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  shared_ptr<Manifold> space_;
  VectorXs target_;

  /// @brief Constructor using the state space, control dimension and state
  /// target.
  template <unsigned int N = arg, typename = std::enable_if_t<N == 2>>
  StateOrControlErrorResidual(const shared_ptr<Manifold> &xspace, const int nu,
                              const ConstVectorRef &target);

  /// @brief Constructor using the state space dimension, control manifold and
  ///        control target.
  template <unsigned int N = arg, typename = std::enable_if_t<N == 1>>
  StateOrControlErrorResidual(const int ndx, const shared_ptr<Manifold> &uspace,
                              const ConstVectorRef &target)
      : Base(ndx, uspace->nx(), uspace->ndx()), space_(uspace),
        target_(target) {}

  /// @brief Constructor using state space and control space dimensions,
  ///        the control space is assumed to be Euclidean.
  template <unsigned int N = arg, typename = std::enable_if_t<N == 1>>
  StateOrControlErrorResidual(const int ndx, const ConstVectorRef &target)
      : StateOrControlErrorResidual(
            ndx, std::make_shared<VectorSpace>(target.size()), target) {}

  template <unsigned int N = arg, typename = std::enable_if_t<N == 1>>
  StateOrControlErrorResidual(const int ndx, const int nu)
      : Base(ndx, nu, ndx, nu), space_(std::make_shared<VectorSpace>(nu)),
        target_(space_->neutral()) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const {
    switch (arg) {
    case 1:
      space_->difference(target_, u, data.value_);
      break;
    case 2:
      space_->difference(target_, y, data.value_);
      break;
    default:
      break;
    }
  }

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const {
    switch (arg) {
    case 1:
      space_->Jdifference(target_, u, data.Ju_, 1);
      break;
    case 2:
      space_->Jdifference(target_, y, data.Jy_, 1);
      break;
    default:
      break;
    }
  }
};

template <typename Scalar, unsigned int arg>
template <unsigned int N, typename>
StateOrControlErrorResidual<Scalar, arg>::StateOrControlErrorResidual(
    const shared_ptr<Manifold> &xspace, const int nu,
    const ConstVectorRef &target)
    : Base(xspace->ndx(), nu, xspace->ndx()), space_(xspace), target_(target) {}
} // namespace detail

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

} // namespace proxddp

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/state-error.txx"
#endif
