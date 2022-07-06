#pragma once

#include "proxddp/fwd.hpp"
#include "proxddp/core/function.hpp"
#include <proxnlp/modelling/spaces/vector-space.hpp>

namespace proxddp {

/**
 * @brief Residual \f$r(z) = z \ominus z_{tar} \f$
 * @details The arg parameter decides with respect to which the error
 * computation operates -- state `x` or control `u`.. We use SFINAE to enable or
 * disable the relevant constructors.
 */
template <typename _Scalar, unsigned int arg>
struct StateOrControlErrorResidual : StageFunctionTpl<_Scalar> {
  static_assert(arg <= 2, "arg value must be 0, 1 or 2!");
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Data = FunctionDataTpl<Scalar>;
  using Manifold = ManifoldAbstractTpl<Scalar>;
  using VectorSpace = proxnlp::VectorSpaceTpl<Scalar, Eigen::Dynamic>;

  VectorXs target_;
  shared_ptr<Manifold> space_;

  /// @brief Constructor using the state space, control dimension and state
  /// target.
  template <unsigned int N = arg,
            typename = typename std::enable_if<N == 0 || N == 2>::type>
  StateOrControlErrorResidual(const shared_ptr<Manifold> &xspace, const int nu,
                              const ConstVectorRef &target)
      : StageFunctionTpl<Scalar>(xspace->ndx(), nu, xspace->ndx()),
        target_(target), space_(xspace) {}

  /**
   * @brief Constructor using the state space dimension, control manifold and
   *        control target.
   */
  template <unsigned int N = arg,
            typename = typename std::enable_if<N == 1>::type>
  StateOrControlErrorResidual(const int ndx, const shared_ptr<Manifold> &uspace,
                              const ConstVectorRef &target)
      : StageFunctionTpl<Scalar>(ndx, uspace->nx(), uspace->ndx()),
        target_(target), space_(uspace) {}

  /**
   * @brief Constructor using state space and control space dimensions,
   *        the control space is assumed to be Euclidean.
   */
  template <unsigned int N = arg,
            typename = typename std::enable_if<N == 1>::type>
  StateOrControlErrorResidual(const int ndx, const int nu,
                              const ConstVectorRef &target)
      : StateOrControlErrorResidual(ndx, std::make_shared<VectorSpace>(nu),
                                    target) {}

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, Data &data) const {
    switch (arg) {
    case 0:
      space_->difference(target_, x, data.value_);
      break;
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

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, Data &data) const {
    switch (arg) {
    case 0:
      space_->Jdifference(target_, x, data.Jx_, 1);
      break;
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

template <typename Scalar> struct ErrorResidualData : FunctionDataTpl<Scalar> {
  using VectorXs = typename math_types<Scalar>::VectorXs;
};

template <typename Scalar>
using StateErrorResidual = StateOrControlErrorResidual<Scalar, 0>;

template <typename Scalar>
using ControlErrorResidual = StateOrControlErrorResidual<Scalar, 1>;

} // namespace proxddp
