#pragma once

#include "aligator/core/unary-function.hpp"

namespace aligator {

template <typename Scalar> struct AngularMomentumDataTpl;

/**
 * @brief This residual returns the angular momentum for a centroidal model
 * with state \f$x = (c, h, L) \f$.
 *
 * @details The residual returns the last three components of the state:
 * \f$r(x) = L - L_r \f$ with \f$ L \f$ angular momentym and \f$ L_r \f$ desired
 * reference for angular momentum.
 */

template <typename _Scalar>
struct AngularMomentumResidualTpl : UnaryFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Data = AngularMomentumDataTpl<Scalar>;

  AngularMomentumResidualTpl(const int ndx, const int nu, const Vector3s &L_ref)
      : Base(ndx, nu, 3), L_ref_(L_ref) {}

  const Vector3s &getReference() const { return L_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &L_new) { L_ref_ = L_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Vector3s L_ref_;
};

template <typename Scalar>
struct AngularMomentumDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  AngularMomentumDataTpl(const AngularMomentumResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/centroidal/angular-momentum.txx"
#endif
