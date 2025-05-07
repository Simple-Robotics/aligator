#pragma once

#include "aligator/core/unary-function.hpp"

namespace aligator {

template <typename Scalar> struct LinearMomentumDataTpl;

/**
 * @brief This residual returns the linear momentum for a centroidal model
 * with state \f$x = (c, h, L) \f$.
 *
 * @details The residual returns the middle three components of the state:
 * \f$r(x) = h - h_r \f$ with \f$ h \f$ linear momentym and \f$ h_r \f$ desired
 * reference for linear momentum.
 */

template <typename _Scalar>
struct LinearMomentumResidualTpl : UnaryFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Data = LinearMomentumDataTpl<Scalar>;

  LinearMomentumResidualTpl(const int ndx, const int nu, const Vector3s &h_ref)
      : Base(ndx, nu, 3)
      , h_ref_(h_ref) {}

  const Vector3s &getReference() const { return h_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &h_new) { h_ref_ = h_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Vector3s h_ref_;
};

template <typename Scalar>
struct LinearMomentumDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  LinearMomentumDataTpl(const LinearMomentumResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/centroidal/linear-momentum.txx"
#endif
