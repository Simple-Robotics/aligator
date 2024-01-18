#pragma once

#include "aligator/core/unary-function.hpp"

namespace aligator {

template <typename Scalar> struct AngularMomentumDataTpl;

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
    return allocate_shared_eigen_aligned<Data>(this);
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

#include "aligator/modelling/centroidal/angular-momentum.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./angular-momentum.txx"
#endif
