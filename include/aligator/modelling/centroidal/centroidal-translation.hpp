#pragma once

#include "aligator/core/unary-function.hpp"

namespace aligator {

template <typename Scalar> struct CentroidalCoMDataTpl;

template <typename _Scalar>
struct CentroidalCoMResidualTpl : UnaryFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Data = CentroidalCoMDataTpl<Scalar>;

  CentroidalCoMResidualTpl(const int ndx, const int nu, const Vector3s &p_ref)
      : Base(ndx, nu, 3), p_ref_(p_ref) {}

  const Vector3s &getReference() const { return p_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &p_new) { p_ref_ = p_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Vector3s p_ref_;
};

template <typename Scalar>
struct CentroidalCoMDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  CentroidalCoMDataTpl(const CentroidalCoMResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/centroidal/centroidal-translation.txx"
#endif
