#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

template <typename Scalar> struct AngularAccelerationDataTpl;

template <typename _Scalar>
struct AngularAccelerationResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = AngularAccelerationDataTpl<Scalar>;

  AngularAccelerationResidualTpl(const int ndx, const int nu, const double mass,
                                 const Vector3s gravity)
      : Base(ndx, nu, 3), nk_(nu / 3), mass_(mass), gravity_(gravity) {
    contact_points_ = StdVectorEigenAligned<Vector3s>(nk_, Vector3s::Zero());
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

  StdVectorEigenAligned<Vector3s> contact_points_;

protected:
  const std::size_t nk_;
  const double mass_;
  Vector3s gravity_;
};

template <typename Scalar>
struct AngularAccelerationDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  AngularAccelerationDataTpl(
      const AngularAccelerationResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/centroidal/angular-acceleration.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./angular-acceleration.txx"
#endif
