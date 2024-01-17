#pragma once

#include "aligator/core/unary-function.hpp"

namespace aligator {

template <typename Scalar> struct CentroidalAccelerationDataTpl;

template <typename _Scalar>
struct CentroidalAccelerationResidualTpl : UnaryFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Data = CentroidalAccelerationDataTpl<Scalar>;

  CentroidalAccelerationResidualTpl(const int ndx, const int nu,
                                    const double mass, const Vector3s gravity)
      : Base(ndx, nu, 3), nk_(nu / 3), mass_(mass), gravity_(gravity) {}

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

  std::vector<bool> active_contacts_;
  StdVectorEigenAligned<Vector3s> contact_points_;

protected:
  const std::size_t nk_;
  const double mass_;
  Vector3s gravity_;
};

template <typename Scalar>
struct CentroidalAccelerationDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;

  CentroidalAccelerationDataTpl(
      const CentroidalAccelerationResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/centroidal/centroidal-acceleration.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./centroidal-acceleration.txx"
#endif
