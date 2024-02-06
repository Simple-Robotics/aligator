#pragma once

#include "aligator/core/function-abstract.hpp"

namespace aligator {

template <typename Scalar> struct CentroidalAccelerationDataTpl;

template <typename _Scalar>
struct CentroidalAccelerationResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = CentroidalAccelerationDataTpl<Scalar>;

  CentroidalAccelerationResidualTpl(
      const int &ndx, const int &nu, const double &mass,
      const Vector3s &gravity,
      const std::vector<std::pair<bool, Vector3s>> &contact_map)
      : Base(ndx, nu, 3), nk_(nu / 3), mass_(mass), gravity_(gravity),
        contact_map_(contact_map) {
    if (contact_map.size() != nk_) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("Contact ids and nk should be the same: now "
                      "({} and {}).",
                      contact_map.size(), nk_));
    }
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

  std::vector<std::pair<bool, Vector3s>> contact_map_;

protected:
  const int nk_;
  const double mass_;
  const Vector3s gravity_;
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
