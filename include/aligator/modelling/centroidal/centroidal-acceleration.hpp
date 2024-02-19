#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator {

template <typename Scalar> struct CentroidalAccelerationDataTpl;

/**
 * @brief This residual returns the linear acceleration of a centroidal model
 * with state \f$x = (c, h, L) \f$, computed from the external forces and
 * contact poses.
 *
 * @details The cost is described by \f$r(x,u) = g + \frac{1}{m} * \sum_{k \in
 * K} u_k\f$ with \f$K\f$ set of active contacts, \f$g\f$ gravity, \f$m\f$ model
 * mass, and \f$u_k\f$ 3D unilateral force for contact k. All contacts are
 * considered as unilateral.
 */

template <typename _Scalar>
struct CentroidalAccelerationResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = CentroidalAccelerationDataTpl<Scalar>;
  using ContactMap = ContactMapTpl<Scalar>;

  CentroidalAccelerationResidualTpl(const int ndx, const int nu,
                                    const double mass, const Vector3s &gravity,
                                    const ContactMap &contact_map,
                                    const int force_size)
      : Base(ndx, nu, 3), contact_map_(contact_map),
        nk_(size_t(nu) / size_t(force_size)), mass_(mass), gravity_(gravity),
        force_size_(force_size) {
    if (contact_map.getSize() != nk_) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("Contact ids and nk should be the same: now "
                      "({} and {}).",
                      contact_map.getSize(), nk_));
    }
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }

  ContactMap contact_map_;

protected:
  size_t nk_;
  double mass_;
  Vector3s gravity_;
  int force_size_;
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
