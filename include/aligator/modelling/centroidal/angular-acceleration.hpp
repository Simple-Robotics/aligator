#pragma once

#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

namespace aligator {

template <typename Scalar> struct AngularAccelerationDataTpl;

/**
 * @brief This residual returns the angular acceleration of a centroidal model
 * with state \f$x = (c, h, L) \f$, computed from the external forces and
 * contact poses.
 *
 * @details The cost is described by \f$r(x,u) = \sum_{k \in K} (p_k - c) \times
 * u_k\f$ with \f$K\f$ set of active contacts, \f$p_k\f$ contact pose k, \f$c\f$
 * Center of Mass, and \f$u_k\f$ 3D unilateral force for contact k. All contacts
 * are considered as unilateral.
 */

template <typename _Scalar>
struct AngularAccelerationResidualTpl : StageFunctionTpl<_Scalar> {

public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Data = AngularAccelerationDataTpl<Scalar>;
  using ContactMap = ContactMapTpl<Scalar>;

  AngularAccelerationResidualTpl(const int ndx, const int nu, const double mass,
                                 const Vector3s &gravity,
                                 const ContactMap &contact_map,
                                 const int force_size)
      : Base(ndx, nu, 3)
      , contact_map_(contact_map)
      , nk_(size_t(nu) / size_t(force_size))
      , mass_(mass)
      , gravity_(gravity)
      , force_size_(force_size) {
    if (contact_map.size_ != nk_) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("Contact ids and nk should be the same: now "
                      "({} and {}).",
                      contact_map.size_, nk_));
    }
  }

  void evaluate(const ConstVectorRef &, const ConstVectorRef &u,
                BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

  ContactMap contact_map_;

protected:
  size_t nk_;
  double mass_;
  Vector3s gravity_;
  int force_size_;
};

template <typename Scalar>
struct AngularAccelerationDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  Matrix3s Jtemp_;

  AngularAccelerationDataTpl(
      const AngularAccelerationResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/centroidal/angular-acceleration.txx"
#endif
