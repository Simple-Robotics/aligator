#pragma once

#include "./fwd.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/centroidal/angular-acceleration.hpp"
#include "aligator/modelling/contact-map.hpp"
#include <pinocchio/algorithm/center-of-mass.hpp>

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct AngularMomentumConstraintDataTpl;

/**
 * @brief This residual returns the difference between angular momentum
 * from centroidal model and resulting momentum from multibody model.
 */

template <typename _Scalar>
struct AngularMomentumConstraintResidualTpl : StageFunctionTpl<_Scalar>,
                                              frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using Centroidal = AngularAccelerationResidualTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = AngularMomentumConstraintDataTpl<Scalar>;
  using ContactMap = ContactMapTpl<Scalar>;

  shared_ptr<Model> pin_model_;
  shared_ptr<Centroidal> centroidal_model_;

  AngularMomentumConstraintResidualTpl(const shared_ptr<Model> &model,
                                       const Vector3s &gravity,
                                       const ContactMap &contact_map)
      : Base(9 + 2 * model->nv, (int)contact_map.getSize() * 3 + model->nv, 3),
        pin_model_(model) {
    double mass = pinocchio::computeTotalMass(*model);
    centroidal_model_ = std::make_shared<Centroidal>(Centroidal(
        9, (int)contact_map.getSize() * 3, mass, gravity, contact_map));
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &c, const ConstVectorRef &u,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }
};

template <typename Scalar>
struct AngularMomentumConstraintDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;

  shared_ptr<Base> centroidal_data_;
  /// Pinocchio data object.
  PinData pin_data_;
  Matrix6Xs dh_dq_;
  Matrix6Xs dhdot_dq_;
  Matrix6Xs dhdot_dv_;
  Matrix6Xs dhdot_da_;

  AngularMomentumConstraintDataTpl(
      const AngularMomentumConstraintResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/multibody/angular-momentum-constraint.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./angular-momentum-constraint.txx"
#endif
