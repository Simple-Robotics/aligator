#pragma once

#include "./fwd.hpp"
#include "aligator/core/function-abstract.hpp"
#include "aligator/modelling/contact-map.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

namespace aligator {

template <typename Scalar> struct CentroidalMomentumDerivativeDataTpl;

/**
 * @brief This residual returns the difference between angular momentum
 * from centroidal model and resulting momentum from multibody model.
 */

template <typename _Scalar>
struct CentroidalMomentumDerivativeResidualTpl : StageFunctionTpl<_Scalar>,
                                                 frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = CentroidalMomentumDerivativeDataTpl<Scalar>;
  using ContactMap = ContactMapTpl<Scalar>;

  Model pin_model_;
  double mass_;
  Vector3s gravity_;
  ContactMap contact_map_;

  CentroidalMomentumDerivativeResidualTpl(const Model &model,
                                          const Vector3s &gravity,
                                          const ContactMap &contact_map)
      : Base(6 + 2 * model.nv, (int)contact_map.getSize() * 3 + model.nv, 6),
        pin_model_(model), gravity_(gravity), contact_map_(contact_map) {
    mass_ = pinocchio::computeTotalMass(model);
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
struct CentroidalMomentumDerivativeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Force = pinocchio::ForceTpl<Scalar>;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  /// Pinocchio data object.
  PinData pin_data_;
  Matrix3s Jtemp_;
  Force hdot_;
  Matrix6Xs dh_dq_;
  Matrix6Xs dhdot_dq_;
  Matrix6Xs dhdot_dv_;
  Matrix6Xs dhdot_da_;

  CentroidalMomentumDerivativeDataTpl(
      const CentroidalMomentumDerivativeResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/multibody/centroidal-momentum-derivative.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./centroidal-momentum-derivative.txx"
#endif
