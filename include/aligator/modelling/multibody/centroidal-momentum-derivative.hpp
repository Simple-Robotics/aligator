#pragma once

#include "./fwd.hpp"
#include "aligator/core/function-abstract.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct CentroidalMomentumDerivativeDataTpl;

/**
 * @brief This residual returns the derivative of centroidal momentum
 * for a kinodynamics model.
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

  Model pin_model_;
  double mass_;
  Vector3s gravity_;
  std::vector<bool> contact_states_;
  std::vector<pinocchio::FrameIndex> contact_ids_;
  int force_size_;

  CentroidalMomentumDerivativeResidualTpl(
      const int ndx, const Model &model, const Vector3s &gravity,
      const std::vector<bool> &contact_states,
      const std::vector<pinocchio::FrameIndex> &contact_ids,
      const int force_size);

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }
};

template <typename Scalar>
struct CentroidalMomentumDerivativeDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Matrix3Xs = typename math_types<Scalar>::Matrix3Xs;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;
  using Matrix3s = Eigen::Matrix<Scalar, 3, 3>;

  /// Pinocchio data object.
  PinData pin_data_;
  Matrix3s Jtemp_;
  Matrix3Xs temp_;
  Matrix6Xs fJf_;

  CentroidalMomentumDerivativeDataTpl(
      const CentroidalMomentumDerivativeResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/multibody/centroidal-momentum-derivative.txx"
#endif
