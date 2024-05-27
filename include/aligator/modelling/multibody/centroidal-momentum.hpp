#pragma once

#include "./fwd.hpp"
#include "aligator/core/unary-function.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct CentroidalMomentumDataTpl;

/**
 * @brief This residual returns the derivative of centroidal momentum
 * for a kinodynamics model.
 */

template <typename _Scalar>
struct CentroidalMomentumResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = UnaryFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = CentroidalMomentumDataTpl<Scalar>;

  Model pin_model_;
  Vector6s h_ref_;

  CentroidalMomentumResidualTpl(const int ndx, const int nu, const Model &model,
                                const Vector6s &h_ref)
      : Base(ndx, nu, 6), pin_model_(model), h_ref_(h_ref) {}

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  const Vector6s &getReference() const { return h_ref_; }
  void setReference(const Eigen::Ref<const Vector6s> &h_new) { h_ref_ = h_new; }

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }
};

template <typename Scalar>
struct CentroidalMomentumDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using Matrix6Xs = typename math_types<Scalar>::Matrix6Xs;

  /// Pinocchio data object.
  PinData pin_data_;
  Matrix6Xs dh_dq_;
  Matrix6Xs dhdot_dq_;
  Matrix6Xs dhdot_dv_;
  Matrix6Xs dhdot_da_;

  CentroidalMomentumDataTpl(const CentroidalMomentumResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/multibody/centroidal-momentum.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./centroidal-momentum.txx"
#endif
