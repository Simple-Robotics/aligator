#pragma once

#include "proxddp/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

namespace proxddp {

template <typename Scalar> struct FrameTranslationDataTpl;

template <typename _Scalar>
struct FrameTranslationResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  PROXDDP_DYNAMIC_TYPEDEFS(Scalar);
  PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FrameTranslationDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;

  FrameTranslationResidualTpl(const int ndx, const int nu,
                              const shared_ptr<Model> &model,
                              const Vector3s &frame_trans,
                              const pinocchio::FrameIndex frame_id)
      : Base(ndx, nu, 3), pin_model_(model), p_ref_(frame_trans) {
    pin_frame_id_ = frame_id;
  }

  const Vector3s &getReference() const { return p_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &p_new) { p_ref_ = p_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(*this);
  }

protected:
  Vector3s p_ref_;
};

template <typename Scalar>
struct FrameTranslationDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;

  /// Jacobian of the error, local frame
  typename math_types<Scalar>::Matrix6Xs fJf_;

  FrameTranslationDataTpl(const FrameTranslationResidualTpl<Scalar> &model);
};

} // namespace proxddp

#include "proxddp/modelling/multibody/frame-translation.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./frame-translation.txx"
#endif
