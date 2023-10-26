#pragma once

#include "proxddp/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

namespace proxddp {

template <typename Scalar> struct FramePlacementDataTpl;

template <typename _Scalar>
struct FramePlacementResidualTpl : UnaryFunctionTpl<_Scalar>, frame_api {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = StageFunctionDataTpl<Scalar>;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FramePlacementDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;

  FramePlacementResidualTpl(const int ndx, const int nu,
                            const shared_ptr<Model> &model, const SE3 &frame,
                            const pinocchio::FrameIndex frame_id)
      : Base(ndx, nu, 6), pin_model_(model), p_ref_(frame),
        p_ref_inverse_(frame.inverse()) {
    pin_frame_id_ = frame_id;
  }

  const SE3 &getReference() const { return p_ref_; }
  void setReference(const SE3 &p_new) {
    p_ref_ = p_new;
    p_ref_inverse_ = p_new.inverse();
  }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

protected:
  SE3 p_ref_;
  SE3 p_ref_inverse_;
};

template <typename Scalar>
struct FramePlacementDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;
  /// Placement error of the frame.
  SE3 rMf_;
  /// Jacobian of the error
  typename math_types<Scalar>::Matrix6s rJf_;
  /// Jacobian of the error, local frame
  typename math_types<Scalar>::Matrix6Xs fJf_;

  FramePlacementDataTpl(const FramePlacementResidualTpl<Scalar> &model);
};

} // namespace proxddp

#include "proxddp/modelling/multibody/frame-placement.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "proxddp/modelling/multibody/frame-placement.txx"
#endif
