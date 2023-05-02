#pragma once

#include "proxddp/core/function-abstract.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

namespace proxddp {

template <typename Scalar> struct FramePlacementDataTpl;

template <typename _Scalar>
struct FramePlacementResidualTpl : StageFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FramePlacementDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;
  pinocchio::FrameIndex pin_frame_id_;

  FramePlacementResidualTpl(const int ndx, const int nu,
                            const shared_ptr<Model> &model, const SE3 &frame,
                            const pinocchio::FrameIndex id)
      : Base(ndx, nu, 6), pin_model_(model), pin_frame_id_(id), p_ref_(frame),
        p_ref_inverse_(frame.inverse()) {}

  pinocchio::FrameIndex getFrameId() const { return pin_frame_id_; }
  void setFrameId(const std::size_t id) { pin_frame_id_ = id; }

  const SE3 &getReference() const { return p_ref_; }
  void setReference(const SE3 &p_new) {
    p_ref_ = p_new;
    p_ref_inverse_ = p_new.inverse();
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

protected:
  SE3 p_ref_;
  SE3 p_ref_inverse_;
};

template <typename Scalar>
struct FramePlacementDataTpl : FunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = FunctionDataTpl<Scalar>;
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
