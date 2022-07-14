#pragma once

#include "proxddp/core/function.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/frame.hpp>

namespace proxddp {

template <typename Scalar> struct FrameTranslationDataTpl;

template <typename _Scalar>
struct FrameTranslationResidualTpl : StageFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using ManifoldPtr = shared_ptr<ManifoldAbstractTpl<Scalar>>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = FrameTranslationDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;
  pinocchio::FrameIndex pin_frame_id_;

  FrameTranslationResidualTpl(const int ndx, const int nu,
                            const shared_ptr<Model> &model, const VectorXs &frame_trans,
                            const pinocchio::FrameIndex id)
      : Base(ndx, nu, 3), pin_model_(model), pin_frame_id_(id), p_ref_(frame_trans){}

  pinocchio::FrameIndex getFrameId() const { return pin_frame_id_; }
  void setFrameId(const std::size_t id) { pin_frame_id_ = id; }

  const VectorXs &getReference() const { return p_ref_; }
  void setReference(const VectorXs &p_new) {
    p_ref_ = p_new;
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &y, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, const ConstVectorRef &u,
                        const ConstVectorRef &y, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  VectorXs p_ref_;
};

template <typename Scalar>
struct FrameTranslationDataTpl : FunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = FunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;

  /// Jacobian of the error, local frame
  typename math_types<Scalar>::Matrix6Xs fJf_;

  FrameTranslationDataTpl(const FrameTranslationResidualTpl<Scalar> *model);
};

} // namespace proxddp

#include "proxddp/modelling/multibody/frame-translation.hxx"
