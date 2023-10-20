#pragma once

#include "proxddp/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>

namespace proxddp {

template <typename Scalar> struct CenterOfMassTranslationDataTpl;

template <typename _Scalar>
struct CenterOfMassTranslationResidualTpl : UnaryFunctionTpl<_Scalar>,
                                            frame_api {
public:
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  PROXDDP_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = CenterOfMassTranslationDataTpl<Scalar>;

  shared_ptr<Model> pin_model_;

  CenterOfMassTranslationResidualTpl(const int ndx, const int nu,
                                     const shared_ptr<Model> &model,
                                     const Vector3s &frame_trans)
      : Base(ndx, nu, 3), pin_model_(model), p_ref_(frame_trans) {}

  const Vector3s &getReference() const { return p_ref_; }
  void setReference(const Eigen::Ref<const Vector3s> &p_new) { p_ref_ = p_new; }

  void evaluate(const ConstVectorRef &x, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }

protected:
  Vector3s p_ref_;
};

template <typename Scalar>
struct CenterOfMassTranslationDataTpl : FunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = FunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;

  CenterOfMassTranslationDataTpl(
      const CenterOfMassTranslationResidualTpl<Scalar> *model);
};

} // namespace proxddp

#include "proxddp/modelling/multibody/center-of-mass-translation.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./center-of-mass-translation.txx"
#endif
