#pragma once

#include "aligator/core/unary-function.hpp"
#include "./fwd.hpp"

#include <pinocchio/multibody/model.hpp>

namespace aligator {

template <typename Scalar> struct CenterOfMassTranslationDataTpl;

/**
 * @brief This residual returns the Center of Mass translation for a centroidal
 * model with state \f$x = (c, h, L) \f$.
 *
 * @details The residual returns the first three components of the state:
 * \f$r(x) = c - c_r \f$ with \f$ c \f$ center of mass and \f$ c_r \f$ desired
 * reference for center of mass.
 */

template <typename _Scalar>
struct CenterOfMassTranslationResidualTpl : UnaryFunctionTpl<_Scalar>,
                                            frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  ALIGATOR_UNARY_FUNCTION_INTERFACE(Scalar);
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = CenterOfMassTranslationDataTpl<Scalar>;

  Model pin_model_;

  CenterOfMassTranslationResidualTpl(const int ndx, const int nu,
                                     const Model &model,
                                     const Vector3s &frame_trans)
      : Base(ndx, nu, 3)
      , pin_model_(model)
      , p_ref_(frame_trans) {}

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
struct CenterOfMassTranslationDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;

  /// Pinocchio data object.
  PinData pin_data_;

  CenterOfMassTranslationDataTpl(
      const CenterOfMassTranslationResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./center-of-mass-translation.txx"
#endif
