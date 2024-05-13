#pragma once

#include "./fwd.hpp"
#include "aligator/core/function-abstract.hpp"

#include <pinocchio/multibody/model.hpp>
#include <proxsuite-nlp/modelling/spaces/multibody.hpp>
#include <pinocchio/multibody/data.hpp>

#include <pinocchio/algorithm/proximal.hpp>

namespace aligator {

template <typename Scalar> struct ContactForceDataTpl;

/**
 * @brief This residual returns the derivative of centroidal momentum
 * for a kinodynamics model.
 */

template <typename _Scalar>
struct ContactForceResidualTpl : StageFunctionTpl<_Scalar>, frame_api {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = ContactForceDataTpl<Scalar>;
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;

  Model pin_model_;
  MatrixXs actuation_matrix_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;
  Vector6s fref_;
  int contact_id_;

  ContactForceResidualTpl(const int ndx, const Model &model,
                          const MatrixXs &actuation,
                          const RigidConstraintModelVector &constraint_models,
                          const ProxSettings &prox_settings,
                          const Vector6s &fref, const int contact_id)
      : Base(ndx, (int)actuation.cols(), 6), pin_model_(model),
        actuation_matrix_(actuation), constraint_models_(constraint_models),
        prox_settings_(prox_settings), fref_(fref), contact_id_(contact_id) {
    if (model.nv != actuation.rows()) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("actuation matrix should have number of rows = pinocchio "
                      "model nv ({} and {}).",
                      actuation.rows(), model.nv));
    }
  }

  const Vector6s &getReference() const { return fref_; }
  void setReference(const Eigen::Ref<const Vector6s> &fnew) { fref_ = fnew; }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                const ConstVectorRef &, BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        const ConstVectorRef &, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return allocate_shared_eigen_aligned<Data>(this);
  }
};

template <typename Scalar>
struct ContactForceDataTpl : StageFunctionDataTpl<Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Base = StageFunctionDataTpl<Scalar>;
  using PinData = pinocchio::DataTpl<Scalar>;
  using VectorXs = typename math_types<Scalar>::VectorXs;
  using MatrixXs = typename math_types<Scalar>::MatrixXs;
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);

  /// Pinocchio data object.
  PinData pin_data_;
  VectorXs tau_;

  RigidConstraintDataVector constraint_datas_;
  pinocchio::ProximalSettingsTpl<Scalar> settings;

  ContactForceDataTpl(const ContactForceResidualTpl<Scalar> *model);
};

} // namespace aligator

#include "aligator/modelling/multibody/contact-force.hxx"

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "./contact-force.txx"
#endif
