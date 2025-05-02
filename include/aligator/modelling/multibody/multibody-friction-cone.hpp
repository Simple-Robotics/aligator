#pragma once

#include "fwd.hpp"
#include "aligator/core/function-abstract.hpp"

#include <proxsuite-nlp/modelling/spaces/multibody.hpp>

#include <pinocchio/algorithm/proximal.hpp>

namespace aligator {

template <typename Scalar> struct MultibodyFrictionConeDataTpl;

/**
 * @brief This residual returns the derivative of centroidal momentum
 * for a kinodynamics model.
 */

template <typename _Scalar>
struct MultibodyFrictionConeResidualTpl : StageFunctionTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = StageFunctionTpl<Scalar>;
  using BaseData = typename Base::Data;
  using Model = pinocchio::ModelTpl<Scalar>;
  using SE3 = pinocchio::SE3Tpl<Scalar>;
  using Data = MultibodyFrictionConeDataTpl<Scalar>;
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;

  Model pin_model_;
  MatrixXs actuation_matrix_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;
  double mu_;
  int contact_id_;

  MultibodyFrictionConeResidualTpl(
      const int ndx, const Model &model, const MatrixXs &actuation,
      const RigidConstraintModelVector &constraint_models,
      const ProxSettings &prox_settings, const std::string &contact_name,
      const double mu)
      : Base(ndx, (int)actuation.cols(), 2), pin_model_(model),
        actuation_matrix_(actuation), constraint_models_(constraint_models),
        prox_settings_(prox_settings), mu_(mu) {
    if (model.nv != actuation.rows()) {
      ALIGATOR_DOMAIN_ERROR(
          fmt::format("actuation matrix should have number of rows = pinocchio "
                      "model nv ({} and {}).",
                      actuation.rows(), model.nv));
    }
    contact_id_ = -1;
    for (std::size_t i = 0; i < constraint_models.size(); i++) {
      if (constraint_models[i].name == contact_name) {
        contact_id_ = (int)i;
      }
    }
    if (contact_id_ == -1) {
      ALIGATOR_RUNTIME_ERROR(
          "Contact name is not included in constraint models");
    }
  }

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const;

  void computeJacobians(const ConstVectorRef &, const ConstVectorRef &,
                        BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(this);
  }
};

template <typename Scalar>
struct MultibodyFrictionConeDataTpl : StageFunctionDataTpl<Scalar> {
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
  MatrixXs temp_;
  Eigen::Matrix<Scalar, 1, 3> dcone_df_;

  RigidConstraintDataVector constraint_datas_;
  pinocchio::ProximalSettingsTpl<Scalar> settings;

  MultibodyFrictionConeDataTpl(
      const MultibodyFrictionConeResidualTpl<Scalar> *model);
};

} // namespace aligator

#ifdef ALIGATOR_ENABLE_TEMPLATE_INSTANTIATION
#include "aligator/modelling/multibody/multibody-friction-cone.txx"
#endif
