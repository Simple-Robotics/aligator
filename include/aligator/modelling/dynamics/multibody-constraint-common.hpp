/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/core/common-model-abstract.hpp"

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/constrained-dynamics.hpp>
#include <pinocchio/algorithm/constrained-dynamics-derivatives.hpp>

#include <optional>

namespace aligator {
namespace dynamics {
template <typename _Scalar> struct MultibodyConstraintCommonDataTpl;
template <typename _Scalar> class MultibodyConstraintCommonBuilderTpl;

template <typename _Scalar>
struct MultibodyConstraintCommonTpl : CommonModelTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_COMMON_MODEL_TYPEDEFS(Scalar, MultibodyConstraintCommonDataTpl,
                                 MultibodyConstraintCommonBuilderTpl);

  using PinocchioModel = pinocchio::ModelTpl<Scalar>;
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    const int nq = pin_model_.nq;
    const int nv = pin_model_.nv;
    if (run_aba_) {
      d.tau_.noalias() = actuation_matrix_ * u;
      d.qdd_.noalias() = pinocchio::constraintDynamics(
          pin_model_, d.pin_data_, x.head(nq), x.segment(nq, nv), d.tau_,
          constraint_models_, d.constraint_datas_, d.prox_settings_);
    }
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    if (run_aba_) {
      pinocchio::computeConstraintDynamicsDerivatives(
          pin_model_, d.pin_data_, constraint_models_, d.constraint_datas_,
          d.prox_settings_);
      d.qdd_dtau_.noalias() = d.pin_data_.ddq_dtau * actuation_matrix_;
    }
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       BaseData &) const override {}

  std::shared_ptr<BaseData> createData() const override {
    return std::make_shared<Data>(this);
  }

  PinocchioModel pin_model_;
  MatrixXs actuation_matrix_;
  bool run_aba_;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;
};

template <typename _Scalar>
class MultibodyConstraintCommonBuilderTpl
    : public CommonModelBuilderTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_COMMON_BUILDER_TYPEDEFS(Scalar, MultibodyConstraintCommonTpl);

  using PinocchioModel = pinocchio::ModelTpl<Scalar>;
  using RigidConstraintModelVector = PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(
      pinocchio::RigidConstraintModel);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;

  /// @return Reference to this to allow method-chaining
  Self &withPinocchioModel(PinocchioModel model) {
    pin_model_ = std::move(model);
    return *this;
  }

  /// @return Reference to this to allow method-chaining
  Self &withActuationMatrix(const MatrixXs &actuation_matrix) {
    actuation_matrix_ = actuation_matrix;
    return *this;
  }

  /// @return Reference to this to allow method-chaining
  Self &withRunAba(bool run_aba) {
    run_aba_ = run_aba;
    return *this;
  }

  /// @return Reference to this to allow method-chaining
  Self &withConstraintModels(RigidConstraintModelVector constraint_models) {
    constraint_models_ = std::move(constraint_models);
    return *this;
  }

  Self &withProxSettings(ProxSettings prox_settings) {
    prox_settings_ = std::move(prox_settings);
    return *this;
  }

  std::shared_ptr<BaseModel> build() const override {
    auto model = std::make_shared<Model>();
    const int nv = pin_model_->nv;
    if (!pin_model_) {
      ALIGATOR_RUNTIME_ERROR("No pinocchio::Model provided");
    }
    if (actuation_matrix_.cols() == 0 && actuation_matrix_.rows() == 0) {
      model->actuation_matrix_.setIdentity(nv, nv);
    } else {
      if (actuation_matrix_.rows() != nv || actuation_matrix_.cols() != nv) {
        ALIGATOR_RUNTIME_ERROR(fmt::format(
            "Wrong actuation_matrix size: "
            "({}, {}) provided but should be ({}, {})",
            actuation_matrix_.rows(), actuation_matrix_.cols(), nv, nv));
      }
      model->actuation_matrix_ = actuation_matrix_;
    }
    model->pin_model_ = *pin_model_;
    model->run_aba_ = run_aba_;
    model->constraint_models_ = constraint_models_;
    model->prox_settings_ = prox_settings_;
    return model;
  }

private:
  std::optional<PinocchioModel> pin_model_;
  MatrixXs actuation_matrix_;
  bool run_aba_ = false;
  RigidConstraintModelVector constraint_models_;
  ProxSettings prox_settings_;
};

template <typename _Scalar>
struct MultibodyConstraintCommonDataTpl : CommonModelDataTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_COMMON_DATA_TYPEDEFS(Scalar, MultibodyConstraintCommonTpl);

  using PinocchioData = pinocchio::DataTpl<Scalar>;
  using RigidConstraintDataVector =
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(pinocchio::RigidConstraintData);
  using ProxSettings = pinocchio::ProximalSettingsTpl<Scalar>;

  MultibodyConstraintCommonDataTpl(const Model *model)
      : nv(model->pin_model_.nv), tau_(model->pin_model_.nv),
        qdd_(model->pin_model_.nv),
        qdd_dtau_(model->pin_model_.nv, model->pin_model_.nv),
        actuation_matrix_(model->actuation_matrix_),
        pin_data_(model->pin_model_), prox_settings_(model->prox_settings_) {
    pinocchio::initConstraintDynamics(model->pin_model_, pin_data_,
                                      model->constraint_models_);
    for (const auto &cm : model->constraint_models_) {
      constraint_datas_.push_back(
          pinocchio::RigidConstraintDataTpl<Scalar, 0>(cm));
    }
  }

  // TODO remove if we share Model
  Eigen::DenseIndex nv;
  VectorXs tau_;
  VectorXs qdd_;
  MatrixXs qdd_dtau_;
  // TODO remove if we share Model
  MatrixXs actuation_matrix_;
  PinocchioData pin_data_;
  RigidConstraintDataVector constraint_datas_;
  ProxSettings prox_settings_;
};

} // namespace dynamics
} // namespace aligator

// #include "aligator/modelling/dynamics/multibody-common.hxx"

// TODO template instantiation
