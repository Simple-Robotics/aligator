/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include <optional>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include "aligator/core/common-model-abstract.hpp"

namespace aligator {
namespace dynamics {
template <typename _Scalar> struct MultibodyCommonDataTpl;
template <typename _Scalar> class MultibodyCommonBuilderTpl;

template <typename _Scalar>
struct MultibodyCommonTpl : CommonModelTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_COMMON_MODEL_TYPEDEFS(Scalar, MultibodyCommonDataTpl,
                                 MultibodyCommonBuilderTpl);

  using PinocchioModel = pinocchio::ModelTpl<Scalar>;

  void evaluate(const ConstVectorRef &x, const ConstVectorRef &u,
                BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    const int nq = pin_model_.nq;
    const int nv = pin_model_.nv;
    if (run_aba_) {
      d.tau_.noalias() = actuation_matrix_ * u;
      d.qdd_.noalias() = pinocchio::aba(pin_model_, d.pin_data_, x.head(nq),
                                        x.segment(nq, nv), d.tau_);
    }
  }

  void computeGradients(const ConstVectorRef &x, const ConstVectorRef &,
                        BaseData &data) const override {
    Data &d = static_cast<Data &>(data);
    const int nq = pin_model_.nq;
    const int nv = pin_model_.nv;
    if (run_aba_) {
      pinocchio::computeABADerivatives(pin_model_, d.pin_data_, x.head(nq),
                                       x.tail(nv), d.tau_, d.qdd_dq_, d.qdd_dv_,
                                       d.pin_data_.Minv);
      d.qdd_dtau_.noalias() = d.pin_data_.Minv * actuation_matrix_;
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
  /// TODO allow to configure how to retrieve q, v and tau
};

template <typename _Scalar>
class MultibodyCommonBuilderTpl : public CommonModelBuilderTpl<_Scalar> {
public:
  using Scalar = _Scalar;
  ALIGATOR_COMMON_BUILDER_TYPEDEFS(Scalar, MultibodyCommonTpl);

  using PinocchioModel = pinocchio::ModelTpl<Scalar>;

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
    return model;
  }

private:
  std::optional<PinocchioModel> pin_model_;
  MatrixXs actuation_matrix_;
  bool run_aba_ = false;
};

template <typename _Scalar>
struct MultibodyCommonDataTpl : CommonModelDataTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_COMMON_DATA_TYPEDEFS(Scalar, MultibodyCommonTpl);

  using PinocchioData = pinocchio::DataTpl<Scalar>;

  MultibodyCommonDataTpl(const Model *model)
      : nv(model->pin_model_.nv), tau_(model->pin_model_.nv),
        qdd_(model->pin_model_.nv),
        qdd_dq_(model->pin_model_.nv, model->pin_model_.nv),
        qdd_dv_(model->pin_model_.nv, model->pin_model_.nv),
        qdd_dtau_(model->pin_model_.nv, model->pin_model_.nv),
        pin_data_(model->pin_model_) {}

  // TODO remove if we share Model
  Eigen::DenseIndex nv;
  VectorXs tau_;
  VectorXs qdd_;
  MatrixXs qdd_dq_;
  MatrixXs qdd_dv_;
  MatrixXs qdd_dtau_;
  PinocchioData pin_data_;
};

} // namespace dynamics
} // namespace aligator

// #include "aligator/modelling/dynamics/multibody-common.hxx"

// TODO template instantiation
