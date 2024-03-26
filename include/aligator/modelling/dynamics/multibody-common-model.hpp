/// @copyright Copyright (C) 2024 LAAS-CNRS, INRIA
#pragma once

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include "aligator/core/common-model-abstract.hpp"

namespace aligator {
namespace dynamics {
template <typename _Scalar> struct MultibodyCommonModelDataTpl;

template <typename _Scalar>
struct MultibodyCommonModelTpl : CommonModelTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CommonModelTpl<Scalar>;
  using BaseData = typename CommonModelTpl<Scalar>::Data;
  using Data = MultibodyCommonModelDataTpl<Scalar>;
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
struct MultibodyCommonModelDataTpl : CommonModelDataTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using Scalar = _Scalar;
  ALIGATOR_DYNAMIC_TYPEDEFS(Scalar);
  using Base = CommonModelDataTpl<Scalar>;
  using Model = MultibodyCommonModelTpl<Scalar>;
  using PinocchioData = pinocchio::DataTpl<Scalar>;

  MultibodyCommonModelDataTpl(const Model *model)
      : tau_(model->pin_model_.nv), qdd_(model->pin_model_.nv),
        qdd_dq_(model->pin_model_.nv, model->pin_model_.nv),
        qdd_dv_(model->pin_model_.nv, model->pin_model_.nv),
        qdd_dtau_(model->pin_model_.nv, model->pin_model_.nv),
        pin_data_(model->pin_model_) {}

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
