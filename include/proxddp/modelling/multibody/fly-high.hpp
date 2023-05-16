#pragma once

#include "proxddp/core/unary-function.hpp"
#include <proxnlp/modelling/spaces/multibody.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/frames-derivatives.hpp>

namespace proxddp {

namespace {
namespace pin = pinocchio;
}

/// A port of sobec's ResidualModelFlyHighTpl.
template <typename _Scalar>
struct FlyHighResidualTpl : UnaryFunctionTpl<_Scalar> {
  using Scalar = _Scalar;
  PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
  static constexpr int NR = 2;

  using Base = UnaryFunctionTpl<Scalar>;
  using BaseData = FunctionDataTpl<Scalar>;
  using PhaseSpace = proxnlp::MultibodyPhaseSpace<Scalar>;

  struct Data;

  FlyHighResidualTpl(shared_ptr<PhaseSpace> space,
                     const pin::FrameIndex frame_id, Scalar slope,
                     std::size_t nu);

  void evaluate(const ConstVectorRef &x, BaseData &data) const;
  void computeJacobians(const ConstVectorRef &x, BaseData &data) const;

  shared_ptr<BaseData> createData() const {
    return std::make_shared<Data>(*this);
  }

  pinocchio::FrameIndex getFrameId() const { return frame_id_; }
  void setFrameId(const std::size_t id) { frame_id_ = id; }

  const auto &getModel() const { return pmodel_; }

  pin::FrameIndex frame_id_;
  Scalar slope_;

private:
  pin::ModelTpl<Scalar> pmodel_;
};

template <typename Scalar>
using FlyHighResidualDataTpl = typename FlyHighResidualTpl<Scalar>::Data;

template <typename Scalar>
struct FlyHighResidualTpl<Scalar>::Data : FunctionDataTpl<Scalar> {
  using BaseData::ndx1;
  using BaseData::nr;
  using BaseData::nu;

  Data(FlyHighResidualTpl const &model)
      : BaseData(model.ndx1, model.nu, model.ndx2, model.nr),
        pdata_(model.pmodel_), d_dq(6, model.pmodel_.nv),
        d_dv(6, model.pmodel_.nv), l_dnu_dq(6, model.pmodel_.nv),
        l_dnu_dv(6, model.pmodel_.nv), o_dv_dq(3, model.pmodel_.nv),
        o_dv_dv(3, model.pmodel_.nv), vxJ(3, model.pmodel_.nv) {
    d_dq.setZero();
    d_dv.setZero();
    l_dnu_dq.setZero();
    l_dnu_dv.setZero();
    o_dv_dq.setZero();
    o_dv_dv.setZero();
    vxJ.setZero();
  }

  pin::DataTpl<Scalar> pdata_;
  Matrix6Xs d_dq, d_dv;
  Matrix6Xs l_dnu_dq, l_dnu_dv;
  Matrix3Xs o_dv_dq, o_dv_dv, vxJ;
  Scalar ez;
};

} // namespace proxddp

#include "proxddp/modelling/multibody/fly-high.hxx"

#ifdef PROXDDP_ENABLE_TEMPLATE_INSTANTIATION
#include "./fly-high.txx"
#endif
