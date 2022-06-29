#pragma once

#include "proxddp/modelling/dynamics/multibody-free-fwd.hpp"

#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>

#include <stdexcept>


namespace proxddp
{
namespace dynamics
{
  template<typename Scalar>
  MultibodyFreeFwdDynamicsTpl<Scalar>::
  MultibodyFreeFwdDynamicsTpl(
    const ManifoldPtr& state,
    const MatrixXs& actuation)
    : Base(state, actuation.cols())
    , space_(state)
    , actuation_matrix_(actuation)
  {
    const int nv = state->getModel().nv;
    if (nv != actuation.rows())
    {
      throw std::domain_error(fmt::format("actuation matrix should have number of rows = pinocchio model nv ({} and {}).", actuation.rows(), nv));
    }
  }

  template<typename Scalar>
  void MultibodyFreeFwdDynamicsTpl<Scalar>::
  forward(const ConstVectorRef& x, const ConstVectorRef& u, ODEData& data) const
  {
    Data& d = static_cast<Data&>(data);
    d.tau_ = actuation_matrix_ * u;
    const pinocchio::ModelTpl<Scalar>& model = space().getModel();
    const int nq = model.nq;
    const int nv = model.nv;
    d.xdot_.head(nv) = x.tail(nv);
    d.xdot_.tail(nv) = pinocchio::aba(model, *d.pin_data_, x.head(nq), x.tail(nv), d.tau_);
  }

  template<typename Scalar>
  void MultibodyFreeFwdDynamicsTpl<Scalar>::
  dForward(const ConstVectorRef& x, const ConstVectorRef& u, ODEData& data) const
  {
    Data& d = static_cast<Data&>(data);
    // do not change d.dtau_du_ which is already set in createData()
    const auto& model = space().getModel();
    const int nq = model.nq;
    const int nv = model.nv;
    const ConstVectorRef q = x.head(nq);
    const ConstVectorRef v = x.head(nv);
    pinocchio::computeABADerivatives(model, *d.pin_data_, q, v, d.tau_, d.Jx_.leftCols(nv), d.Jx_.rightCols(nv),
                                     d.pin_data_->Minv);
    d.Ju_ = d.pin_data_->Minv * d.dtau_du_;
  }

  template<typename Scalar>
  shared_ptr<ContinuousDynamicsDataTpl<Scalar>>
  MultibodyFreeFwdDynamicsTpl<Scalar>::createData() const
  {
    auto data = std::make_shared<MultibodyFreeFwdDataTpl<Scalar>>(this->ndx(), this->nu());
    data->tau_ = VectorXs::Zero(space().getModel().nv);
    data->pin_data_ = std::make_shared<pinocchio::DataTpl<Scalar>>(space().getModel());
    data->dtau_du_ = this->actuation_matrix_;
    return data;
  }
} // namespace dynamics
} // namespace proxddp

