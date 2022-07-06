#pragma once

#include "proxddp/modelling/dynamics/integrator-semi-euler.hpp"

namespace proxddp {
namespace dynamics {

template <typename Scalar>
IntegratorSemiImplEulerTpl<Scalar>::IntegratorSemiImplEulerTpl(
    const shared_ptr<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics), timestep_(timestep) {}

template <typename Scalar>
void IntegratorSemiImplEulerTpl<Scalar>::forward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEDataTpl<Scalar> &cdata =
      static_cast<ODEDataTpl<Scalar> &>(*d.continuous_data);
  this->ode_->forward(x, u, cdata);
  int ndx = this->ndx1;
  const int ndx_2 = ndx / 2;
  d.dx_.bottomRows(ndx_2) = cdata.xdot_.bottomRows(ndx_2) * timestep_;
  this->out_space().integrate(x, d.dx_, d.xnext_);
  d.dx_.topRows(ndx_2) = d.xnext_.bottomRows(ndx_2) * timestep_;
  this->out_space().integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorSemiImplEulerTpl<Scalar>::dForward(
    const ConstVectorRef &x, const ConstVectorRef &u,
    ExplicitDynamicsDataTpl<Scalar> &data) const {
  Data &d = static_cast<Data &>(data);
  ODEDataTpl<Scalar> &cdata =
      static_cast<ODEDataTpl<Scalar> &>(*d.continuous_data);
  int ndx = this->ndx1;
  int nu = this->nu;
  const int ndx_2 = ndx / 2;
  const auto &space = this->out_space();
  MatrixXs Jx_tmp(ndx, ndx),
      Ju_tmp(ndx, nu); // TODO: remove dynamical allocation and use data (solve
                       // segfault)

  this->ode_->dForward(x, u, cdata);
  // dv_dx and dv_du are same as euler explicit
  d.Jx_ = timestep_ * cdata.Jx_; // dddx_dx
  d.Ju_ = timestep_ * cdata.Ju_; // ddx_du
  space.JintegrateTransport(x, d.dx_, d.Jx_, 1);
  space.JintegrateTransport(x, d.dx_, d.Ju_, 1);
  space.Jintegrate(x, d.dx_, d.Jtmp_xnext, 0);
  d.Jx_ += d.Jtmp_xnext;

  // dq_dx and dq_du needs to be modified
  Jx_tmp.topRows(ndx_2) = timestep_ * d.Jx_.bottomRows(ndx_2);
  Jx_tmp.bottomRows(ndx_2) = timestep_ * cdata.Jx_.bottomRows(ndx_2);
  Ju_tmp.topRows(ndx_2) = timestep_ * d.Ju_.bottomRows(ndx_2);
  Ju_tmp.bottomRows(ndx_2) = timestep_ * cdata.Ju_.bottomRows(ndx_2);

  space.JintegrateTransport(x, d.dx_, Jx_tmp, 1);
  space.JintegrateTransport(x, d.dx_, Ju_tmp, 1);
  Jx_tmp += d.Jtmp_xnext;
  d.Jx_.topRows(ndx_2) = Jx_tmp.topRows(ndx_2);
  d.Ju_.topRows(ndx_2) = Ju_tmp.topRows(ndx_2);
}

template <typename Scalar>
IntegratorSemiImplDataTpl<Scalar>::IntegratorSemiImplDataTpl(
    const IntegratorSemiImplEulerTpl<Scalar> *integrator)
    : Base(integrator), Jtmp_xnext2(integrator->ndx1, integrator->ndx1),
      Jtmp_u(integrator->ndx1, integrator->nu) {
  Jtmp_xnext2.setZero();
  Jtmp_u.setZero();
}

} // namespace dynamics

} // namespace proxddp
