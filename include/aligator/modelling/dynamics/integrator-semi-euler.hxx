/// @copyright Copyright (C) 2022-2024 LAAS-CNRS, INRIA
#pragma once

#include "aligator/modelling/dynamics/integrator-semi-euler.hpp"

namespace aligator {
namespace dynamics {

template <typename Scalar>
IntegratorSemiImplEulerTpl<Scalar>::IntegratorSemiImplEulerTpl(
    const xyz::polymorphic<ODEType> &cont_dynamics, const Scalar timestep)
    : Base(cont_dynamics)
    , timestep_(timestep) {
  assert(((this->ndx1()) % 2 == 0) &&
         "IntegratorSemiImplEuler must be used with even ndx.");
}

template <typename Scalar>
void IntegratorSemiImplEulerTpl<Scalar>::forward(const ConstVectorRef &x,
                                                 const ConstVectorRef &u,
                                                 BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  ODEData &cdata = *d.continuous_data;
  this->ode_->forward(x, u, cdata);
  int ndx = this->ndx1();
  const int ndx_2 = ndx / 2;
  d.dx_.bottomRows(ndx_2) = cdata.xdot_.bottomRows(ndx_2) * timestep_;
  this->space_next().integrate(x, d.dx_, d.xnext_);
  d.dx_.topRows(ndx_2) = d.xnext_.bottomRows(ndx_2) * timestep_;
  this->space_next().integrate(x, d.dx_, d.xnext_);
}

template <typename Scalar>
void IntegratorSemiImplEulerTpl<Scalar>::dForward(const ConstVectorRef &x,
                                                  const ConstVectorRef &u,
                                                  BaseData &data) const {
  Data &d = static_cast<Data &>(data);
  ODEData &cdata = *d.continuous_data;
  int ndx = this->ndx1();
  const int ndx_2 = ndx / 2;
  const auto &space = this->space_next();

  this->ode_->dForward(x, u, cdata);
  // dv_dx and dv_du are same as euler explicit
  d.Jx() = timestep_ * cdata.Jx(); // dddx_dx
  d.Ju() = timestep_ * cdata.Ju(); // ddx_du
  space.JintegrateTransport(x, d.dx_, d.Jx(), 1);
  space.JintegrateTransport(x, d.dx_, d.Ju(), 1);
  space.Jintegrate(x, d.dx_, d.Jtmp_xnext, 0);
  d.Jx() += d.Jtmp_xnext;

  // dq_dx and dq_du needs to be modified
  d.Jtmp_xnext2.topRows(ndx_2) = timestep_ * d.Jx().bottomRows(ndx_2);
  d.Jtmp_xnext2.bottomRows(ndx_2) = timestep_ * cdata.Jx().bottomRows(ndx_2);
  d.Jtmp_u.topRows(ndx_2) = timestep_ * d.Ju().bottomRows(ndx_2);
  d.Jtmp_u.bottomRows(ndx_2) = timestep_ * cdata.Ju().bottomRows(ndx_2);

  space.JintegrateTransport(x, d.dx_, d.Jtmp_xnext2, 1);
  space.JintegrateTransport(x, d.dx_, d.Jtmp_u, 1);
  d.Jtmp_xnext2 += d.Jtmp_xnext;
  d.Jx().topRows(ndx_2) = d.Jtmp_xnext2.topRows(ndx_2);
  d.Ju().topRows(ndx_2) = d.Jtmp_u.topRows(ndx_2);
}

} // namespace dynamics
} // namespace aligator
