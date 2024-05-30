#pragma once

#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/dynamics/integrator-euler.hpp"

using T = double;
using SE2 = proxsuite::nlp::SETpl<2, T>;

using namespace aligator;
using StateError = StateErrorResidualTpl<T>;
using QuadStateCost = QuadraticStateCostTpl<T>;
using QuadControlCost = QuadraticControlCostTpl<T>;
using context::StageModel;
using context::TrajOptProblem;
ALIGATOR_DYNAMIC_TYPEDEFS(T);

/// @details The dynamics of the car are given by
/// \f[
///   \dot x = f(x,u) = \begin{bmatrix} v\cos\theta \\ v\sin\theta \\ \omega
///   \end{bmatrix}
/// \f]
/// with state \f$x = (x,y,\cos\theta,\sin\theta) \in \mathrm{SE}(2)\f$, control
/// \f$u=(v,\omega)\f$.
///
struct CarDynamics : dynamics::ODEAbstractTpl<T> {
  using Base = dynamics::ODEAbstractTpl<T>;
  using ODEData = dynamics::ODEDataTpl<T>;
  CarDynamics() : Base(SE2(), 2) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ODEData &data) const override {
    // actuation matrix
    T s, c;
    s = std::sin(x[2]);
    c = std::cos(x[2]);
    data.Ju_.col(0) << c, s, 0.;
    data.Ju_(2, 1) = 1.;

    data.xdot_.noalias() = data.Ju_ * u;
  }
  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ODEData &data) const override {
    // Ju_ already computed
    data.Jx_.setZero();
    T v = u[0];
    T s, c;
    s = std::sin(x[2]);
    c = std::cos(x[2]);

    data.Jx_.col(2) << -s * v, c * v, 0.;
  }
};

inline auto create_se2_problem(std::size_t nsteps) {
  auto space = SE2();
  const int nu = 2;
  const int ndx = space.ndx();

  VectorXs x0(space.nx());
  {
    double theta = 0.15355;
    pinocchio::SINCOS(theta, &x0[2], &x0[3]);
    x0[0] = 0.7;
    x0[1] = -0.1;
  }
  const VectorXs x_target = space.neutral();

  MatrixXs w_x = MatrixXs::Zero(ndx, ndx);
  w_x.diagonal().array() = 0.01;
  MatrixXs w_term = w_x * 10;
  MatrixXs w_u = MatrixXs::Identity(nu, nu);
  w_u = w_u.transpose() * w_u;

  const T timestep = 0.05;

  auto rcost = CostStackTpl<T>(space, nu);
  auto rc1 = QuadStateCost(space, nu, x_target, w_x * timestep);
  auto rc2 = QuadControlCost(space, nu, w_u * timestep);
  rcost.addCost(rc1);
  rcost.addCost(rc2);

  auto ode = CarDynamics(); // xyz::polymorphic<CarDynamics>();
  auto discrete_dyn = dynamics::IntegratorEulerTpl<T>(ode, timestep);

  auto stage = StageModel(rcost, discrete_dyn);

  std::vector<xyz::polymorphic<StageModel>> stages(nsteps, stage);

  auto term_cost = QuadStateCost(space, nu, x_target, w_term);
  return TrajOptProblem(x0, stages, term_cost);
}
