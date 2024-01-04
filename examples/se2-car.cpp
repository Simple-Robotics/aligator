/// A car in SE2

#include <proxsuite-nlp/modelling/spaces/pinocchio-groups.hpp>
#include <pinocchio/multibody/liegroup/special-euclidean.hpp>

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/modelling/state-error.hpp"
#include "aligator/modelling/costs/composite-costs.hpp"
#include "aligator/modelling/costs/sum-of-costs.hpp"
#include "aligator/modelling/dynamics/ode-abstract.hpp"
#include "aligator/modelling/dynamics/integrator-euler.hpp"

using T = double;
using SE2 = proxsuite::nlp::SETpl<2, T>;

using namespace aligator;
using StateError = StateErrorResidualTpl<T>;
using QuadResidualCost = QuadraticResidualCostTpl<T>;
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
  CarDynamics() : Base(std::make_shared<SE2>(), 2) {}

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

int main() {
  auto space = std::make_shared<SE2>();
  const int nu = 2;
  const int ndx = space->ndx();

  const VectorXs x0 = space->rand();
  VectorXs utest = VectorXs::Random(nu);
  const VectorXs x_target = space->neutral();

  auto state_err = std::make_shared<StateError>(space, nu, x_target);

  MatrixXs w_x = MatrixXs::Zero(ndx, ndx);
  w_x.diagonal().array() = 0.01;
  MatrixXs w_term = w_x * 10;
  MatrixXs w_u = MatrixXs::Random(nu, nu);
  w_u = w_u.transpose() * w_u;

  const T timestep = 0.05;

  auto rcost = std::make_shared<CostStackTpl<T>>(space, nu);
  auto rc1 =
      std::make_shared<QuadResidualCost>(space, state_err, w_x * timestep);
  auto control_err =
      std::make_shared<ControlErrorResidualTpl<T>>(space->ndx(), nu);
  auto rc2 =
      std::make_shared<QuadResidualCost>(space, control_err, w_u * timestep);
  rcost->addCost(rc1);
  rcost->addCost(rc2);

  auto term_cost = std::make_shared<QuadResidualCost>(space, state_err, w_term);
  auto ode = std::make_shared<CarDynamics>();
  auto discrete_dyn =
      std::make_shared<dynamics::IntegratorEulerTpl<T>>(ode, timestep);

  auto stage = std::make_shared<StageModel>(rcost, discrete_dyn);

  const size_t nsteps = 40;
  std::vector<decltype(stage)> stages(nsteps);
  std::fill_n(stages.begin(), nsteps, stage);

  TrajOptProblem problem(x0, stages, term_cost);
  const T mu_init = 1e-2;
  SolverProxDDP<T> solver(1e-4, mu_init);
  solver.verbose_ = VERBOSE;
  solver.setup(problem);
  solver.run(problem);

  fmt::print("{}\n", fmt::streamed(solver.results_));

  return 0;
}
