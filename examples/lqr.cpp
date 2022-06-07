/// @file
/// @brief Linear-quadratic regulator

#include "proxddp/core/explicit-dynamics.hpp"
#include "proxddp/core/shooting-problem.hpp"
#include "proxddp/utils.hpp"
#include "proxddp/modelling/costs.hpp"
#include "proxddp/solver-proxddp.hpp"

#include <proxnlp/modelling/spaces/vector-space.hpp>
#include <proxnlp/modelling/constraints/negative-orthant.hpp>

#include "proxddp/modelling/box-constraints.hpp"

#include "boost/optional.hpp"


namespace proxddp
{

  template<typename _Scalar>
  struct LinearDiscreteDynamics : ExplicitDynamicsModelTpl<_Scalar>
  {
    using Scalar = _Scalar;
    PROXNLP_DYNAMIC_TYPEDEFS(Scalar);
    const MatrixXs A_;
    const MatrixXs B_;
    VectorXs c_;

    using Base = ExplicitDynamicsModelTpl<double>;
    LinearDiscreteDynamics(const MatrixXs& A,
                          const MatrixXs& B,
                          const boost::optional<VectorXs>& c = boost::none)
      : Base(std::make_shared<proxnlp::VectorSpaceTpl<double>>((int)A.cols()), (int)B.cols())
      , A_(A), B_(B)
      {
        if (boost::optional<VectorXs> value = c)
        {
          c_ = *value;
        } else {
          c_ = VectorXs::Zero(A.cols());
        }
      }

    void forward(const ConstVectorRef& x,
                const ConstVectorRef& u,
                VectorRef out) const override
    {
      out = A_ * x + B_ * u + c_;
    }
    
    void dForward(const ConstVectorRef&,
                  const ConstVectorRef&,
                  MatrixRef Jx, MatrixRef Ju) const override
    {
      Jx = A_;
      Ju = B_;
    }

  };

} // namespace proxddp

using namespace proxddp;


int main()
{

  const int dim = 2;
  const int nu = 1;
  Eigen::MatrixXd A(dim, dim);
  Eigen::MatrixXd B(dim, nu);
  Eigen::VectorXd c_(dim);
  A.setIdentity();
  B << -0.6, 0.3;
  c_ << 0.1, 0.;

  Eigen::MatrixXd w_x(dim, dim), w_u(nu, nu);
  w_x.setIdentity();
  w_u.setIdentity();
  w_x(0, 0) = 2.1;
  w_u(0, 0) = 0.06;

  auto dynptr = std::make_shared<LinearDiscreteDynamics<double>>(A, B, c_);
  auto& dynamics = *dynptr;
  fmt::print("matrix A:\n{}\n", dynamics.A_);
  fmt::print("matrix B:\n{}\n", dynamics.B_);
  fmt::print("drift  c:\n{}\n", dynamics.c_);
  const auto& space = dynamics.out_space();

  QuadraticCost<double> rcost(w_x, w_u);
  ControlBoxFunction<double> ctrl_bounds_fun(dim, nu, -0.1, 0.1);
  auto ctrl_bounds_cstr = std::make_shared<StageConstraintTpl<double>>(
    ctrl_bounds_fun,
    std::make_shared<proxnlp::NegativeOrthant<double>>());

  // Define stage

  StageModelTpl<double> stage(space, nu, rcost, dynamics);
  // stage.addConstraint(ctrl_bounds_cstr);

  auto x0 = space.rand();
  x0 << 1., -0.1;
  ShootingProblemTpl<double> problem(x0);

  std::size_t nsteps = 8;

  std::vector<Eigen::VectorXd> us;
  for (std::size_t i = 0; i < nsteps; i++)
  {
    us.push_back(Eigen::VectorXd::Random(nu));
    problem.addStage(*stage.clone());
  }

  problem.term_cost_ = std::make_shared<decltype(rcost)>(rcost);
  auto xs = rollout(dynamics, x0, us);
  fmt::print("Initial traj.:\n");
  for(std::size_t i = 0; i <= nsteps; i++)
  {
    fmt::print("x[{:d}] = {}\n", i, xs[i].transpose());
  }

  double TOL = 1e-6;
  double mu_init = 1e-4;
  double rho_init = 0.;

  SolverProxDDP<double> solver(TOL, mu_init, rho_init);

  WorkspaceTpl<double> workspace(problem);
  ResultsTpl<double> results(problem);
  assert(results.xs_.size() == nsteps + 1);
  assert(results.us_.size() == nsteps);

  solver.run(problem, workspace, results, xs, us);

  std::string line_ = "";
  for (std::size_t i = 0; i < 20; i++)
  { line_.append("="); }
  line_.append("\n");
  for (std::size_t i = 0; i < nsteps + 1; i++)
  {
    // fmt::print("x[{:d}] = {}\n", i, results.xs_[i].transpose());
    fmt::print("x[{:d}] = {}\n", i, workspace.trial_xs_[i].transpose());
  }
  for (std::size_t i = 0; i < nsteps; i++)
  {
    // fmt::print("u[{:d}] = {}\n", i, results.us_[i].transpose());
    fmt::print("u[{:d}] = {}\n", i, workspace.trial_us_[i].transpose());
  }

  return 0;
}

