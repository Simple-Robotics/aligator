#include "talos-walk-utils.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/core/enums.hpp"
#include <fmt/ostream.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using aligator::SolverProxDDPTpl;

constexpr double TOL = 1e-4;
constexpr std::size_t max_iters = 100;
constexpr double mu_init = 1e-8;

int main(int, char **) {

  std::size_t T_ss = 80;
  std::size_t T_ds = 20;

  std::size_t nsteps = T_ss * 2 + T_ds * 3;

  auto problem = defineLocomotionProblem(T_ss, T_ds);

  SolverProxDDPTpl<double> solver(TOL, mu_init, max_iters, aligator::VERBOSE);
  std::vector<VectorXd> xs_i, us_i;
  Eigen::VectorXd u0 = Eigen::VectorXd::Zero(22);
  xs_i.assign(nsteps + 1, problem.getInitState());
  us_i.assign(nsteps, u0);

  solver.rollout_type_ = aligator::RolloutType::LINEAR;
  solver.sa_strategy = aligator::StepAcceptanceStrategy::FILTER;
  solver.filter_.beta_ = 1e-5;
  solver.force_initial_condition_ = true;
  solver.reg_min = 1e-6;
  solver.linear_solver_choice = aligator::LQSolverChoice::PARALLEL;
  solver.setNumThreads(4); // call before setup()
  solver.setup(problem);
  solver.run(problem, xs_i, us_i);

  auto &res = solver.results_;
  fmt::print("Results: {}\n", res);
}
