#include "croc-talos-arm.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"
#include <fmt/ostream.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using aligator::SolverProxDDPTpl;

int main(int, char **) {
  constexpr double TOL = 1e-4;
  const std::size_t nsteps = 30;

  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto problem = aligator::compat::croc::convertCrocoddylProblem(croc_problem);

  double mu_init = 0.001;
  SolverProxDDPTpl<double> solver(TOL, mu_init, 30ul, aligator::VERBOSE);

  std::vector<VectorXd> xs_i, us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  solver.setup(problem);
  solver.run(problem, xs_i, us_i);

  auto &res = solver.results_;
  fmt::print("Results: {}\n", res);
}
