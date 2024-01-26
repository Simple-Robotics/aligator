#include "croc-talos-arm.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/compat/crocoddyl/problem-wrap.hpp"
#include <fmt/ostream.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using aligator::SolverProxDDPTpl;

constexpr double TOL = 1e-4;
constexpr std::size_t max_iters = 10;

const std::size_t nsteps = 30;

int main(int, char **) {

  auto croc_problem = defineCrocoddylProblem(nsteps);
  auto problem = aligator::compat::croc::convertCrocoddylProblem(croc_problem);

  double mu_init = 0.01;
  SolverProxDDPTpl<double> solver(TOL, mu_init, 0., max_iters,
                                  aligator::VERBOSE);

  std::vector<VectorXd> xs_i, us_i;
  getInitialGuesses(croc_problem, xs_i, us_i);

  solver.setup(problem);
  solver.run(problem, xs_i, us_i);

  auto &res = solver.results_;
  fmt::print("Results: {}\n", res);
}
