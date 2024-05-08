#include "aligator/gar/cholmod-solver.hpp"
#include "aligator/gar/utils.hpp"

#include <boost/test/unit_test.hpp>
#include "test_util.hpp"

using namespace aligator;

BOOST_AUTO_TEST_CASE(cholmod_short_horz) {
  const double mu = 1e-14;
  uint nx = 2, nu = 2;
  VectorXs x0;
  x0.setRandom(nx);
  problem_t problem = generate_problem(x0, nx, nu, 0);
  gar::CholmodLqSolver<double> solver{problem};

  auto [xs, us, vs, lbdas] = gar::lqrInitializeSolution(problem);

  solver.backward(mu, mu);
  solver.forward(xs, us, vs, lbdas);

  auto [dynErr, cstErr, dualErr] = gar::lqrComputeKktError(
      problem, xs, us, vs, lbdas, mu, mu, std::nullopt, true);
  fmt::print("KKT errors: d = {:.4e} / dual = {:.4e}\n", dynErr, dualErr);
}
