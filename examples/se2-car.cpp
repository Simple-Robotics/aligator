/// A car in SE2

#include "se2-car.hpp"

#include "aligator/solvers/proxddp/solver-proxddp.hpp"

int main() {
  TrajOptProblem problem = create_se2_problem(40);
  const T mu_init = 1e-2;
  SolverProxDDPTpl<T> solver(1e-4, mu_init);
  solver.verbose_ = VERBOSE;
  solver.sa_strategy_ = StepAcceptanceStrategy::FILTER;
  solver.linear_solver_choice = LQSolverChoice::PARALLEL;
  solver.setNumThreads(4);
  solver.rollout_type_ = RolloutType::LINEAR;
  solver.setup(problem);
  solver.run(problem);

  fmt::print("{}\n", fmt::streamed(solver.results_));

  return 0;
}
