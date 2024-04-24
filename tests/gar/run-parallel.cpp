#define TRACY_SAMPLING_HZ 100000
#include "test_util.hpp"
#include "aligator/gar/parallel-solver.hpp"
#include "aligator/gar/utils.hpp"

using namespace aligator::gar;

const double mu = 1e-14;

int main(int argc, char *argv[]) {
  ZoneScoped;

  uint num_threads;
  uint horz;
  if (argc >= 2) {
    try {
      num_threads = (uint)std::stoi(argv[1], nullptr);
      horz = (uint)std::stoi(argv[2], nullptr);
    } catch (const std::invalid_argument &ex) {
      fmt::print(ex.what());
      std::terminate();
    }
  } else {
    std::terminate();
  }
  aligator::omp::set_default_options(num_threads);

  uint nx = 36;
  uint nu = 12;
  Eigen::VectorXd x0(nx);
  x0.setRandom();
  auto problem = generate_problem(x0, horz, nx, nu);

  size_t nreps = 20;
  {
    ZoneScopedN("parallel_runs");
    auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
    ParallelRiccatiSolver<double> solver(problem, num_threads);

    for (size_t i = 0; i < nreps; i++) {
      ZoneScopedNC("parallel", 0x9500ff);
      solver.backward(mu, mu);
      solver.forward(xs, us, vs, lbdas);
    }
  }

#if 0
  {
    ZoneScopedN("serial_runs");
    auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
    ProximalRiccatiSolver<double> solver(problem);

    for (size_t i = 0; i < nreps; i++) {
      ZoneScopedN("serial");
      solver.backward(mu, mu);
      solver.forward(xs, us, vs, lbdas);
    }
  }
#endif
}
