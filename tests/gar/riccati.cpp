/// @copyright Copyright (C) 2023-2024 LAAS-CNRS, 2023-2025 INRIA
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(4, 0, ",", "\n", "[", "]")

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_get_random_seed.hpp>

// used for timings, numbers are merely indicative, this *not* a benchmark
#include <chrono>

#include "aligator/gar/utils.hpp"
#include "aligator/gar/proximal-riccati.hpp"
#include "aligator/gar/dense-riccati.hpp"

#include "test_util.hpp"
#include <aligator/fmt-eigen.hpp>

using namespace aligator::gar;
static std::pmr::monotonic_buffer_resource mbr{10 * 1024 * 1048};
static aligator::polymorphic_allocator alloc{&mbr};

static_assert(
    std::uses_allocator_v<problem_t, aligator::polymorphic_allocator>);

TEST_CASE("riccati_short_horz_pb", "[gar]") {
  // dual regularization parameters
  const double mueq = 1e-14;

  uint nx = 2, nu = 2;
  VectorXs x0 = VectorXs::Ones(nx);
  VectorXs x1 = -VectorXs::Ones(nx);
  const auto init_knot = [&](uint nc) -> knot_t {
    knot_t knot(nx, nu, nc, alloc);
    knot.A << 0.1, 0., -0.1, 0.01;
    knot.B.setRandom();
    knot.f.setRandom();
    knot.Q.setIdentity();
    knot.Q *= 0.01;
    knot.R.setIdentity();
    knot.R *= 0.1;
    return knot;
  };
  auto base_knot = init_knot(0u);
  auto knot1 = base_knot;
  knot1.Q.setIdentity();
  knot1.q = -x1;

  uint horz = GENERATE(4, 8, 16);
  problem_t::KnotVector knots(horz + 1, base_knot);
  knots[4] = init_knot(nu);
  knots[4].D.setIdentity();
  knots[4].d.setConstant(0.1);
  knots[horz] = std::move(knot1);
  problem_t prob(std::move(knots), nx);
  prob.g0 = -x0;
  prob.G0.setIdentity();
  ProximalRiccatiSolver solver{prob};
  fmt::print("Horizon: {:d}\n", prob.horizon());

  auto bwbeg = std::chrono::system_clock::now();
  CHECK(solver.backward(mueq));
  auto bwend = std::chrono::system_clock::now();
  auto t_bwd =
      std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
  fmt::print("Elapsed time (bwd): {:d}\n", t_bwd.count());

  auto [xs, us, vs, lbdas] = lqrInitializeSolution(prob);
  REQUIRE(xs.size() == size_t(prob.horizon()) + 1);
  REQUIRE(vs.size() == size_t(prob.horizon()) + 1);
  REQUIRE(lbdas.size() == size_t(prob.horizon()) + 1);

  auto fwbeg = std::chrono::system_clock::now();
  CHECK(solver.forward(xs, us, vs, lbdas));
  auto fwend = std::chrono::system_clock::now();
  auto t_fwd =
      std::chrono::duration_cast<std::chrono::microseconds>(fwend - fwbeg);
  fmt::print("Elapsed time (fwd): {:d}\n", t_fwd.count());

  // check error
  KktError err = computeKktError(prob, xs, us, vs, lbdas);
  fmt::print("{}\n", err);

  REQUIRE(err.max <= 1e-9);
}

TEST_CASE("riccati_one_knot_prob", "[gar]") {
  std::mt19937 rng{Catch::getSeed()};
  uint nx = 2;
  uint nu = 2;
  Eigen::VectorXd x0;
  x0.setZero(nx);
  auto problem = generateLqProblem(rng, x0, 0, nx, nu, 0, 0, true, alloc);
  ProximalRiccatiSolver solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  REQUIRE(xs.size() == 1);
  REQUIRE(us.size() == 0);
  REQUIRE(lbdas.size() == 1);
  const double mueq = 1e-13;
  solver.backward(mueq);
  solver.forward(xs, us, vs, lbdas);

  KktError err = computeKktError(problem, xs, us, vs, lbdas);
  REQUIRE(err.max <= 1e-10);
}

TEST_CASE("riccati_random_large_problem", "[gar]") {
  std::mt19937 rng{Catch::getSeed()};
  uint nx = 36;
  uint nu = 12;
  VectorXs x0;
  x0.setZero(nx);
  uint horz = GENERATE(20, 100);
  const auto problem =
      generateLqProblem(rng, x0, horz, nx, nu, 0, 0, true, alloc);
  const double mueq = 1e-14;

  SECTION("prox riccati") {
    ProximalRiccatiSolver solver{problem};
    auto bwbeg = std::chrono::system_clock::now();
    solver.backward(mueq);
    auto bwend = std::chrono::system_clock::now();
    auto t_bwd =
        std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
    fmt::print("Elapsed time (bwd): {:d}\n", t_bwd.count());

    auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
    auto fwbeg = std::chrono::system_clock::now();
    solver.forward(xs, us, vs, lbdas);
    auto fwend = std::chrono::system_clock::now();
    auto t_fwd =
        std::chrono::duration_cast<std::chrono::microseconds>(fwend - fwbeg);
    fmt::print("Elapsed time (fwd): {:d}\n", t_fwd.count());

    KktError err = computeKktError(problem, xs, us, vs, lbdas);
    fmt::print("{}\n", err);
    /// TODO: tighten this tolerance
    REQUIRE(err.max <= 1e-6);
  }

  SECTION("test dense solver") {
    RiccatiSolverDense denseSolver(problem);
    auto bwbeg = std::chrono::system_clock::now();
    denseSolver.backward(mueq);
    auto bwend = std::chrono::system_clock::now();
    auto t_bwd =
        std::chrono::duration_cast<std::chrono::microseconds>(bwend - bwbeg);
    fmt::print("Elapsed time (bwd, dense): {:d}\n", t_bwd.count());
    auto [xsd, usd, vsd, lbdasd] = lqrInitializeSolution(problem);
    denseSolver.forward(xsd, usd, vsd, lbdasd);
    KktError errd = computeKktError(problem, xsd, usd, vsd, lbdasd);
    fmt::print("{}\n", errd);
    REQUIRE(errd.max <= 1e-8);
  }
}

TEST_CASE("riccati_parametric", "[gar]") {
  std::mt19937 rng{Catch::getSeed()};
  uint nx = 10;
  VectorXs x0 = VectorXs::NullaryExpr(nx, normal_unary_op(rng));
  uint nu = 4;
  uint horz = 100;
  uint nth = 1;
  const auto problem =
      generateLqProblem(rng, x0, horz, nx, nu, nth, 0, true, alloc);
  const double mueq = 1e-12;

  ProximalRiccatiSolver solver(problem);
  auto [xs, us, vs, lbdas] = lqrInitializeSolution(problem);
  solver.backward(mueq);

  VectorXs theta(nth);
  theta.setRandom();
  solver.forward(xs, us, vs, lbdas, theta);

  KktError err = computeKktError(problem, xs, us, vs, lbdas, theta);
  fmt::print("{}\n", err);
  CHECK(err.max <= 1e-9);

  using aligator::math::check_value;
  REQUIRE_FALSE(check_value(solver.kkt0.chol.matrixLDLT()));
  REQUIRE_FALSE(check_value(solver.kkt0.ff.matrix()));
  REQUIRE_FALSE(check_value(solver.kkt0.fth.matrix()));
  REQUIRE_FALSE(check_value(solver.thGrad));
  REQUIRE_FALSE(check_value(solver.thHess));
  REQUIRE_FALSE(check_value(solver.datas[0].vm.vt));
  REQUIRE_FALSE(check_value(solver.datas[0].vm.Vxt));
  REQUIRE_FALSE(check_value(solver.datas[0].vm.Vtt));
  REQUIRE_FALSE(check_value(solver.datas[horz].vm.vt));
  REQUIRE_FALSE(check_value(solver.datas[horz].vm.Vxt));
  REQUIRE_FALSE(check_value(solver.datas[horz].vm.Vtt));
}
