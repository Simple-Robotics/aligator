#include "aligator/gar/lqr-problem.hpp"
#include <proxsuite-nlp/fmt-eigen.hpp>

#include <boost/test/unit_test.hpp>

#include "test_util.hpp"

using namespace aligator::gar;
using Eigen::MatrixXd;
using Eigen::VectorXd;
char MEMORY_BUFFER[1'000'000];
static std::pmr::monotonic_buffer_resource rs{MEMORY_BUFFER,
                                              sizeof(MEMORY_BUFFER)};
static aligator::polymorphic_allocator alloc{&rs};

struct knot_fixture {
  uint nx = 2;
  uint nu = 2;
  knot_t knot;
  knot_fixture() : knot(nx, nu, 0, alloc) {
    knot.Q.setRandom();
    knot.R.setRandom();
    knot.q.setRandom();
    knot.r.setRandom();

    knot.A.setZero();
    knot.B.setZero();
    knot.E.setZero();
    knot.f.setZero();
  }
};

BOOST_FIXTURE_TEST_CASE(move, knot_fixture) {

  MatrixXd Q = knot.Q;
  MatrixXd R = knot.R;

  knot_t knot_moved{std::move(knot)};
  BOOST_CHECK_EQUAL(knot_moved.nx, nx);
  BOOST_CHECK_EQUAL(knot_moved.nu, nu);
  BOOST_CHECK(Q == knot_moved.Q);
  BOOST_CHECK(R == knot_moved.R);

  // copy ctor
  knot_t knot_move2 = std::move(knot_moved);
}

BOOST_FIXTURE_TEST_CASE(copy, knot_fixture) {

  knot_t knot2{knot};
  BOOST_CHECK_EQUAL(knot, knot2);
}

BOOST_FIXTURE_TEST_CASE(swap, knot_fixture) {
  using std::swap;
  MatrixXs Q0 = knot.Q;
  knot_t knot2 = knot;
  knot2.Q.setIdentity();

  fmt::println("knot.Q:\n{}", knot.Q);
  fmt::println("knot2.Q:\n{}", knot2.Q);

  swap(knot, knot2);
  BOOST_CHECK(Q0 == knot2.Q);

  fmt::println("knot2.Q:\n{}", knot2.Q);
}

BOOST_FIXTURE_TEST_CASE(gen_knot, knot_fixture) {
  knot_t knot2 = generate_knot(nx, nu, 0);
  this->knot = std::move(knot2);
}

BOOST_AUTO_TEST_CASE(knot_vec) {
  uint nx = 4;
  uint nu = 2;
  std::vector<knot_t> v;
  v.reserve(10);
  for (int i = 0; i < 10; i++) {
    v.push_back(generate_knot(nx, nu, 0));
  }

  for (size_t i = 0; i < 10; i++) {
    fmt::println("v [{:d}].q = {}", i, v[i].q.transpose());
  }

  std::vector<knot_t> v2 = std::move(v);
  for (size_t i = 0; i < 10; i++) {
    fmt::println("v2[{:d}].q = {}", i, v2[i].q.transpose());
  }

  std::vector<knot_t> vc{v2};
  for (size_t i = 0; i < 10; i++) {
    fmt::println("vc[{:d}].q = {}", i, vc[i].q.transpose());
    BOOST_CHECK_EQUAL(v2[i], vc[i]);
  }
}

BOOST_AUTO_TEST_CASE(problem) {
  BOOST_TEST_MESSAGE("problem");
  uint nx = 4;
  uint nu = 2;
  std::pmr::vector<knot_t> v{alloc};
  v.reserve(10);
  for (int i = 0; i < 10; i++) {
    v.push_back(generate_knot(nx, nu, 0));
  }
  problem_t prob{v, nx};
  BOOST_CHECK(prob.get_allocator() == alloc);
  BOOST_CHECK(prob.get_allocator().resource() == alloc.resource());
  BOOST_CHECK(prob.G0.cols() == prob.stages[0].nx);
  fmt::print("Q[0] = \n{}\n", prob.stages[0].Q);

  problem_t prob_move{std::move(prob)};

  prob_move.addParameterization(1);
  for (size_t i = 0; i < 10; i++) {
    BOOST_CHECK_EQUAL(prob_move.stages[i].nth, 1);
  }
}
