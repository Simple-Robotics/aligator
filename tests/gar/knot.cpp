#include "aligator/gar/lqr-problem.hpp"
#include "aligator/fmt-eigen.hpp"

#include <catch2/catch_test_macros.hpp>

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
  knot_fixture()
      : knot(nx, nu, 0, alloc) {
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

TEST_CASE_METHOD(knot_fixture, "move", "[knot]") {

  MatrixXd Q = knot.Q;
  MatrixXd R = knot.R;

  knot_t knot_moved{std::move(knot)};
  REQUIRE(knot_moved.nx == nx);
  REQUIRE(knot_moved.nu == nu);
  REQUIRE(knot_moved.Q.isApprox(Q));
  REQUIRE(knot_moved.R.isApprox(R));

  knot_t knot_moved2{std::move(knot_moved), alloc};
  REQUIRE(knot_moved2.get_allocator() == alloc);
  REQUIRE(knot_moved2.Q.isApprox(Q));
  REQUIRE(knot_moved2.R.isApprox(R));
}

TEST_CASE_METHOD(knot_fixture, "copy", "[knot]") {

  // use default allocator
  knot_t knot2{knot};
  REQUIRE(knot2.get_allocator() == aligator::polymorphic_allocator{});
  REQUIRE(knot2 == knot);

  knot_t knot3{knot, alloc};
  REQUIRE(knot3.get_allocator() == alloc);
  REQUIRE(knot3 == knot);
}

TEST_CASE_METHOD(knot_fixture, "swap", "[knot]") {
  using std::swap;
  MatrixXs Q0 = knot.Q;
  knot_t knot2 = knot;
  knot2.Q.setIdentity();

  fmt::println("knot.Q:\n{}", knot.Q);
  fmt::println("knot2.Q:\n{}", knot2.Q);

  swap(knot, knot2);
  REQUIRE(knot2.Q.isApprox(Q0));

  fmt::println("knot2.Q:\n{}", knot2.Q);
}

TEST_CASE_METHOD(knot_fixture, "gen_knot", "[knot]") {
  knot_t knot2 = generate_knot(nx, nu, 0);
  this->knot = std::move(knot2);
}

TEST_CASE("copy_assignment_diff_allocator", "[knot]") {
  // test copying data from a different allocator
  knot_t knot{2, 2, 0, alloc};
  REQUIRE(knot.get_allocator() == alloc);

  aligator::polymorphic_allocator default_alloc{};
  knot_t knot2{4, 2, 1, default_alloc};
  knot = knot2;
  REQUIRE(knot.get_allocator() == alloc);
  REQUIRE(knot.nx == 4);
  REQUIRE(knot.nu == 2);
  REQUIRE(knot.nc == 1);

  REQUIRE(knot.q.rows() == 4);
  REQUIRE(knot.r.rows() == 2);
  REQUIRE(knot.d.rows() == 1);
}

TEST_CASE("move_assignment_diff_allocator", "[knot]") {
  // test copying data from a different allocator
  knot_t knot{2, 2, 0, alloc};
  REQUIRE(knot.get_allocator() == alloc);

  aligator::polymorphic_allocator default_alloc{};
  knot = generate_knot(2, 2, 0, default_alloc);
  REQUIRE(knot.get_allocator() == alloc);

  // different dimensions: force reallocation
  knot = generate_knot(4, 1, 0, default_alloc);
  REQUIRE(knot.get_allocator() == alloc);
  REQUIRE(knot.nx == 4);
  REQUIRE(knot.nu == 1);
}

TEST_CASE("knot_vec_basic", "[knot_vec]") {
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

  std::vector<knot_t> vm = std::move(v);
  for (size_t i = 0; i < 10; i++) {
    fmt::println("v2[{:d}].q = {}", i, vm[i].q.transpose());
  }

  std::vector<knot_t> vc{vm};
  for (size_t i = 0; i < 10; i++) {
    fmt::println("vc[{:d}].q = {}", i, vc[i].q.transpose());
    REQUIRE(vm[i] == vc[i]);
  }
}

TEST_CASE("knot_vec_emplace", "[knot_vec]") {
  uint nx = 5;
  uint nu = 2;
  uint nc = 1;
  std::pmr::vector<knot_t> vpmr{alloc};
  REQUIRE(vpmr.get_allocator() == alloc);
  REQUIRE(vpmr.get_allocator().resource() == &rs);

  for (size_t i = 0; i < 20; i++) {
    auto &knot = vpmr.emplace_back(nx, nu, nc);
    REQUIRE(knot.nx == nx);
    REQUIRE(knot.nu == nu);
    REQUIRE(knot.nc == nc);
    REQUIRE(knot.nx2 == nx);
    REQUIRE(knot.get_allocator() == alloc);
  }

  REQUIRE(vpmr.size() == 20);
}

TEST_CASE("problem", "[problem]") {
  INFO("problem");
  uint nx = 4;
  uint nu = 2;
  std::pmr::vector<knot_t> v{alloc};
  v.reserve(10);
  for (int i = 0; i < 10; i++) {
    v.push_back(generate_knot(nx, nu, 0));
  }
  problem_t prob{v, nx, alloc};
  REQUIRE(prob.get_allocator() == alloc);
  REQUIRE(prob.G0.cols() == prob.stages[0].nx);

  problem_t prob_move{std::move(prob)};

  prob_move.addParameterization(1);
  for (size_t i = 0; i < 10; i++) {
    REQUIRE(prob_move.stages[i].nth == 1);
  }
}
