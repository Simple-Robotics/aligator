#include "aligator/gar/lqr-problem.hpp"
#include "aligator/fmt-eigen.hpp"
#include "aligator/core/mimalloc-resource.hpp"
#include "aligator/fmt.hpp"

#include <catch2/catch_test_macros.hpp>

#include "test_util.hpp"

using namespace aligator::gar;
using Eigen::MatrixXd;
using Eigen::VectorXd;
static aligator::mimalloc_resource rs;
static aligator::polymorphic_allocator alloc{&rs};
static aligator::polymorphic_allocator default_alloc{};

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
    knot.f.setZero();
  }
};

TEST_CASE_METHOD(knot_fixture, "move", "[knot]") {

  MatrixXd Q = knot.Q;
  MatrixXd R = knot.R;

  SECTION("move") {
    knot_t knot_moved{std::move(knot)};
    REQUIRE(knot_moved.nx == nx);
    REQUIRE(knot_moved.nu == nu);
    REQUIRE(knot_moved.Q.isApprox(Q));
    REQUIRE(knot_moved.R.isApprox(R));
  }

  SECTION("move extended") {
    knot_t knot_moved2{std::move(knot), alloc};
    REQUIRE(knot_moved2.get_allocator() == alloc);
    REQUIRE(knot_moved2.Q.isApprox(Q));
    REQUIRE(knot_moved2.R.isApprox(R));
  }
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
  knot_t knot2 = generateKnot({nx, nu}, alloc);
  this->knot = std::move(knot2);
}

TEST_CASE("copy_assignment_diff_allocator", "[knot]") {
  // test copying data from a different allocator
  knot_t knot{2, 2, 0, alloc};
  REQUIRE(knot.get_allocator() == alloc);

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

  SECTION("move assignment (same dims)") {
    knot = generateKnot({2, 2, 1}, default_alloc);
    REQUIRE(knot.get_allocator() == alloc);
    REQUIRE(knot.nx == 2);
    REQUIRE(knot.nu == 2);
    REQUIRE(knot.nc == 1);
  }

  SECTION("move assignment (different dims, no realloc)") {
    knot = generateKnot({4, 1, 0}, default_alloc);
    REQUIRE(knot.get_allocator() == alloc);
    REQUIRE(knot.nx == 4);
    REQUIRE(knot.nu == 1);
  }

  SECTION("move assignment (larger dims, realloc)") {
    knot = generateKnot({4, 2, 2}, default_alloc);
    REQUIRE(knot.get_allocator() == alloc);
    REQUIRE(knot.nx == 4);
    REQUIRE(knot.nu == 2);
    REQUIRE(knot.nc == 2);
  }
}

TEST_CASE("knot_vec_basic", "[knot_vec]") {
  uint nx = 4;
  uint nu = 2;
  std::pmr::vector<knot_t> v{alloc};
  v.reserve(10);
  for (size_t i = 0; i < 10; i++) {
    v.push_back(generateKnot({nx, nu}));
    fmt::println("v [{:d}].q = {}", i, v[i].q.transpose());
  }

  SECTION("move ctor") {
    std::pmr::vector<knot_t> vm{std::move(v)};
    REQUIRE(vm.get_allocator() == alloc);
    for (const auto &knot : vm)
      REQUIRE(knot.get_allocator() == alloc);
  }

  SECTION("move assignment") {
    std::pmr::vector<knot_t> vm = std::move(v);
    REQUIRE(vm.get_allocator() == alloc);
    for (size_t i = 0; i < 10; i++) {
      fmt::println("v2[{:d}].q = {}", i, vm[i].q.transpose());
    }
  }

  SECTION("copy ctor") {
    std::pmr::vector<knot_t> vc{v};
    REQUIRE(vc.get_allocator() == default_alloc);
    for (size_t i = 0; i < 10; i++) {
      fmt::println("vc[{:d}].q = {}", i, vc[i].q.transpose());
      REQUIRE(v[i] == vc[i]);
    }
  }

  SECTION("copy ctor (extended)") {
    std::pmr::vector<knot_t> vc{v, alloc};
    REQUIRE(vc.get_allocator() == alloc);
    for (size_t i = 0; i < 10; i++) {
      fmt::println("vc[{:d}].q = {}", i, vc[i].q.transpose());
      REQUIRE(v[i] == vc[i]);
    }
  }
}

TEST_CASE("knot_vec_emplace", "[knot_vec]") {
  const uint nx = 5;
  const uint nu = 2;
  const uint nc = 1;
  std::pmr::vector<knot_t> vpmr{alloc};
  REQUIRE(vpmr.get_allocator() == alloc);

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
    v.push_back(generateKnot({nx, nu}));
  }
  problem_t prob{v, nx, alloc};
  REQUIRE(prob.get_allocator() == alloc);
  REQUIRE(prob.G0.cols() == prob.stages[0].nx);

  SECTION("move ctor") {
    problem_t prob2{std::move(prob)};
    REQUIRE(prob2.get_allocator() == alloc);

    prob2.addParameterization(1);
    for (auto &stage : prob2.stages)
      REQUIRE(stage.nth == 1);
  }

  SECTION("move ctor (extended)") {
    problem_t prob2{std::move(prob), default_alloc};
    REQUIRE(prob2.get_allocator() == default_alloc);

    prob2.addParameterization(2);
    for (auto &stage : prob2.stages)
      REQUIRE(stage.nth == 2);
  }
}
