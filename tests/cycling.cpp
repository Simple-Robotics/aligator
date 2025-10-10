#include "aligator/core/traj-opt-problem.hpp"
#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/core/explicit-dynamics.hpp"
#include "aligator/core/cost-abstract.hpp"
#include "aligator/utils/mpc-util.hpp"
#include "aligator/modelling/constraints/equality-constraint.hpp"
#include "aligator/modelling/spaces/pinocchio-groups.hpp"
#include "aligator/third-party/polymorphic_cxx14.h"
#include <catch2/catch_test_macros.hpp>

using namespace aligator;
using context::Manifold;
using context::SolverProxDDP;
using ManifoldPtr = xyz::polymorphic<Manifold>;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct MyModel : ExplicitDynamicsModelTpl<double> {
  using ExplicitData = ExplicitDynamicsDataTpl<double>;
  explicit MyModel(polymorphic<Manifold> space)
      : ExplicitDynamicsModelTpl<double>(std::move(space), space->ndx()) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ExplicitData &data) const {
    space_next().integrate(x, u, data.xnext_);
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ExplicitData &data) const {
    space_next().Jintegrate(x, u, data.Jx(), 0);
    space_next().Jintegrate(x, u, data.Ju(), 1);
  }
};

struct MyCost : CostAbstractTpl<double> {
  using CostAbstractTpl<double>::CostAbstractTpl;
  void evaluate(const ConstVectorRef &, const ConstVectorRef &,
                CostData &data) const {
    data.value_ = 0.;
  }

  void computeGradients(const ConstVectorRef &, const ConstVectorRef &,
                        CostData &data) const {
    data.grad_.setZero();
  }

  void computeHessians(const ConstVectorRef &, const ConstVectorRef &,
                       CostData &data) const {
    data.hess_.setZero();
  }
};

using PolyManifold = xyz::polymorphic<context::Manifold>;
using StageModel = StageModelTpl<double>;
using EqualityConstraint = EqualityConstraintTpl<double>;

struct MyFixture {
  SETpl<3, double> space;
  const int nu;
  const MyModel dyn_model;
  const MyCost cost;
  TrajOptProblemTpl<double> problem;
  std::vector<shared_ptr<StageDataTpl<double>>> problem_data;

  MyFixture()
      : space()
      , nu(space.ndx())
      , dyn_model(space)
      , cost(MyCost(space, nu))
      , problem(space.neutral(), nu, space, cost) {
    for (size_t i = 0; i < 20; i++) {
      auto stage = StageModel(cost, dyn_model);
      if (i >= 10) {
        auto func = StateErrorResidualTpl<double>(space, nu, space.neutral());
        stage.addConstraint(func, EqualityConstraint());
      }
      problem.addStage(stage);
      problem_data.push_back(stage.createData());
    }
  }
};

TEST_CASE("test_cycling", "[cycling]") {
  MyFixture f;

  auto nsteps = f.problem.numSteps();
  double tol = 1e-6;
  double mu_init = 1e-8;
  SolverProxDDP ddp(tol, mu_init);
  ddp.rollout_type_ = RolloutType::LINEAR;
  ddp.max_iters = 10;
  ddp.verbose_ = VERBOSE;
  ddp.linear_solver_choice = LQSolverChoice::SERIAL;

  ddp.setup(f.problem);
  bool conv = ddp.run(f.problem);
  REQUIRE(conv);

  for (size_t i = 0; i < 30; i++) {
    f.problem.replaceStageCircular(f.problem.stages_[0]);
    rotate_vec_left(f.problem_data);
    ddp.cycleProblem(f.problem, f.problem_data[nsteps - 1]);
    bool conv = ddp.run(f.problem);
    REQUIRE(conv);
  }
}
