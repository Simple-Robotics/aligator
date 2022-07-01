#pragma once
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/core/explicit-dynamics.hpp"

#include <proxnlp/modelling/spaces/pinocchio-groups.hpp>

using namespace proxddp;

/// @brief    Addition dynamics.
/// @details  It maps \f$(x,u)\f$ to \f$ x + u \f$.
struct MyModel : ExplicitDynamicsModelTpl<double> {
  using Manifold = ManifoldAbstractTpl<double>;
  using ExplicitData = ExplicitDynamicsDataTpl<double>;
  MyModel(const shared_ptr<Manifold> &space)
      : ExplicitDynamicsModelTpl<double>(space, space->ndx()) {}

  void forward(const ConstVectorRef &x, const ConstVectorRef &u,
               ExplicitData &data) const {
    out_space().integrate(x, u, data.xnext_);
  }

  void dForward(const ConstVectorRef &x, const ConstVectorRef &u,
                ExplicitData &data) const {
    out_space().Jintegrate(x, u, data.Jx_, 0);
    out_space().Jintegrate(x, u, data.Ju_, 1);
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

// using Manifold = proxnlp::VectorSpaceTpl<double>;
using Manifold = proxnlp::PinocchioLieGroup<
    pinocchio::SpecialEuclideanOperationTpl<3, double>>;
using StageModel = proxddp::StageModelTpl<double>;

struct MyFixture {
  shared_ptr<Manifold> space;
  const int nx;
  const int nu;
  MyModel dyn_model;
  MyCost cost;
  StageModel stage;
  TrajOptProblemTpl<double> problem;

  MyFixture()
      : space(std::make_shared<Manifold>()), nx(space->nx()), nu(space->ndx()),
        dyn_model(space), cost(nx, nu),
        stage(space, nu, shared_ptr<MyCost>(&cost),
              shared_ptr<MyModel>(&dyn_model)),
        problem(space->neutral(), nu, *space, shared_ptr<MyCost>(&cost)) {
    problem.addStage(stage);
    problem.addStage(stage);
  }
};
