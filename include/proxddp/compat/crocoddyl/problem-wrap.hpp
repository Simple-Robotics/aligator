#pragma once

#include <crocoddyl/core/optctrl/shooting.hpp>
#include "proxddp/core/traj-opt-problem.hpp"
#include "proxddp/compat/crocoddyl/action-model.hpp"

namespace proxddp {
namespace compat {
namespace croc {
template <typename Scalar> struct ProblemWrapper {
  using CrocProblem = ::crocoddyl::ShootingProblemTpl<Scalar>;
  CrocProblem problem_;
  ProblemWrapper(const CrocProblem &problem) : problem_(problem) {}
};

} // namespace croc
} // namespace compat
} // namespace proxddp
