/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/utils/rollout.hpp"

namespace proxddp {
namespace python {
void exposeUtils() {
  using DynamicsType = DynamicsModelTpl<context::Scalar>;
  using ExplicitDynamics = ExplicitDynamicsModelTpl<context::Scalar>;

  using rollout_generic_t = context::VectorOfVectors (*)(
      const DynamicsType &, const context::VectorXs &,
      const context::VectorOfVectors &);

  using rollout_vec_generic_t = context::VectorOfVectors (*)(
      const std::vector<shared_ptr<DynamicsType>> &, const context::VectorXs &,
      const context::VectorOfVectors &);

  using rollout_explicit_t = context::VectorOfVectors (*)(
      const ExplicitDynamics &, const context::VectorXs &,
      const context::VectorOfVectors &);

  using rollout_vec_explicit_t = context::VectorOfVectors (*)(
      const std::vector<shared_ptr<ExplicitDynamics>> &,
      const context::VectorXs &, const context::VectorOfVectors &);

  bp::def<rollout_generic_t>(
      "rollout_implicit", &proxddp::rollout, bp::args("dyn_model", "x0", "us"),
      "Perform a dynamics rollout, for a dynamics model.");

  bp::def<rollout_explicit_t>(
      "rollout", &proxddp::rollout, bp::args("dyn_model", "x0", "us"),
      "Perform a rollout of a single explicit dynamics model.");

  bp::def<rollout_vec_generic_t>(
      "rollout_implicit", &proxddp::rollout, bp::args("dyn_models", "x0", "us"),
      "Perform a dynamics rollout, for multiple discrete dynamics models.");

  bp::def<rollout_vec_explicit_t>(
      "rollout", &proxddp::rollout, bp::args("dyn_models", "x0", "us"),
      "Perform a rollout of multiple explicit dynamics model.");
}

} // namespace python
} // namespace proxddp
