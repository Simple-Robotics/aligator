/// @copyright Copyright (C) 2022 LAAS-CNRS, 2022-2025 INRIA
#include "aligator/python/fwd.hpp"
#include "aligator/core/dynamics.hpp"
#include "aligator/utils/rollout.hpp"

namespace aligator {
namespace python {
void exposeUtils() {
  using DynamicsType = DynamicsModelTpl<context::Scalar>;
  using ExplicitDynamics = ExplicitDynamicsModelTpl<context::Scalar>;

  using rollout_generic_t = context::VectorOfVectors (*)(
      const DynamicsType &, const context::ConstVectorRef &,
      const context::VectorOfVectors &);

  using rollout_vec_generic_t = context::VectorOfVectors (*)(
      const std::vector<xyz::polymorphic<DynamicsType>> &,
      const context::ConstVectorRef &, const context::VectorOfVectors &);

  using rollout_explicit_t = context::VectorOfVectors (*)(
      const ExplicitDynamics &, const context::ConstVectorRef &,
      const context::VectorOfVectors &);

  using rollout_vec_explicit_t = context::VectorOfVectors (*)(
      const std::vector<xyz::polymorphic<ExplicitDynamics>> &,
      const context::ConstVectorRef &, const context::VectorOfVectors &);

  bp::def<rollout_generic_t>(
      "rollout_implicit", &aligator::rollout, ("dyn_model"_a, "x0", "us"),
      "Perform a dynamics rollout, for a dynamics model.");

  bp::def<rollout_explicit_t>(
      "rollout", &aligator::rollout, ("dyn_model"_a, "x0", "us"),
      "Perform a rollout of a single explicit dynamics model.");

  bp::def<rollout_vec_generic_t>(
      "rollout_implicit", &aligator::rollout, ("dyn_models"_a, "x0", "us"),
      "Perform a dynamics rollout, for multiple discrete dynamics models.");

  bp::def<rollout_vec_explicit_t>(
      "rollout", &aligator::rollout, ("dyn_models"_a, "x0", "us"),
      "Perform a rollout of multiple different dynamics models.");
}

} // namespace python
} // namespace aligator
