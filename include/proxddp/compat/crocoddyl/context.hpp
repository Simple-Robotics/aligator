/**
 * @file  context.hpp
 * @brief Defines the context for instantiating the templates.
 *
 */
#include "proxddp/compat/crocoddyl/fwd.hpp"
#include "proxddp/context.hpp"

namespace proxddp {
namespace compat {
namespace croc {

namespace context {

using Scalar = ::proxddp::context::Scalar;

using StateWrapper = StateWrapperTpl<Scalar>;
using CostModelWrapper = CrocCostModelWrapperTpl<Scalar>;
using CostDataWrapper = CrocCostDataWrapperTpl<Scalar>;
using DynamicsDataWrapper = DynamicsDataWrapperTpl<Scalar>;
using ActionModelWrapper = CrocActionModelWrapperTpl<Scalar>;
using ActionDataWrapper = CrocActionDataWrapperTpl<Scalar>;

using CrocCostModel = crocoddyl::CostModelAbstractTpl<Scalar>;
using CrocActionModel = crocoddyl::ActionModelAbstractTpl<Scalar>;
using CrocShootingProblem = crocoddyl::ShootingProblemTpl<Scalar>;

} // namespace context
} // namespace croc

} // namespace compat

} // namespace proxddp
