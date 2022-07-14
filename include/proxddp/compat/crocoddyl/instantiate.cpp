/**
 * @file    instantiate.cpp
 * @details Instantiates the templates for the chosen context's Scalar type.
 */

#include "proxddp/compat/crocoddyl/cost.hpp"
#include "proxddp/compat/crocoddyl/action-model.hpp"
#include "proxddp/compat/crocoddyl/context.hpp"

namespace proxddp {
namespace compat {
namespace croc {

using context::Scalar;
using CostWrapper = CrocCostWrapperTpl<Scalar>;
using ActionModelWrapper = ActionModelWrapperTpl<Scalar>;

} // namespace croc
} // namespace compat
} // namespace proxddp
