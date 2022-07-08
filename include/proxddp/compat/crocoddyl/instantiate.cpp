/**
 * @file    instantiate.cpp
 * @details Instantiates the templates for the chosen context's Scalar type.
 */

#include "proxddp/compat/crocoddyl/cost-abstract.hpp"
#include "proxddp/compat/crocoddyl/context.hpp"

namespace proxddp {
namespace compat {
namespace croc {

using context::Scalar;
using CostWrapper = CostAbstractWrapper<Scalar>;

} // namespace croc
} // namespace compat
} // namespace proxddp
