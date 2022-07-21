/**
 * @file  context.hpp
 * @brief Defines the context for instantiating the templates.
 *
 */
#include "proxddp/compat/crocoddyl/fwd.hpp"

namespace proxddp {
namespace compat {
namespace croc {

namespace context {

using Scalar = double;

using StateWrapper = StateWrapperTpl<Scalar>;
using CostModelWrapper = CrocCostModelWrapperTpl<Scalar>;
using CostDataWrapper = CrocCostDataWrapperTpl<Scalar>;
using DynamicsDataWrapper = DynamicsDataWrapperTpl<Scalar>;
using ActionModelWrapper = CrocActionModelWrapperTpl<Scalar>;
using ActionDataWrapper = CrocActionDataWrapperTpl<Scalar>;

} // namespace context
} // namespace croc

} // namespace compat

} // namespace proxddp
