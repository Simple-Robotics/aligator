#pragma once

#include <crocoddyl/core/fwd.hpp>

namespace proxddp {
/// @brief Headers for compatibility modules.
namespace compat {
/// @brief Headers for the Crocoddyl compatibility module.
namespace croc {

template <typename Scalar> struct StateWrapperTpl;

template <typename Scalar> struct CrocCostModelWrapperTpl;

template <typename Scalar> struct CrocCostDataWrapperTpl;

template <typename Scalar> struct DynamicsDataWrapperTpl;

template <typename Scalar> struct ActionModelWrapperTpl;

template <typename Scalar> struct ActionDataWrapperTpl;

} // namespace croc
} // namespace compat
} // namespace proxddp
