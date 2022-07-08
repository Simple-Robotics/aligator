#pragma once

#include "proxddp/core/solver-workspace.hpp"
#include "crocoddyl/core/optctrl/shooting.hpp"

namespace proxddp {
namespace compat {
namespace croc {

template <typename Scalar> struct CrocWorkspace : WorkspaceTpl<Scalar> {};

} // namespace croc
} // namespace compat
} // namespace proxddp
