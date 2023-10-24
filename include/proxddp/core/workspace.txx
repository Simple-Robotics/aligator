#pragma once

#include "proxddp/context.hpp"
#include "./workspace.hpp"

namespace proxddp {

extern template struct WorkspaceTpl<context::Scalar>;

} // namespace proxddp
