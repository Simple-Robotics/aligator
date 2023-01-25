#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template
WorkspaceTpl<context::Scalar>::WorkspaceTpl(const context::TrajOptProblem &, LDLTChoice);

} // namespace proxddp
