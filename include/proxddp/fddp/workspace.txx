#pragma once

#include "proxddp/context.hpp"
#include "proxddp/fddp/workspace.hpp"

namespace proxddp {

extern template
WorkspaceFDDPTpl<context::Scalar>::WorkspaceFDDPTpl(const context::TrajOptProblem &);

}
