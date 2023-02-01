#pragma once

#include "proxddp/context.hpp"
#include "proxddp/fddp/workspace.hpp"

namespace proxddp {

extern template struct WorkspaceFDDPTpl<context::Scalar>;

}
