#pragma once

#include "proxddp/context.hpp"
#include "./workspace.hpp"

namespace proxddp {

extern template struct WorkspaceFDDPTpl<context::Scalar>;

}
