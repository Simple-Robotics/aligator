#pragma once

#include "proxddp/context.hpp"
#include "./workspace.hpp"

namespace aligator {

extern template struct WorkspaceFDDPTpl<context::Scalar>;

}
