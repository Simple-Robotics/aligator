#pragma once

#include "proxddp/context.hpp"
#include "./results.hpp"

namespace proxddp {

extern template struct ResultsFDDPTpl<context::Scalar>;

}
