#pragma once

#include "proxddp/context.hpp"

namespace proxddp {

extern template struct StageModelTpl<context::Scalar>;

extern template struct StageDataTpl<context::Scalar>;

}
