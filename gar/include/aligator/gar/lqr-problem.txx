/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#pragma once

#include "aligator/context.hpp"
#include "aligator/gar/lqr-problem.hpp"

namespace aligator {
namespace gar {
extern template struct LqrKnotTpl<context::Scalar>;
extern template struct LqrProblemTpl<context::Scalar>;
} // namespace gar
} // namespace aligator
