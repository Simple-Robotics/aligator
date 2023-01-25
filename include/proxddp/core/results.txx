/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/results.hpp"


namespace proxddp {

extern template
ResultsTpl<context::Scalar>::ResultsTpl(const context::TrajOptProblem &);

} // namespace proxddp
