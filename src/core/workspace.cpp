/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "proxddp/core/workspace.hpp"

namespace proxddp {

template WorkspaceTpl<context::Scalar>::WorkspaceTpl(
    const context::TrajOptProblem &, LDLTChoice);

}
