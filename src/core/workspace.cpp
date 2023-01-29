/// @file
/// @copyright Copyright (C) 2022-2023 LAAS-CNRS, INRIA
#include "proxddp/core/workspace.hpp"

namespace proxddp {

template unique_ptr<ldlt_base<context::Scalar>>
allocate_ldlt_algorithm(const std::vector<isize> &nprims,
                        const std::vector<isize> &nduals, LDLTChoice choice);

template WorkspaceTpl<context::Scalar>::WorkspaceTpl(
    const context::TrajOptProblem &, LDLTChoice);

} // namespace proxddp
