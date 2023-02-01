#pragma once

#include "proxddp/context.hpp"
#include "proxddp/core/workspace.hpp"

namespace proxddp {

extern template
unique_ptr<ldlt_base<context::Scalar>>
allocate_ldlt_algorithm(const std::vector<isize> &,
                            const std::vector<isize> &,
                            LDLTChoice);

extern template struct WorkspaceTpl<context::Scalar>;

} // namespace proxddp
