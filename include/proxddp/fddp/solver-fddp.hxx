#pragma once

#include "proxddp/fddp/solver-fddp.hpp"

namespace proxddp {

template <typename Scalar>
SolverFDDP<Scalar>::SolverFDDP(const Scalar tol, const Scalar reg_init,
                               VerboseLevel verbose)
    : tol_(tol), xreg_(reg_init), ureg_(reg_init), verbose_(verbose) {}

template <typename Scalar>
void SolverFDDP<Scalar>::setup(const Problem &problem) {
  results_ = std::make_unique<Results>(problem);
  workspace_ = std::make_unique<Workspace>(problem);
}

} // namespace proxddp
