#include "proxddp/core/solver-proxddp.hpp"

namespace proxddp {

template SolverProxDDP<context::Scalar>::SolverProxDDP(
    const context::Scalar, const context::Scalar, const context::Scalar,
    const std::size_t, VerboseLevel, HessianApprox);

} // namespace proxddp
