#include "aligator/solvers/proxddp/solver-proxddp.hpp"
#include "aligator/gar/riccati-base.hpp"
#include "aligator/core/cost-abstract.hpp"

namespace aligator {

template struct SolverProxDDPTpl<context::Scalar>;

} // namespace aligator
