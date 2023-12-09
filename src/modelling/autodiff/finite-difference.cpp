#include "proxddp/modelling/autodiff/finite-difference.hpp"

namespace proxddp {
namespace autodiff {

template struct FiniteDifferenceHelper<context::Scalar>;
template struct DynamicsFiniteDifferenceHelper<context::Scalar>;
template struct CostFiniteDifferenceHelper<context::Scalar>;

} // namespace autodiff
} // namespace proxddp
