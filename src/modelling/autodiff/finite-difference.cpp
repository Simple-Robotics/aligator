#include "proxddp/modelling/autodiff/finite-difference.hpp"

namespace proxddp {
namespace autodiff {

template struct finite_difference_wrapper<context::Scalar>;
template struct cost_finite_difference_wrapper<context::Scalar>;

} // namespace autodiff
} // namespace proxddp
