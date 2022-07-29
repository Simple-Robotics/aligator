// #include "proxnlp/python/fwd.hpp"

// #include "proxnlp/manifold-base.hpp"
// #include "proxnlp/modelling/autodiff/finite-difference.hpp"

#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/autodiff/finite-difference.hpp"

namespace proxddp {
namespace python {

/// Expose finite difference helpers.
void expose_finite_differences() {
  using namespace autodiff;
  using context::StageFunction;
  using context::Manifold;
  using context::Scalar;

  bp::enum_<FDLevel>("FDLevel", "Finite difference level (to compute Jacobians or both Jacobians and Hessians).")
      .value("ToC1", FDLevel::TOC1)
      .value("ToC2", FDLevel::TOC2);

  bp::class_<finite_difference_wrapper<Scalar, FDLevel::TOC1>,
             bp::bases<StageFunction>>(
      "FiniteDifferenceHelper",
      "Make a function into a differentiable function using"
      " finite differences.",
      bp::init<const Manifold &, const StageFunction &, const Scalar>(
          bp::args("self", "func", "eps")));

}

void exposeAutodiff() { expose_finite_differences(); }

} // namespace python
} // namespace proxddp