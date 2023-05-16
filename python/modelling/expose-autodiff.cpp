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
  using context::FunctionData;
  using context::Manifold;
  using context::Scalar;
  using context::StageFunction;

  bp::enum_<FDLevel>("FDLevel", "Finite difference level (to compute Jacobians "
                                "or both Jacobians and Hessians).")
      .value("ToC1", FDLevel::TOC1)
      .value("ToC2", FDLevel::TOC2);

  using fdiff_wrapper = finite_difference_wrapper<Scalar, FDLevel::TOC1>;

  {
    bp::scope _ = bp::class_<fdiff_wrapper, bp::bases<StageFunction>>(
        "FiniteDifferenceHelper",
        "Make a function into a differentiable function using"
        " finite differences.",
        bp::init<shared_ptr<Manifold>, shared_ptr<StageFunction>, Scalar>(
            bp::args("self", "func", "eps")));
    bp::class_<fdiff_wrapper::Data, bp::bases<FunctionData>>("Data",
                                                             bp::no_init);
  }
}

void exposeAutodiff() { expose_finite_differences(); }

} // namespace python
} // namespace proxddp
