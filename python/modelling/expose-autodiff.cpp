#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/autodiff/finite-difference.hpp"

namespace proxddp {
namespace python {

/// Expose finite difference helpers.
void exposeAutodiff() {
  using namespace autodiff;
  using context::CostBase;
  using context::CostData;
  using context::Manifold;
  using context::Scalar;
  using context::StageFunction;
  using context::StageFunctionData;

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
            bp::args("self", "space", "func", "eps")));
    bp::class_<fdiff_wrapper::Data, bp::bases<StageFunctionData>>("Data",
                                                                  bp::no_init);
  }

  using cost_fdiff = cost_finite_difference_wrapper<Scalar>;
  {
    bp::scope _ =
        bp::class_<cost_fdiff, bp::bases<CostBase>>(
            "CostFiniteDifference",
            "Define a cost function's derivatives using finite differences.",
            bp::no_init)
            .def(bp::init<shared_ptr<CostBase>, Scalar>(
                bp::args("self", "cost", "fd_eps")));
    bp::class_<cost_fdiff::Data, bp::bases<CostData>>("Data", bp::no_init)
        .def_readonly("c1", &cost_fdiff::Data::c1)
        .def_readonly("c2", &cost_fdiff::Data::c2);
  }
}

} // namespace python
} // namespace proxddp
