#include "proxddp/python/fwd.hpp"

#include "proxddp/modelling/autodiff/finite-difference.hpp"

namespace proxddp {
namespace python {

/// Expose finite difference helpers.
void exposeAutodiff() {
  using namespace autodiff;
  using context::CostBase;
  using context::CostData;
  using context::DynamicsModel;
  using context::Manifold;
  using context::Scalar;
  using context::StageFunction;
  using context::StageFunctionData;

  bp::enum_<FDLevel>("FDLevel", "Finite difference level (to compute Jacobians "
                                "or both Jacobians and Hessians).")
      .value("ToC1", FDLevel::TOC1)
      .value("ToC2", FDLevel::TOC2);

  using FiniteDiffType = finite_difference_wrapper<Scalar, FDLevel::TOC1>;

  {
    bp::scope _ = bp::class_<FiniteDiffType, bp::bases<StageFunction>>(
        "FiniteDifferenceHelper",
        "Make a function into a differentiable function/dynamics using"
        " finite differences.",
        bp::init<shared_ptr<Manifold>, shared_ptr<StageFunction>, const Scalar>(
            bp::args("self", "space", "func", "eps")));
    bp::class_<FiniteDiffType::Data, bp::bases<StageFunctionData>>("Data",
                                                                   bp::no_init);
  }

  using CostFDType = CostFiniteDifferenceHelper<Scalar>;
  {
    bp::scope _ =
        bp::class_<CostFDType, bp::bases<CostBase>>(
            "CostFiniteDifference",
            "Define a cost function's derivatives using finite differences.",
            bp::no_init)
            .def(bp::init<shared_ptr<CostBase>, Scalar>(
                bp::args("self", "cost", "fd_eps")));
    bp::class_<CostFDType::Data, bp::bases<CostData>>("Data", bp::no_init)
        .def_readonly("c1", &CostFDType::Data::c1)
        .def_readonly("c2", &CostFDType::Data::c2);
  }
}

} // namespace python
} // namespace proxddp
