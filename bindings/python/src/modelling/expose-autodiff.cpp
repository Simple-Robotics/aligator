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

  {
    using FiniteDiffType = FiniteDifferenceHelper<Scalar>;
    bp::scope _ = bp::class_<FiniteDiffType, bp::bases<StageFunction>>(
        "FiniteDifferenceHelper",
        "Make a function into a differentiable function/dynamics using"
        " finite differences.",
        bp::init<shared_ptr<Manifold>, shared_ptr<StageFunction>, const Scalar>(
            bp::args("self", "space", "func", "eps")));
    bp::class_<FiniteDiffType::Data, bp::bases<StageFunctionData>>("Data",
                                                                   bp::no_init);
  }

  {
    using DynFiniteDiffType = DynamicsFiniteDifferenceHelper<Scalar>;
    bp::scope _ = bp::class_<DynFiniteDiffType, bp::bases<DynamicsModel>>(
        "DynamicsFiniteDifferenceHelper",
        bp::init<shared_ptr<Manifold>, shared_ptr<DynamicsModel>, const Scalar>(
            bp::args("self", "space", "dyn", "eps")));
    bp::class_<DynFiniteDiffType::Data>("Data", bp::no_init);
  }

  {
    using CostFiniteDiffType = CostFiniteDifferenceHelper<Scalar>;
    bp::scope _ =
        bp::class_<CostFiniteDiffType, bp::bases<CostBase>>(
            "CostFiniteDifference",
            "Define a cost function's derivatives using finite differences.",
            bp::no_init)
            .def(bp::init<shared_ptr<CostBase>, Scalar>(
                bp::args("self", "cost", "fd_eps")));
    bp::class_<CostFiniteDiffType::Data, bp::bases<CostData>>("Data",
                                                              bp::no_init)
        .def_readonly("c1", &CostFiniteDiffType::Data::c1)
        .def_readonly("c2", &CostFiniteDiffType::Data::c2);
  }
}

} // namespace python
} // namespace proxddp
