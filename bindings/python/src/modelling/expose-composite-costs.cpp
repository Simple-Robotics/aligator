#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/log-residual-cost.hpp"
#include "aligator/python/polymorphic-convertible.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::CostAbstract;
using context::CostData;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::StageFunction;
using PolyFunction = xyz::polymorphic<StageFunction>;
using ManifoldPtr = xyz::polymorphic<Manifold>;
using QuadResCost = QuadraticResidualCostTpl<Scalar>;

void exposeComposites() {

  using CompositeData = CompositeCostDataTpl<Scalar>;
  using QuadStateCost = QuadraticStateCostTpl<Scalar>;
  using QuadControlCost = QuadraticControlCostTpl<Scalar>;
  using LogResCost = LogResidualCostTpl<Scalar>;

  PolymorphicMultiBaseVisitor<CostAbstract> visitor;
  bp::class_<QuadResCost, bp::bases<CostAbstract>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<ManifoldPtr, PolyFunction, const ConstMatrixRef &>(
          bp::args("self", "space", "function", "weights"))
          [bp::with_custodian_and_ward<1, 2,
                                       bp::with_custodian_and_ward<1, 3>>()])
      .def_readwrite("residual", &QuadResCost::residual_)
      .def_readwrite("weights", &QuadResCost::weights_)
      .def(CopyableVisitor<QuadResCost>())
      .def(visitor);

  bp::class_<LogResCost, bp::bases<CostAbstract>>(
      "LogResidualCost", "Weighted log-cost composite cost.",
      bp::init<ManifoldPtr, PolyFunction, const ConstVectorRef &>(
          bp::args("self", "space", "function", "barrier_weights"))
          [bp::with_custodian_and_ward<1, 2,
                                       bp::with_custodian_and_ward<1, 3>>()])
      .def(bp::init<ManifoldPtr, PolyFunction, Scalar>(
          bp::args("self", "function", "scale"))[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .def_readwrite("residual", &LogResCost::residual_)
      .def_readwrite("weights", &LogResCost::barrier_weights_)
      .def(CopyableVisitor<LogResCost>())
      .def(visitor);

  bp::class_<CompositeData, bp::bases<CostData>>(
      "CompositeCostData",
      bp::init<int, int, shared_ptr<context::StageFunctionData>>(
          bp::args("self", "ndx", "nu", "rdata")))
      .def_readwrite("residual_data", &CompositeData::residual_data);

  bp::class_<QuadStateCost, bp::bases<QuadResCost>>(
      "QuadraticStateCost",
      "Quadratic distance over the state manifold. This is a shortcut to "
      "create a `QuadraticResidualCost` over a state error residual.",
      bp::no_init)
      .def(bp::init<QuadStateCost::StateError &, const MatrixXs &>(
          bp::args("self", "resdl", "weights")))
      .def(bp::init<ManifoldPtr, const int, const ConstVectorRef &,
                    const MatrixXs &>(
          bp::args("self", "space", "nu", "target",
                   "weights"))[bp::with_custodian_and_ward<1, 2>()])
      .add_property("target", &QuadStateCost::getTarget,
                    &QuadStateCost::setTarget,
                    "Target of the quadratic distance.")
      .def(visitor);

  bp::class_<QuadControlCost, bp::bases<QuadResCost>>(
      "QuadraticControlCost", "Quadratic control cost.", bp::no_init)
      .def(bp::init<ManifoldPtr, ConstVectorRef, const MatrixXs &>(bp::args(
          "space", "target", "weights"))[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<ManifoldPtr, QuadControlCost::Error, const MatrixXs &>(
          bp::args("self", "space", "resdl",
                   "weights"))[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<ManifoldPtr, int, const MatrixXs &>(bp::args(
          "space", "nu", "weights"))[bp::with_custodian_and_ward<1, 2>()])
      .add_property("target", &QuadControlCost::getTarget,
                    &QuadControlCost::setTarget,
                    "Reference of the control cost.")
      .def(visitor);
}
} // namespace python
} // namespace aligator
