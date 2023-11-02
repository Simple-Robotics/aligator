#include "aligator/python/fwd.hpp"

#include "aligator/modelling/costs/composite-costs.hpp"
#include "aligator/modelling/costs/quad-state-cost.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::CostBase;
using context::CostData;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::StageFunction;
using FunctionPtr = shared_ptr<StageFunction>;
using ManifoldPtr = shared_ptr<Manifold>;

void exposeComposites() {

  using CompositeData = CompositeCostDataTpl<Scalar>;
  using QuadResCost = QuadraticResidualCostTpl<Scalar>;
  using QuadStateCost = QuadraticStateCostTpl<Scalar>;
  using QuadControlCost = QuadraticControlCostTpl<Scalar>;
  using LogResCost = LogResidualCostTpl<Scalar>;

  bp::class_<QuadResCost, bp::bases<CostBase>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<ManifoldPtr, FunctionPtr, const ConstMatrixRef &>(
          bp::args("self", "space", "function", "weights")))
      .def_readwrite("residual", &QuadResCost::residual_)
      .def_readwrite("weights", &QuadResCost::weights_)
      .def(CopyableVisitor<QuadResCost>());

  bp::class_<LogResCost, bp::bases<CostBase>>(
      "LogResidualCost", "Weighted log-cost composite cost.",
      bp::init<ManifoldPtr, FunctionPtr, const ConstVectorRef &>(
          bp::args("self", "space", "function", "barrier_weights")))
      .def(bp::init<ManifoldPtr, FunctionPtr, Scalar>(
          bp::args("self", "function", "scale")))
      .def_readwrite("residual", &LogResCost::residual_)
      .def_readwrite("weights", &LogResCost::barrier_weights_)
      .def(CopyableVisitor<LogResCost>());

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
      .def(bp::init<shared_ptr<QuadStateCost::StateError>, const MatrixXs &>(
          bp::args("self", "resdl", "weights")))
      .def(bp::init<ManifoldPtr, const int, const ConstVectorRef &,
                    const MatrixXs &>(
          bp::args("self", "space", "nu", "target", "weights")))
      .add_property("target", &QuadStateCost::getTarget,
                    &QuadStateCost::setTarget,
                    "Target of the quadratic distance.");

  bp::class_<QuadControlCost, bp::bases<QuadResCost>>(
      "QuadraticControlCost", "Quadratic control cost.", bp::no_init)
      .def(bp::init<ManifoldPtr, ConstVectorRef, const MatrixXs &>(
          bp::args("space", "target", "weights")))
      .def(bp::init<ManifoldPtr, shared_ptr<QuadControlCost::Error>,
                    const MatrixXs &>(
          bp::args("self", "space", "resdl", "weights")))
      .def(bp::init<ManifoldPtr, int, const MatrixXs &>(
          bp::args("space", "nu", "weights")))
      .add_property("target", &QuadControlCost::getTarget,
                    &QuadControlCost::setTarget,
                    "Reference of the control cost.");
}
} // namespace python
} // namespace aligator
