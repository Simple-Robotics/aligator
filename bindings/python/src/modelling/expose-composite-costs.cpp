#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/quad-state-cost.hpp"
#include "aligator/modelling/costs/log-residual-cost.hpp"
#include "aligator/modelling/costs/relaxed-log-barrier.hpp"

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
using PolyManifold = xyz::polymorphic<Manifold>;
using QuadResCost = QuadraticResidualCostTpl<Scalar>;

void exposeComposites() {

  using CompositeData = CompositeCostDataTpl<Scalar>;
  using QuadStateCost = QuadraticStateCostTpl<Scalar>;
  using QuadControlCost = QuadraticControlCostTpl<Scalar>;
  using LogResCost = LogResidualCostTpl<Scalar>;
  using RelaxedLogCost = RelaxedLogBarrierCostTpl<Scalar>;

  PolymorphicMultiBaseVisitor<CostAbstract> visitor;
  bp::class_<QuadResCost, bp::bases<CostAbstract>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<PolyManifold, PolyFunction, const ConstMatrixRef &>(
          bp::args("self", "space", "function", "weights")))
      .def_readwrite("residual", &QuadResCost::residual_)
      .def_readwrite("weights", &QuadResCost::weights_)
      .def(CopyableVisitor<QuadResCost>())
      .def(visitor);

  bp::class_<LogResCost, bp::bases<CostAbstract>>(
      "LogResidualCost", "Weighted log-cost composite cost.",
      bp::init<PolyManifold, PolyFunction, const ConstVectorRef &>(
          bp::args("self", "space", "function", "barrier_weights")))
      .def(bp::init<PolyManifold, PolyFunction, Scalar>(
          bp::args("self", "function", "scale")))
      .def_readwrite("residual", &LogResCost::residual_)
      .def_readwrite("weights", &LogResCost::barrier_weights_)
      .def(CopyableVisitor<LogResCost>())
      .def(visitor);

  bp::class_<RelaxedLogCost, bp::bases<CostAbstract>>(
      "RelaxedLogBarrierCost", "Relaxed log-barrier composite cost.",
      bp::init<PolyManifold, PolyFunction, const ConstVectorRef &,
               const Scalar>(
          bp::args("self", "space", "function", "weights", "threshold")))
      .def(bp::init<PolyManifold, PolyFunction, const Scalar, const Scalar>(
          bp::args("self", "space", "function", "weights", "threshold")))
      .def_readwrite("residual", &RelaxedLogCost::residual_)
      .def_readwrite("weights", &RelaxedLogCost::barrier_weights_)
      .def(CopyableVisitor<RelaxedLogCost>())
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
      .def(bp::init<PolyManifold, const int, const ConstVectorRef &,
                    const MatrixXs &>(
          bp::args("self", "space", "nu", "target", "weights")))
      .add_property("target", &QuadStateCost::getTarget,
                    &QuadStateCost::setTarget,
                    "Target of the quadratic distance.")
      .def(visitor);

  bp::class_<QuadControlCost, bp::bases<QuadResCost>>(
      "QuadraticControlCost", "Quadratic control cost.", bp::no_init)
      .def(bp::init<PolyManifold, ConstVectorRef, const MatrixXs &>(
          bp::args("space", "target", "weights")))
      .def(bp::init<PolyManifold, QuadControlCost::ControlError,
                    const ConstMatrixRef &>(
          bp::args("self", "space", "resdl", "weights")))
      .def(bp::init<PolyManifold, int, const MatrixXs &>(
          bp::args("space", "nu", "weights")))
      .add_property("target", &QuadControlCost::getTarget,
                    &QuadControlCost::setTarget,
                    "Reference of the control cost.")
      .def(visitor);
}
} // namespace python
} // namespace aligator
