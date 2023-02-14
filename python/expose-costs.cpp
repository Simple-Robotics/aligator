/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/costs.hpp"

#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/constant-cost.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"

namespace proxddp {
namespace python {

void exposeQuadCost() {
  using context::ConstMatrixRef;
  using context::ConstVectorRef;
  using context::CostBase;
  using context::CostData;
  using context::Scalar;
  using QuadraticCost = QuadraticCostTpl<Scalar>;

  bp::class_<ConstantCostTpl<Scalar>, bp::bases<CostBase>>(
      "ConstantCost", "A constant cost term.",
      bp::init<int, int, Scalar>(bp::args("self", "ndx", "nu", "value")))
      .def_readwrite("value", &ConstantCostTpl<Scalar>::value_)
      .def(CopyableVisitor<ConstantCostTpl<Scalar>>());

  bp::class_<QuadraticCost, bp::bases<CostBase>>(
      "QuadraticCost",
      "Quadratic cost in both state and control - only for Euclidean spaces.",
      bp::no_init)
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef,
                    ConstVectorRef, ConstVectorRef>(
          bp::args("self", "w_x", "w_u", "w_cross", "interp_x", "interp_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstVectorRef,
                    ConstVectorRef>(
          bp::args("self", "w_x", "w_u", "interp_x", "interp_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef>(
          "Constructor with just weights (no cross-term).",
          bp::args("self", "w_x", "w_u")))
      .def(bp::init<ConstMatrixRef, ConstMatrixRef, ConstMatrixRef>(
          "Constructor with just weights (with cross-term).",
          bp::args("self", "w_x", "w_u", "w_cross")))
      .def_readwrite("w_x", &QuadraticCost::weights_x, "Weights on the state.")
      .def_readwrite("w_u", &QuadraticCost::weights_u,
                     "Weights on the control.")
      .def_readwrite("interp_x", &QuadraticCost::interp_x)
      .def_readwrite("interp_u", &QuadraticCost::interp_u)
      .add_property("has_cross_term", &QuadraticCost::hasCrossTerm,
                    "Whether there is a cross term.")
      .add_property("weights_cross", &QuadraticCost::getCrossWeights,
                    &QuadraticCost::setCrossWeight, "Cross term weight.")
      .def(CopyableVisitor<QuadraticCostTpl<Scalar>>());

  bp::class_<QuadraticCost::Data, bp::bases<CostData>>(
      "QuadraticCostData", "Quadratic cost data.", bp::no_init);
}

void exposeComposites() {
  using context::CostBase;
  using context::CostData;
  using context::MatrixXs;
  using context::Scalar;
  using context::StageFunction;
  using context::VectorXs;

  using CompositeData = CompositeCostDataTpl<Scalar>;
  using QuadResCost = QuadraticResidualCostTpl<Scalar>;

  bp::class_<QuadResCost, bp::bases<CostBase>>(
      "QuadraticResidualCost", "Weighted 2-norm of a given residual function.",
      bp::init<const shared_ptr<StageFunction> &, const context::MatrixXs &>(
          bp::args("self", "function", "weights")))
      .def_readwrite("residual", &QuadResCost::residual_)
      .def_readwrite("weights", &QuadResCost::weights_)
      .def(CopyableVisitor<QuadResCost>());

  using LogResCost = LogResidualCostTpl<Scalar>;
  bp::class_<LogResCost, bp::bases<CostBase>>(
      "LogResidualCost", "Weighted log-cost composite cost.",
      bp::init<shared_ptr<StageFunction>, context::VectorXs>(
          bp::args("self", "function", "barrier_weights")))
      .def(bp::init<shared_ptr<StageFunction>, Scalar>(
          bp::args("self", "function", "scale")))
      .def_readwrite("residual", &LogResCost::residual_)
      .def_readwrite("weights", &LogResCost::barrier_weights_)
      .def(CopyableVisitor<LogResCost>());

  bp::class_<CompositeData, bp::bases<CostData>>(
      "CompositeCostData",
      bp::init<int, int, shared_ptr<context::FunctionData>>(
          bp::args("self", "ndx", "nu", "rdata")))
      .def_readwrite("residual_data", &CompositeData::residual_data);
}

void exposeCostStack() {
  using context::CostBase;
  using context::CostData;
  using context::Scalar;
  using CostStack = CostStackTpl<Scalar>;
  using CostPtr = CostStack::CostPtr;
  using CostStackData = CostStackDataTpl<Scalar>;

  bp::class_<CostStack, bp::bases<CostBase>>(
      "CostStack", "A weighted sum of other cost functions.",
      bp::init<int, int, const std::vector<CostPtr> &,
               const std::vector<Scalar> &>((
          bp::arg("self"), bp::arg("ndx"), bp::arg("nu"),
          bp::arg("components") = bp::list(), bp::arg("weights") = bp::list())))
      .def_readwrite("components", &CostStack::components_,
                     "Components of this cost stack.")
      .def_readonly("weights", &CostStack::weights_,
                    "Weights of this cost stack.")
      .def("addCost", &CostStack::addCost,
           (bp::arg("self"), bp::arg("cost"), bp::arg("weight") = 1.),
           "Add a cost to the stack of costs.")
      .def("size", &CostStack::size, "Get the number of cost components.")
      .def(CopyableVisitor<CostStack>());

  bp::register_ptr_to_python<shared_ptr<CostStackData>>();
  bp::class_<CostStackData, bp::bases<CostData>>(
      "CostStackData", "Data struct for CostStack.", bp::no_init)
      .add_property(
          "sub_cost_data",
          bp::make_getter(&CostStackData::sub_cost_data,
                          bp::return_value_policy<bp::return_by_value>()));
}

void exposeCosts() {
  using context::CostBase;
  using context::CostData;
  using context::Scalar;
  using context::StageFunction;

  bp::register_ptr_to_python<shared_ptr<CostBase>>();

  bp::class_<internal::PyCostFunction<>, boost::noncopyable>(
      "CostAbstract", "Base class for cost functions.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def("evaluate", bp::pure_virtual(&CostBase::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients", bp::pure_virtual(&CostBase::computeGradients),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians", bp::pure_virtual(&CostBase::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .add_property("ndx", &CostBase::ndx)
      .add_property("nu", &CostBase::nu)
      .def(CreateDataPythonVisitor<CostBase>());

  bp::register_ptr_to_python<shared_ptr<CostData>>();
  bp::class_<CostData>(
      "CostData", "Cost function data struct.",
      bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def_readwrite("value", &CostData::value_)
      .def_readwrite("grad", &CostData::grad_)
      .def_readwrite("hess", &CostData::hess_)
      .add_property(
          "Lx", bp::make_getter(&CostData::Lx_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property(
          "Lu", bp::make_getter(&CostData::Lu_,
                                bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxx", bp::make_getter(
                               &CostData::Lxx_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lxu", bp::make_getter(
                               &CostData::Lxu_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Lux", bp::make_getter(
                               &CostData::Lux_,
                               bp::return_value_policy<bp::return_by_value>()))
      .add_property("Luu", bp::make_getter(
                               &CostData::Luu_,
                               bp::return_value_policy<bp::return_by_value>()));

  StdVectorPythonVisitor<std::vector<shared_ptr<CostBase>>, true>::expose(
      "StdVec_CostAbstract", "Vector of cost objects.");
  StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>, true>::expose(
      "StdVec_CostData", "Vector of CostData objects.");

  exposeCostStack();
  exposeQuadCost();
  exposeComposites();
}

} // namespace python
} // namespace proxddp
