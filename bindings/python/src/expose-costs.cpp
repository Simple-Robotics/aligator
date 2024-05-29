/// @file
/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include <proxsuite-nlp/python/polymorphic.hpp>
#include "aligator/python/costs.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/quad-costs.hpp"
#include "aligator/modelling/costs/constant-cost.hpp"

namespace aligator {
namespace python {
using context::ConstMatrixRef;
using context::ConstVectorRef;
using context::CostAbstract;
using context::CostData;
using context::Manifold;
using context::MatrixXs;
using context::Scalar;
using context::VectorXs;
using internal::PyCostFunction;
using QuadraticCost = QuadraticCostTpl<Scalar>;
using PolyCost = xyz::polymorphic<CostAbstract>;

struct CostDataWrapper : CostData, bp::wrapper<CostData> {
  using CostData::CostData;
};

void exposeQuadCost() {

  bp::implicitly_convertible<ConstantCostTpl<Scalar>,
                             xyz::polymorphic<CostAbstract>>();
  bp::class_<ConstantCostTpl<Scalar>, bp::bases<CostAbstract>>(
      "ConstantCost", "A constant cost term.",
      bp::init<xyz::polymorphic<Manifold>, int, Scalar>(
          bp::args("self", "space", "nu", "value")))
      .def_readwrite("value", &ConstantCostTpl<Scalar>::value_)
      .def(CopyableVisitor<ConstantCostTpl<Scalar>>());

  bp::implicitly_convertible<QuadraticCost, xyz::polymorphic<CostAbstract>>();
  bp::class_<QuadraticCost, bp::bases<CostAbstract>>(
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
      .def_readwrite("w_x", &QuadraticCost::Wxx_, "Weights on the state.")
      .def_readwrite("w_u", &QuadraticCost::Wuu_, "Weights on the control.")
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

/// Composite cost functions.
void exposeComposites();

/// Centroidal cost functions.
void exposeContactMap();
void exposeCentroidalFunctions();
/// fwd-declare exposeCostStack()
void exposeCostStack();

void exposeCostAbstract() {
  proxsuite::nlp::python::register_polymorphic_to_python<PolyCost>();
  bp::class_<PyCostFunction<>, boost::noncopyable>(
      "CostAbstract", "Base class for cost functions.", bp::no_init)
      .def(bp::init<xyz::polymorphic<Manifold>, const int>(
          bp::args("self", "space", "nu")))
      .def("evaluate", bp::pure_virtual(&CostAbstract::evaluate),
           bp::args("self", "x", "u", "data"), "Evaluate the cost function.")
      .def("computeGradients",
           bp::pure_virtual(&CostAbstract::computeGradients),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function gradients.")
      .def("computeHessians", bp::pure_virtual(&CostAbstract::computeHessians),
           bp::args("self", "x", "u", "data"),
           "Compute the cost function hessians.")
      .def_readonly("space", &CostAbstract::space)
      .add_property("nx", &CostAbstract::nx)
      .add_property("ndx", &CostAbstract::ndx)
      .add_property("nu", &CostAbstract::nu)
      .def(CreateDataPythonVisitor<CostAbstract>());

  bp::register_ptr_to_python<shared_ptr<CostData>>();
  bp::class_<CostDataWrapper, boost::noncopyable>(
      "CostData", "Cost function data struct.", bp::no_init)
      .def(bp::init<const int, const int>(bp::args("self", "ndx", "nu")))
      .def(bp::init<const CostAbstract &>(bp::args("self", "cost")))
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

  StdVectorPythonVisitor<std::vector<PolyCost>, true>::expose(
      "StdVec_CostAbstract");
  StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>, true>::expose(
      "StdVec_CostData");
}

void exposeCostOps();

void exposeCosts() {
  exposeCostAbstract();
  exposeCostStack();
  exposeQuadCost();
  exposeComposites();
  exposeCostOps();
  exposeContactMap();
  exposeCentroidalFunctions();
}

} // namespace python
} // namespace aligator
