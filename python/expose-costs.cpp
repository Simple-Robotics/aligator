/// @copyright Copyright (C) 2022 LAAS-CNRS, INRIA
#include "proxddp/python/fwd.hpp"
#include "proxddp/python/costs.hpp"

#include "proxddp/modelling/quad-costs.hpp"
#include "proxddp/modelling/composite-costs.hpp"
#include "proxddp/modelling/sum-of-costs.hpp"


namespace proxddp
{
  namespace python
  {
    
    void exposeCosts()
    {
      using context::Scalar;
      using context::StageFunction;
      using context::VectorXs;
      using context::MatrixXs;
      using CostData = CostDataAbstract<Scalar>;

      bp::register_ptr_to_python<shared_ptr<context::CostBase>>();

      bp::class_<internal::PyCostFunction<>>(
        "CostAbstract", "Base class for cost functions.",
        bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
        )
      )
        .def("evaluate", bp::pure_virtual(&context::CostBase::evaluate),
             bp::args("self", "x", "u", "data"),
             "Evaluate the cost function.")
        .def("computeGradients", bp::pure_virtual(&context::CostBase::evaluate),
             bp::args("self", "x", "u", "data"),
             "Compute the cost function gradients.")
        .def("computeHessians", bp::pure_virtual(&context::CostBase::computeHessians),
             bp::args("self", "x", "u", "data"),
             "Compute the cost function hessians.")
        .def(CreateDataPythonVisitor<context::CostBase>());

      bp::class_<QuadraticCost<Scalar>, bp::bases<context::CostBase>>(
        "QuadraticCost", "Quadratic cost in both state and control.",
        bp::init<const MatrixXs&, const MatrixXs&, const VectorXs&, const VectorXs&>(
          bp::args("self", "w_x", "w_u", "interp_x", "interp_u")
        )
      )
        .def(bp::init<const MatrixXs&, const MatrixXs&>(bp::args("self", "w_x", "w_u")))
      ;

      bp::class_<QuadraticResidualCost<Scalar>, bp::bases<context::CostBase>>(
        "QuadraticResidualCost",
        "Weighted 2-norm of a given residual function.",
        bp::init<const shared_ptr<StageFunction>&,
                 const context::MatrixXs&>(
                   bp::args("self", "function", "weights")
                 )
      )
        .def_readwrite("residual", &QuadraticResidualCost<Scalar>::residual_)
        .def_readwrite("weights", &QuadraticResidualCost<Scalar>::weights_)
        .def(CreateDataPythonVisitor<QuadraticResidualCost<Scalar>>());

      bp::class_<CostData, shared_ptr<CostData>>(
        "CostData", "Cost function data struct.",
        bp::init<const int, const int>(
          bp::args("self", "ndx", "nu")
        )
      )
        .def_readwrite("value", &CostData::value_)
        .add_property("Lx",  bp::make_getter(&CostData::Lx_, bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lu",  bp::make_getter(&CostData::Lu_, bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lxx", bp::make_getter(&CostData::Lxx_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lxu", bp::make_getter(&CostData::Lxu_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Lux", bp::make_getter(&CostData::Lux_,bp::return_value_policy<bp::return_by_value>()) )
        .add_property("Luu", bp::make_getter(&CostData::Luu_,bp::return_value_policy<bp::return_by_value>()) )
        .def_readwrite("_grad", &CostData::grad_)
        .def_readwrite("_hessian", &CostData::hess_)
        ;

      pinpy::StdVectorPythonVisitor<std::vector<shared_ptr<CostData>>, true>::expose("StdVec_CostData", "Vector of CostData.");
    }
    
  } // namespace python
} // namespace proxddp

