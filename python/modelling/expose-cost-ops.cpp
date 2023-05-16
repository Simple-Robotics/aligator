/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "proxddp/modelling/cost-direct-sum.hpp"

#include "proxddp/python/fwd.hpp"

namespace proxddp {
namespace python {

using context::CostBase;
using context::Manifold;
using context::Scalar;
using DirectSumCost = DirectSumCostTpl<Scalar>;

void exposeCostOps() {
  bp::register_ptr_to_python<shared_ptr<DirectSumCost>>();
  bp::class_<DirectSumCost, bp::bases<CostBase>>("DirectSumCost", bp::no_init)
      .def(bp::init<shared_ptr<CostBase>, shared_ptr<CostBase>>(
          bp::args("self", "cost1", "cost2")))
      .def_readonly("cost1", &DirectSumCost::c1_)
      .def_readonly("cost2", &DirectSumCost::c2_);

  bp::class_<DirectSumCost::Data, bp::bases<context::CostData>>(
      "DirectSumCostData", bp::no_init)
      .def_readonly("data1", &DirectSumCost::Data::data1_)
      .def_readonly("data2", &DirectSumCost::Data::data2_);

  bp::def("directSum", directSum<Scalar>, bp::args("cost1", "cost2"),
          "Perform the direct sum of two cost functions, :math:`l_3(x,u) = "
          "l_1(x_1,u_1) + l_2(x_2,u_2)`");
}

} // namespace python
} // namespace proxddp
