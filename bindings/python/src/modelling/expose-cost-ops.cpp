/// @file
/// @copyright Copyright (C) 2023 LAAS-CNRS, INRIA
#include "aligator/modelling/costs/cost-direct-sum.hpp"

#include "aligator/python/polymorphic-convertible.hpp"

namespace aligator {
namespace python {

using context::CostAbstract;
using context::Manifold;
using context::Scalar;
using DirectSumCost = DirectSumCostTpl<Scalar>;
using PolyCost = xyz::polymorphic<CostAbstract>;

void exposeCostOps() {
  bp::class_<DirectSumCost, bp::bases<CostAbstract>>("DirectSumCost",
                                                     bp::no_init)
      .def(bp::init<xyz::polymorphic<CostAbstract>,
                    xyz::polymorphic<CostAbstract>>(
          bp::args("self", "cost1", "cost2"))[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .def_readonly("cost1", &DirectSumCost::c1_)
      .def_readonly("cost2", &DirectSumCost::c2_)
      .def(PolymorphicVisitor<PolyCost>());

  bp::class_<DirectSumCost::Data, bp::bases<context::CostData>>(
      "DirectSumCostData", bp::no_init)
      .def_readonly("data1", &DirectSumCost::Data::data1_)
      .def_readonly("data2", &DirectSumCost::Data::data2_);

  bp::def("directSum", directSum<Scalar>, bp::args("cost1", "cost2"),
          "Perform the direct sum of two cost functions, :math:`l_3(x,u) = "
          "l_1(x_1,u_1) + l_2(x_2,u_2)`",
          bp::with_custodian_and_ward_postcall<
              0, 1, bp::with_custodian_and_ward_postcall<0, 2>>());
}

} // namespace python
} // namespace aligator
