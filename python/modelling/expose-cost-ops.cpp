#include "proxddp/modelling/cost-direct-sum.hpp"

#include "proxddp/python/fwd.hpp"

namespace proxddp {
namespace python {

using context::CostBase;
using context::Manifold;
using context::Scalar;
using DirectSumCost = DirectSumCostTpl<Scalar>;

void exposeCostOps() {
  bp::class_<DirectSumCost, bp::bases<CostBase>>("DirectSumCost", bp::no_init)
      .def(bp::init<shared_ptr<CostBase>, shared_ptr<CostBase>>(
          bp::args("self", "cost1", "cost2")))
      .def_readonly("cost1", &DirectSumCost::c1_)
      .def_readonly("cost2", &DirectSumCost::c2_);

  bp::class_<DirectSumCost::Data, bp::bases<context::CostData>>(
      "DirectSumCostData", bp::no_init)
      .def_readonly("data1", &DirectSumCost::Data::data1_)
      .def_readonly("data2", &DirectSumCost::Data::data2_);
}

} // namespace python
} // namespace proxddp
