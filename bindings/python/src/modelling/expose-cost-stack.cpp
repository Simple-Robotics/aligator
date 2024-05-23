#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/sum-of-costs.hpp"

namespace aligator {
namespace python {
using context::CostAbstract;
using context::CostData;
using context::Manifold;
using context::Scalar;

void exposeCostStack() {
  using CostStack = CostStackTpl<Scalar>;
  using CostStackData = CostStackDataTpl<Scalar>;

  bp::class_<CostStack, bp::bases<CostAbstract>>(
      "CostStack", "A weighted sum of other cost functions.",
      bp::init<shared_ptr<Manifold>, int,
               const std::vector<shared_ptr<CostAbstract>> &,
               const std::vector<Scalar> &>(("self"_a, "space", "nu",
                                             "components"_a = bp::list(),
                                             "weights"_a = bp::list())))
      .def_readwrite("components", &CostStack::components_,
                     "Components of this cost stack.")
      .def_readonly("weights", &CostStack::weights_,
                    "Weights of this cost stack.")
      .def("addCost", &CostStack::addCost, ("self"_a, "cost", "weight"_a = 1.),
           "Add a cost to the stack of costs.")
      .def("size", &CostStack::size, "Get the number of cost components.")
      .def(CopyableVisitor<CostStack>());

  bp::register_ptr_to_python<shared_ptr<CostStackData>>();
  bp::class_<CostStackData, bp::bases<CostData>>(
      "CostStackData", "Data struct for CostStack.", bp::no_init)
      .def_readonly("sub_cost_data", &CostStackData::sub_cost_data);
}

} // namespace python
} // namespace aligator
