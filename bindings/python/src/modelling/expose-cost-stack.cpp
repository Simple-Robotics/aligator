#include "aligator/python/polymorphic-convertible.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/sum-of-costs.hpp"

#include <eigenpy/std-pair.hpp>
#include <eigenpy/variant.hpp>
#if EIGENPY_VERSION_AT_LEAST(3, 9, 1)
#include <eigenpy/std-map.hpp>
#endif

namespace aligator {
namespace python {
using context::CostAbstract;
using context::CostData;
using context::Manifold;
using context::Scalar;

void exposeCostStack() {
  using CostStack = CostStackTpl<Scalar>;
  using CostStackData = CostStackDataTpl<Scalar>;
  using CostKey = CostStack::CostKey;
  using PolyCost = CostStack::PolyCost;
  using CostItem = CostStack::CostItem;
  using CostMap = CostStack::CostMap;
  eigenpy::StdPairConverter<CostItem>::registration();
  eigenpy::VariantConverter<CostKey>::registration();

  {
    bp::scope scope =
        bp::class_<CostStack, bp::bases<CostAbstract>>(
            "CostStack", "A weighted sum of other cost functions.", bp::no_init)
            .def(bp::init<xyz::polymorphic<Manifold>, const int,
                          const std::vector<PolyCost> &,
                          const std::vector<Scalar> &>(
                ("self"_a, "space", "nu", "components"_a = bp::list(),
                 "weights"_a = bp::list())))
            .def(bp::init<const PolyCost &>(("self"_a, "cost")))
            .def_readwrite("components", &CostStack::components_,
                           "Components of this cost stack.")
            .def(
                "addCost",
                +[](CostStack &self, const PolyCost &cost,
                    const Scalar weight) {
                  // return
                  self.addCost(cost, weight);
                },
                ("self"_a, "cost", "weight"_a = 1.),
                bp::return_internal_reference<>())
            .def(
                "addCost",
                +[](CostStack &self, CostKey key, const PolyCost &cost,
                    const Scalar weight) {
                  // return
                  self.addCost(key, cost, weight);
                },
                ("self"_a, "key", "cost", "weight"_a = 1.),
                bp::return_internal_reference<>())
            .def("size", &CostStack::size, "Get the number of cost components.")
            .def(CopyableVisitor<CostStack>())
            .def(PolymorphicMultiBaseVisitor<CostAbstract>());
#if EIGENPY_VERSION_AT_LEAST(3, 9, 1)
    eigenpy::GenericMapVisitor<CostMap, true>::expose("CostMap");
#endif
  }

  {
    bp::register_ptr_to_python<shared_ptr<CostStackData>>();
    bp::scope scope =
        bp::class_<CostStackData, bp::bases<CostData>>(
            "CostStackData", "Data struct for CostStack.", bp::no_init)
            .def_readonly("sub_cost_data", &CostStackData::sub_cost_data);
#if EIGENPY_VERSION_AT_LEAST(3, 9, 1)
    eigenpy::GenericMapVisitor<CostStackData::DataMap, true>::expose("DataMap");
#endif
  }
}

} // namespace python
} // namespace aligator
