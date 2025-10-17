#include "aligator/python/fwd.hpp"
#include "aligator/python/visitors.hpp"

#include "aligator/modelling/costs/sum-of-costs.hpp"

#include <eigenpy/std-pair.hpp>
#include <eigenpy/variant.hpp>
#include <eigenpy/std-map.hpp>

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
  using PolyManifold = xyz::polymorphic<Manifold>;
  using CostItem = CostStack::CostItem;
  using CostMap = CostStack::CostMap;
  eigenpy::StdPairConverter<CostItem>::registration();
  eigenpy::VariantConverter<CostKey>::registration();

  {
    bp::scope scope =
        bp::class_<CostStack, bp::bases<CostAbstract>>(
            "CostStack", "A weighted sum of other cost functions.", bp::no_init)
            .def(
                bp::init<PolyManifold, const int, const std::vector<PolyCost> &,
                         const std::vector<Scalar> &>(
                    ("self"_a, "space", "nu", "components"_a = bp::list(),
                     "weights"_a = bp::list())))
            .def(bp::init<const PolyCost &>(("self"_a, "cost")))
            .def(bp::init<PolyManifold, int, const CostMap &>(
                ("self"_a, "components"),
                "Construct the CostStack from a CostMap object."))
            .def_readonly("components", &CostStack::components_,
                          "Components of this cost stack.")
            .def(
                "getComponent",
                +[](CostStack &self, const CostKey &key) -> PolyCost & {
                  if (!self.components_.contains(key)) {
                    PyErr_SetString(PyExc_KeyError, "Key not found.");
                    bp::throw_error_already_set();
                  }
                  auto &c = self.components_.at(key);
                  return c.first;
                },
                ("self"_a, "key"), bp::return_internal_reference<>())
            .def(
                "addCost",
                +[](CostStack &self, const PolyCost &cost, const Scalar weight)
                    -> PolyCost & { return self.addCost(cost, weight).first; },
                bp::return_internal_reference<>(),
                ("self"_a, "cost", "weight"_a = 1.))
            .def(
                "addCost",
                +[](CostStack &self, CostKey key, const PolyCost &cost,
                    const Scalar weight) -> PolyCost & {
                  return self.addCost(key, cost, weight).first;
                },
                bp::return_internal_reference<>(),
                ("self"_a, "key", "cost", "weight"_a = 1.))
            .def("size", &CostStack::size, "Get the number of cost components.")
            .def(CopyableVisitor<CostStack>())
            .def(PolymorphicMultiBaseVisitor<CostAbstract>());
    eigenpy::GenericMapVisitor<CostMap, true>::expose("CostMap");
  }

  {
    bp::register_ptr_to_python<shared_ptr<CostStackData>>();
    bp::scope scope =
        bp::class_<CostStackData, bp::bases<CostData>>(
            "CostStackData", "Data struct for CostStack.", bp::no_init)
            .def_readonly("sub_cost_data", &CostStackData::sub_cost_data);
    eigenpy::GenericMapVisitor<CostStackData::DataMap, true>::expose("CostMap");
  }
}

} // namespace python
} // namespace aligator
