#include "aligator/python/fwd.hpp"
#include "aligator/modelling/constraints.hpp"

namespace aligator::python {

using context::ConstraintSet;
using context::ConstVectorRef;
using context::Scalar;

using EqualityConstraint = EqualityConstraintTpl<Scalar>;
using NegativeOrthant = NegativeOrthantTpl<Scalar>;
using L1Penalty = NonsmoothPenaltyL1Tpl<Scalar>;
using ConstraintSetProduct = ConstraintSetProductTpl<Scalar>;
using BoxConstraint = BoxConstraintTpl<Scalar>;

using PolySet = xyz::polymorphic<ConstraintSet>;

template <typename T>
auto exposeSpecificConstraintSet(const char *name, const char *docstring) {
  return bp::class_<T, bp::bases<ConstraintSet>>(name, docstring, bp::no_init)
      .def(PolymorphicVisitor<PolySet>{});
}

void exposeConstraintSets() {
  register_polymorphic_to_python<PolySet>();
  bp::class_<ConstraintSet, boost::noncopyable>(
      "ConstraintSet", "Base class for constraint sets or nonsmooth penalties.",
      bp::no_init)
      .def("evaluate", &ConstraintSet::evaluate, ("self"_a, "z"),
           "Evaluate the constraint indicator function or nonsmooth penalty "
           "on the projection/prox map of :math:`z`.")
      .def("projection", &ConstraintSet::projection, ("self"_a, "z", "zout"))
      .def(
          "projection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.projection(z, zout);
            return zout;
          },
          ("self"_a, "z"))
      .def("normalConeProjection", &ConstraintSet::normalConeProjection,
           ("self"_a, "z", "zout"))
      .def(
          "normalConeProjection",
          +[](const ConstraintSet &c, const ConstVectorRef &z) {
            context::VectorXs zout(z.size());
            c.normalConeProjection(z, zout);
            return zout;
          },
          ("self"_a, "z"))
      .def("applyProjectionJacobian", &ConstraintSet::applyProjectionJacobian,
           ("self"_a, "z", "Jout"), "Apply the projection Jacobian.")
      .def("applyNormalProjectionJacobian",
           &ConstraintSet::applyNormalConeProjectionJacobian,
           ("self"_a, "z", "Jout"),
           "Apply the normal cone projection Jacobian.")
      .def("computeActiveSet", &ConstraintSet::computeActiveSet,
           ("self"_a, "z", "out"))
      .def("evaluateMoreauEnvelope", &ConstraintSet::evaluateMoreauEnvelope,
           ("self"_a, "zin", "zproj"),
           "Evaluate the Moreau envelope with parameter :math:`\\mu`.")
      .def("setProxParameter", &ConstraintSet::setProxParameter,
           ("self"_a, "mu"), "Set proximal parameter.")
      .add_property("mu", &ConstraintSet::mu, "Current proximal parameter.")
      .def(bp::self == bp::self);

  exposeSpecificConstraintSet<EqualityConstraintTpl<Scalar>>(
      "EqualityConstraintSet", "Cast a function into an equality constraint")
      .def(bp::init<>("self"_a));

  exposeSpecificConstraintSet<NegativeOrthantTpl<Scalar>>(
      "NegativeOrthant",
      "Cast a function into a negative inequality constraint h(x) \\leq 0")
      .def(bp::init<>("self"_a));

  exposeSpecificConstraintSet<BoxConstraint>(
      "BoxConstraint",
      "Box constraint of the form :math:`z \\in [z_\\min, z_\\max]`.")
      .def(bp::init<ConstVectorRef, ConstVectorRef>(
          ("self"_a, "lower_limit", "upper_limit")))
      .def_readwrite("upper_limit", &BoxConstraint::upper_limit)
      .def_readwrite("lower_limit", &BoxConstraint::lower_limit);

  exposeSpecificConstraintSet<L1Penalty>("NonsmoothPenaltyL1",
                                         "1-norm penalty function.")
      .def(bp::init<>(("self"_a)));

  exposeSpecificConstraintSet<ConstraintSetProduct>(
      "ConstraintSetProduct", "Cartesian product of constraint sets.")
      .def(bp::init<std::vector<PolySet>, std::vector<Eigen::Index>>(
          ("self"_a, "components", "blockSizes")))
      .add_property("components",
                    bp::make_function(&ConstraintSetProduct::components,
                                      bp::return_internal_reference<>()))
      .add_property("blockSizes",
                    bp::make_function(&ConstraintSetProduct::blockSizes,
                                      bp::return_internal_reference<>()),
                    "Dimensions of each component of the cartesian product.");

  StdVectorPythonVisitor<std::vector<PolySet>>::expose(
      "StdVec_ConstraintObject",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolySet>>());
}

} // namespace aligator::python
