#include "aligator/python/fwd.hpp"

#include "aligator/modelling/spaces/multibody.hpp"
#include "aligator/modelling/spaces/pinocchio-groups.hpp"

#include <proxsuite-nlp/python/polymorphic.hpp>

namespace aligator::python {
using context::Manifold;
using context::Scalar;
using proxsuite::nlp::python::PolymorphicVisitor;
using PolyManifold = xyz::polymorphic<Manifold>;

/// Expose a Pinocchio Lie group with a specified name, docstring,
/// and no-arg default constructor.
template <typename LieGroup>
void exposeLieGroup(const char *name, const char *docstring) {
  bp::class_<PinocchioLieGroup<LieGroup>, bp::bases<Manifold>>(
      name, docstring, bp::init<>("self"_a))
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);
}

/// Expose the tangent bundle of a manifold type @p M.
template <typename M>
bp::class_<TangentBundleTpl<M>, bp::bases<Manifold>>
exposeTangentBundle(const char *name, const char *docstring) {
  using OutType = TangentBundleTpl<M>;
  return bp::class_<OutType, bp::bases<Manifold>>(
             name, docstring, bp::init<M>(("self"_a, "base")))
      .add_property("base",
                    bp::make_function(&OutType::getBaseSpace,
                                      bp::return_internal_reference<>()),
                    "Get the base space.")
      .def(PolymorphicVisitor<PolyManifold>());
}

/// Expose the tangent bundle with an additional constructor.
template <typename M, class Init>
bp::class_<TangentBundleTpl<M>, bp::bases<Manifold>>
exposeTangentBundle(const char *name, const char *docstring, Init init) {
  return exposeTangentBundle<M>(name, docstring).def(init);
}

void exposePinocchioSpaces() {
  namespace pin = pinocchio;
  using pin::ModelTpl;
  using pin::SpecialEuclideanOperationTpl;
  using pin::SpecialOrthogonalOperationTpl;
  using pin::VectorSpaceOperationTpl;

  using DynSizeEuclideanSpace = VectorSpaceOperationTpl<Eigen::Dynamic, Scalar>;
  bp::class_<PinocchioLieGroup<DynSizeEuclideanSpace>, bp::bases<Manifold>>(
      "EuclideanSpace", "Pinocchio's n-dimensional Euclidean vector space.",
      bp::no_init)
      .def(bp::init<int>(("self"_a, "dim")))
      .def(PolymorphicVisitor<PolyManifold>());

  exposeLieGroup<VectorSpaceOperationTpl<1, Scalar>>(
      "R", "One-dimensional Euclidean space AKA real number line.");
  exposeLieGroup<VectorSpaceOperationTpl<2, Scalar>>(
      "R2", "Two-dimensional Euclidean space.");
  exposeLieGroup<VectorSpaceOperationTpl<3, Scalar>>(
      "R3", "Three-dimensional Euclidean space.");
  exposeLieGroup<VectorSpaceOperationTpl<4, Scalar>>(
      "R4", "Four-dimensional Euclidean space.");
  exposeLieGroup<SpecialOrthogonalOperationTpl<2, Scalar>>(
      "SO2", "SO(2) special orthogonal group.");
  exposeLieGroup<SpecialOrthogonalOperationTpl<3, Scalar>>(
      "SO3", "SO(3) special orthogonal group.");
  exposeLieGroup<SpecialEuclideanOperationTpl<2, Scalar>>(
      "SE2", "SE(2) special Euclidean group.");
  exposeLieGroup<SpecialEuclideanOperationTpl<3, Scalar>>(
      "SE3", "SE(3) special Euclidean group.");

  using SO2 = PinocchioLieGroup<SpecialOrthogonalOperationTpl<2, Scalar>>;
  using SO3 = PinocchioLieGroup<SpecialOrthogonalOperationTpl<3, Scalar>>;
  using SE2 = PinocchioLieGroup<SpecialEuclideanOperationTpl<2, Scalar>>;
  using SE3 = PinocchioLieGroup<SpecialEuclideanOperationTpl<3, Scalar>>;

  /* Expose tangent bundles */

  exposeTangentBundle<SO2>("TSO2", "Tangent bundle of the SO(2) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SO3>("TSO3", "Tangent bundle of the SO(3) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SE2>("TSE2", "Tangent bundle of the SE(2) group.")
      .def(bp::init<>(bp::args("self")));
  exposeTangentBundle<SE3>("TSE3", "Tangent bundle of the SE(3) group.")
      .def(bp::init<>(bp::args("self")));

  /* Groups associated w/ Pinocchio models */
  using Multibody = MultibodyConfiguration<Scalar>;
  using Model = ModelTpl<Scalar>;
  bp::class_<Multibody, bp::bases<Manifold>>(
      "MultibodyConfiguration", "Configuration group of a multibody",
      bp::init<const Model &>(bp::args("self", "model")))
      .add_property("model",
                    bp::make_function(&Multibody::getModel,
                                      bp::return_internal_reference<>()),
                    "Return the Pinocchio model instance.")
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);

  using MultiPhase = MultibodyPhaseSpace<Scalar>;
  bp::class_<MultiPhase, bp::bases<Manifold>>(
      "MultibodyPhaseSpace",
      "Tangent space of the multibody configuration group.",
      bp::init<const Model &>(("self"_a, "model")))
      .add_property("model",
                    bp::make_function(&MultiPhase::getModel,
                                      bp::return_internal_reference<>()),
                    "Return the Pinocchio model instance.")
      .add_property("base", bp::make_function(
                                +[](const MultiPhase &m) -> const Multibody & {
                                  return m.getBaseSpace();
                                },
                                bp::return_internal_reference<>()))
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);
}

} // namespace aligator::python
