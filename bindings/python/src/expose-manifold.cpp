#include "aligator/python/fwd.hpp"

#include "aligator/core/manifold-base.hpp"
#include "aligator/core/vector-space.hpp"
#include "aligator/modelling/spaces/cartesian-product.hpp"
#include "aligator/modelling/spaces/tangent-bundle.hpp"

#include <proxsuite-nlp/python/polymorphic.hpp>
#include <eigenpy/std-vector.hpp>

namespace aligator::python {
using namespace proxsuite::nlp::python;
using context::ConstVectorRef;
using context::Manifold;
using context::MatrixRef;
using context::Scalar;
using context::VectorRef;
using PolyManifold = polymorphic<Manifold>;
using CartesianProduct = CartesianProductTpl<Scalar>;

void exposeManifoldBase();
void exposeCartesianProduct();

void exposeManifolds() {

  exposeManifoldBase();

  /* Basic vector space */
  bp::class_<VectorSpaceTpl<Scalar>, bp::bases<Manifold>>(
      "VectorSpace", "Basic Euclidean vector space.", bp::no_init)
      .def(bp::init<const int>(("self"_a, "dim")))
      .def(PolymorphicVisitor<PolyManifold>())
      .enable_pickling_(true);

  exposeCartesianProduct();
}

void exposeManifoldBase() {
  using context::MatrixXs;
  using context::VectorXs;
  register_polymorphic_to_python<PolyManifold>();

  using BinaryFunTypeRet = VectorXs (Manifold::*)(const ConstVectorRef &,
                                                  const ConstVectorRef &) const;
  using BinaryFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, VectorRef) const;
  using JacobianFunType = void (Manifold::*)(
      const ConstVectorRef &, const ConstVectorRef &, MatrixRef, int) const;

  bp::class_<Manifold, boost::noncopyable>(
      "ManifoldAbstract", "Manifold abstract class.", bp::no_init)
      .add_property("nx", &Manifold::nx, "Manifold representation dimension.")
      .add_property("ndx", &Manifold::ndx, "Tangent space dimension.")
      .def(
          "neutral", +[](const Manifold &m) { return m.neutral(); }, "self"_a,
          "Get the neutral point from the manifold (if a Lie group).")
      .def(
          "rand", +[](const Manifold &m) { return m.rand(); }, "self"_a,
          "Sample a random point from the manifold.")
      .def("isNormalized", &Manifold::isNormalized, ("self"_a, "x"),
           "Check if the input vector :math:`x` is a viable element of the "
           "manifold.")
      .def<BinaryFunType>("integrate", &Manifold::integrate,
                          ("self"_a, "x", "v", "out"))
      .def<BinaryFunType>("difference", &Manifold::difference,
                          ("self"_a, "x0", "x1", "out"))
      .def<BinaryFunTypeRet>("integrate", &Manifold::integrate,
                             ("self"_a, "x", "v"))
      .def<BinaryFunTypeRet>("difference", &Manifold::difference,
                             ("self"_a, "x0", "x1"))
      .def("interpolate",
           (void (Manifold::*)(const ConstVectorRef &, const ConstVectorRef &,
                               const Scalar &, VectorRef)
                const)(&Manifold::interpolate),
           ("self"_a, "x0", "x1", "u", "out"))
      .def("interpolate",
           (VectorXs (Manifold::*)(const ConstVectorRef &,
                                   const ConstVectorRef &, const Scalar &)
                const)(&Manifold::interpolate),
           ("self"_a, "x0", "x1", "u"),
           "Interpolate between two points on the manifold. Allocated version.")
      .def<JacobianFunType>("Jintegrate", &Manifold::Jintegrate,
                            ("self"_a, "x", "v", "Jout", "arg"),
                            "Compute the Jacobian of the exp operator.")
      .def<JacobianFunType>("Jdifference", &Manifold::Jdifference,
                            ("self"_a, "x0", "x1", "Jout", "arg"),
                            "Compute the Jacobian of the log operator.")
      .def(
          "Jintegrate",
          +[](const Manifold &m, const ConstVectorRef x,
              const ConstVectorRef &v, int arg) {
            MatrixXs Jout(m.ndx(), m.ndx());
            m.Jintegrate(x, v, Jout, arg);
            return Jout;
          },
          ("self"_a, "x", "v", "arg"),
          "Compute and return the Jacobian of the exp.")
      .def("JintegrateTransport", &Manifold::JintegrateTransport,
           ("self"_a, "x", "v", "J", "arg"),
           "Perform parallel transport of matrix J expressed at point x+v to "
           "point x.")
      .def(
          "Jdifference",
          +[](const Manifold &m, const ConstVectorRef x0,
              const ConstVectorRef &x1, int arg) {
            MatrixXs Jout(m.ndx(), m.ndx());
            m.Jdifference(x0, x1, Jout, arg);
            return Jout;
          },
          ("self"_a, "x0", "x1", "arg"),
          "Compute and return the Jacobian of the log.")
      .def("tangent_space", &Manifold::tangentSpace, bp::args("self"),
           "Returns an object representing the tangent space to this manifold.")
      .def(
          "__mul__",
          +[](const PolyManifold &a, const PolyManifold &b) { return a * b; })
      .def(
          "__mul__", +[](const PolyManifold &a,
                         const CartesianProduct &b) { return a * b; })
      .def(
          "__rmul__", +[](const PolyManifold &a, const CartesianProduct &b) {
            return a * b;
          });

  eigenpy::StdVectorPythonVisitor<std::vector<PolyManifold>>::expose(
      "StdVec_Manifold",
      eigenpy::details::overload_base_get_item_for_std_vector<
          std::vector<PolyManifold>>());
}

} // namespace aligator::python
