#include "aligator/python/fwd.hpp"
#include "aligator/modelling/spaces/cartesian-product.hpp"

#include <proxsuite-nlp/python/polymorphic.hpp>

namespace aligator::python {

using context::Manifold;
using context::Scalar;
using PolymorphicManifold = polymorphic<Manifold>;
using context::ConstVectorRef;
using context::VectorRef;
using context::VectorXs;
using CartesianProduct = CartesianProductTpl<Scalar>;
using proxsuite::nlp::python::PolymorphicVisitor;

std::vector<VectorXs> copy_vec_constref(const std::vector<ConstVectorRef> &x) {
  std::vector<VectorXs> out;
  for (const auto &c : x)
    out.push_back(c);
  return out;
}

void exposeCartesianProduct() {
  const std::string split_doc =
      "Takes an point on the product manifold and splits it up between the two "
      "base manifolds.";
  const std::string split_vec_doc =
      "Takes a tangent vector on the product manifold and splits it up.";

  using MutSplitSig =
      std::vector<VectorRef> (CartesianProduct::*)(VectorRef) const;

  bp::class_<CartesianProduct, bp::bases<Manifold>>(
      "CartesianProduct", "Cartesian product of two or more manifolds.",
      bp::no_init)
      .def(bp::init<>("self"_a))
      .def(bp::init<const std::vector<PolymorphicManifold> &>(
          ("self"_a, "spaces")))
      .def(bp::init<PolymorphicManifold, PolymorphicManifold>(
          ("self"_a, "left", "right")))
      .def(
          "getComponent",
          +[](CartesianProduct const &m, std::size_t i) -> const Manifold & {
            if (i >= m.numComponents()) {
              PyErr_SetString(PyExc_IndexError, "Index out of bounds.");
              bp::throw_error_already_set();
            }
            return m.getComponent(i);
          },
          bp::return_internal_reference<>(), ("self"_a, "i"),
          "Get the i-th component of the Cartesian product.")
      .def("addComponent", &CartesianProduct::addComponent<PolymorphicManifold>,
           ("self"_a, "c"), "Add a component to the Cartesian product.")
      .add_property("num_components", &CartesianProduct::numComponents,
                    "Get the number of components in the Cartesian product.")
      .def(
          "split",
          +[](CartesianProduct const &m, const ConstVectorRef &x) {
            return copy_vec_constref(m.split(x));
          },
          ("self"_a, "x"), split_doc.c_str())
      .def<MutSplitSig>(
          "split", &CartesianProduct::split, ("self"_a, "x"),
          (split_doc +
           " This returns a list of mutable references to each component.")
              .c_str())
      .def(
          "split_vector",
          +[](CartesianProduct const &m, const ConstVectorRef &x) {
            return copy_vec_constref(m.split_vector(x));
          },
          ("self"_a, "v"), split_vec_doc.c_str())
      .def<MutSplitSig>(
          "split_vector", &CartesianProduct::split_vector, ("self"_a, "v"),
          (split_vec_doc +
           " This returns a list of mutable references to each component.")
              .c_str())
      .def("merge", &CartesianProduct::merge, ("self"_a, "xs"),
           "Define a point on the manifold by merging points from each "
           "component.")
      .def("merge_vector", &CartesianProduct::merge_vector, ("self"_a, "vs"),
           "Define a tangent vector on the manifold by merging vectors from "
           "each component.")
      .def(
          "__mul__", +[](const CartesianProduct &a,
                         const CartesianProduct &b) { return a * b; })
      .def(PolymorphicVisitor<PolymorphicManifold>());
}

} // namespace aligator::python
