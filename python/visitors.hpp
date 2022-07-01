#pragma once

#include <boost/python.hpp>

namespace proxddp {
namespace python {
namespace bp = boost::python;

template <typename T>
struct ClonePythonVisitor : bp::def_visitor<ClonePythonVisitor<T>> {
  template <typename PyT> void visit(PyT &obj) const {
    obj.def("clone", &T::clone, bp::args("self"), "Clone the object.");
  }
};

template <typename T>
struct CreateDataPythonVisitor : bp::def_visitor<CreateDataPythonVisitor<T>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    obj.def("createData", &T::createData, bp::args("self"),
            "Create a data object.");
  }
};

template <typename T>
struct PrintableVisitor : bp::def_visitor<PrintableVisitor<T>> {
  template <typename Pyclass> void visit(Pyclass &obj) const {
    obj.def(bp::self_ns::str(bp::self)).def(bp::self_ns::repr(bp::self));
  }
};

} // namespace python
} // namespace proxddp
